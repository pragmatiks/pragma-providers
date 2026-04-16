"""Agno Runner resource - deploys agents, teams, and workflows to Kubernetes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from kubernetes_provider import (
    Deployment as KubernetesDeployment,
)
from kubernetes_provider import (
    DeploymentConfig as KubernetesDeploymentConfig,
)
from kubernetes_provider import (
    KubernetesConfig,
    Namespace,
    NamespaceOutputs,
    Service,
    ServiceConfig,
)
from kubernetes_provider.resources.deployment import (
    ContainerConfig,
    ContainerPortConfig,
    HttpGetConfig,
    ProbeConfig,
    ResourceRequirementsConfig,
)
from kubernetes_provider.resources.service import PortConfig
from pragma_sdk import (
    Config,
    Dependency,
    Field,
    HealthStatus,
    ImmutableDependency,
    LogEntry,
    Outputs,
    Resource,
)
from pydantic import model_validator

from agno_provider.resources.agent import Agent, AgentOutputs, AgentSpec
from agno_provider.resources.base import AgnoSpec
from agno_provider.resources.team import Team, TeamOutputs, TeamSpec


class RunnerSpec(AgnoSpec):
    """Specification for reconstructing a Runner configuration.

    Contains all runner configuration including the nested agent, team, and
    workflow specs. Used for tracking what was deployed.

    Attributes:
        name: Runner name (Kubernetes deployment name).
        namespace: Kubernetes namespace where the runner is deployed.
        agent_specs: Nested agent specs deployed on this runner.
        team_specs: Nested team specs deployed on this runner.
        workflow_specs: Nested workflow specs deployed on this runner.
        replicas: Number of pod replicas.
        image: Container image used for the runner pods.
        cpu: CPU resource request.
        memory: Memory resource request and limit.
        security_key: Whether a security key is configured for auth.
    """

    name: str
    namespace: str
    agent_specs: list[AgentSpec] = []
    team_specs: list[TeamSpec] = []
    workflow_specs: list[Any] = []
    replicas: int
    image: str
    cpu: str
    memory: str
    security_key: bool = False


class RunnerConfig(Config):
    """Configuration for deploying Agno agents, teams, and workflows to Kubernetes.

    A single runner can host multiple agents, teams, and workflows in one
    AgentOS instance. At least one entity across the three lists is required.

    Attributes:
        agents: Agent dependencies to deploy on this runner.
        teams: Team dependencies to deploy on this runner.
        workflows: Workflow dependencies to deploy on this runner. Must be empty
            in v1; reserved for when the Workflow resource type is introduced.
        config: Kubernetes config dependency providing cluster access.
        namespace: Kubernetes namespace dependency for the runner pods.
        replicas: Number of pod replicas. Defaults to 1.
        image: Container image for running the agents/teams/workflows.
        security_key: Bearer token for AgentOS basic auth (dev environments).
        jwt_verification_key: Public key for JWT/RBAC auth (production environments).
        public: Expose the service via LoadBalancer instead of ClusterIP. Defaults to False.
        cpu: CPU resource request (e.g., "200m", "1"). Defaults to "200m".
        memory: Memory resource request and limit (e.g., "1Gi", "2Gi"). Defaults to "1Gi".
    """

    agents: list[Dependency[Agent]] = []
    teams: list[Dependency[Team]] = []
    workflows: list[Any] = []

    config: ImmutableDependency[KubernetesConfig]
    namespace: Dependency[Namespace]
    replicas: Field[int] = 1
    image: Field[str] = "ghcr.io/pragmatiks/agno-runner:latest"
    security_key: Field[str] | None = None
    jwt_verification_key: Field[str] | None = None
    public: Field[bool] = False

    cpu: Field[str] = "200m"
    memory: Field[str] = "1Gi"

    @model_validator(mode="after")
    def validate_at_least_one_entity(self) -> RunnerConfig:
        """Validate that at least one agent, team, or workflow is provided.

        Returns:
            Self after validation.

        Raises:
            ValueError: If no entities are provided, or if workflows is non-empty
                (the Workflow resource type is not yet implemented).
        """
        if self.workflows:
            msg = (
                "workflows is reserved for a future release; pass an empty "
                "list until the Workflow resource type is introduced"
            )
            raise ValueError(msg)

        total = len(self.agents) + len(self.teams) + len(self.workflows)
        if total < 1:
            msg = "At least one agent, team, or workflow must be provided"
            raise ValueError(msg)

        return self


class RunnerOutputs(Outputs):
    """Outputs from Agno runner creation.

    Attributes:
        spec: Specification for the runner.
        url: In-cluster service URL.
        ready: Whether the runner is ready.
    """

    spec: RunnerSpec
    url: str
    ready: bool


class Runner(Resource[RunnerConfig, RunnerOutputs]):
    """Agno multi-entity runner on Kubernetes.

    This is the ONLY agno resource that creates infrastructure. It deploys
    one or more agents, teams, and workflows into a single AgentOS instance
    served by a Kubernetes Deployment + Service using child kubernetes
    provider resources.

    The container receives entity specs as JSON environment variables. The
    transport contract is versioned with the container image; see the runner
    package for the current wire format.

    Example YAML:
        provider: agno
        resource: runner
        name: my-agent-runner
        config:
          agents:
            - provider: agno
              resource: agent
              name: my-agent
          config:
            provider: kubernetes
            resource: config
            name: my-kubernetes-config
          namespace:
            provider: kubernetes
            resource: namespace
            name: agents
          replicas: 2
          security_key: "key-from-agentos-control-plane"

    Lifecycle:
        - on_create: Create child Kubernetes Deployment + Service, wait for ready
        - on_update: Update child Kubernetes resources, wait for ready
        - on_delete: Child resources cascade deleted via owner_references
    """

    def _runner_name(self) -> str:
        """Get Kubernetes deployment name based on resource name.

        Returns:
            Deployment name derived from resource name.
        """
        return f"agno-{self.name}"

    def _service_name(self) -> str:
        """Get Kubernetes service name.

        Returns:
            Service name derived from resource name.
        """
        return f"agno-{self.name}"

    def _labels(self) -> dict[str, str]:
        """Get labels for Kubernetes resources.

        Returns:
            Label dict for selecting pods.
        """
        return {
            "app": self._runner_name(),
            "agno.ai/managed-by": "pragma",
        }

    async def _resolve_namespace_name(self) -> str:
        """Resolve the namespace dependency and return the namespace name.

        Returns:
            Namespace name string from the resolved dependency.

        Raises:
            RuntimeError: If namespace dependency outputs are not available.
        """
        ns = await self.config.namespace.resolve()

        if ns.outputs is None:
            msg = "Namespace dependency outputs not available"
            raise RuntimeError(msg)

        if not isinstance(ns.outputs, NamespaceOutputs):
            msg = f"Expected NamespaceOutputs, got {type(ns.outputs).__name__}"
            raise RuntimeError(msg)

        return ns.outputs.name

    async def _resolve_entity_specs(self) -> tuple[list[AgentSpec], list[TeamSpec]]:
        """Resolve every agent and team dependency into their specs.

        Returns:
            Tuple of (agent_specs, team_specs) in declaration order.

        Raises:
            RuntimeError: If any dependency outputs are not available.
        """
        agent_specs: list[AgentSpec] = []
        for agent_dep in self.config.agents:
            agent = await agent_dep.resolve()

            if agent.outputs is None:
                msg = "Agent dependency outputs not available"
                raise RuntimeError(msg)

            assert isinstance(agent.outputs, AgentOutputs)
            agent_specs.append(agent.outputs.spec)

        team_specs: list[TeamSpec] = []
        for team_dep in self.config.teams:
            team = await team_dep.resolve()

            if team.outputs is None:
                msg = "Team dependency outputs not available"
                raise RuntimeError(msg)

            assert isinstance(team.outputs, TeamOutputs)
            team_specs.append(team.outputs.spec)

        return agent_specs, team_specs

    def _build_kubernetes_deployment(
        self,
        namespace_name: str,
        agent_specs: list[AgentSpec],
        team_specs: list[TeamSpec],
    ) -> KubernetesDeployment:
        """Build kubernetes/deployment child resource.

        Args:
            namespace_name: Resolved namespace name string.
            agent_specs: Agent specs to deploy on this runner.
            team_specs: Team specs to deploy on this runner.

        Returns:
            Kubernetes Deployment resource ready to apply.

        Raises:
            RuntimeError: If both agent_specs and team_specs are empty.
        """
        labels = self._labels()

        primary_spec: AgentSpec | TeamSpec | None = (
            agent_specs[0] if agent_specs else (team_specs[0] if team_specs else None)
        )
        primary_type = "agent" if agent_specs else "team"

        if primary_spec is None:
            msg = "Runner requires at least one agent or team spec"
            raise RuntimeError(msg)

        env = {
            "AGNO_SPEC_TYPE": primary_type,
            "AGNO_SPEC_JSON": primary_spec.model_dump_json(),
        }

        if self.config.security_key:
            env["OS_SECURITY_KEY"] = self.config.security_key

        if self.config.jwt_verification_key:
            env["JWT_VERIFICATION_KEY"] = self.config.jwt_verification_key

        container = ContainerConfig(
            name="agno",
            image=self.config.image,
            ports=[ContainerPortConfig(container_port=8000, name="http")],
            env=env,
            resources=ResourceRequirementsConfig(
                cpu=self.config.cpu,
                memory=self.config.memory,
                cpu_limit="1",
                memory_limit=self.config.memory,
            ),
            startup_probe=ProbeConfig(
                http_get=HttpGetConfig(path="/health", port=8000),
                period_seconds=2,
                failure_threshold=15,
                timeout_seconds=3,
            ),
            liveness_probe=ProbeConfig(
                http_get=HttpGetConfig(path="/health", port=8000),
                period_seconds=10,
                timeout_seconds=5,
                failure_threshold=3,
            ),
            readiness_probe=ProbeConfig(
                http_get=HttpGetConfig(path="/health", port=8000),
                period_seconds=5,
                timeout_seconds=3,
                failure_threshold=3,
            ),
        )

        config = KubernetesDeploymentConfig(
            config=self.config.config,
            namespace=namespace_name,
            replicas=self.config.replicas,
            selector=labels,
            labels=labels,
            containers=[container],
            strategy="RollingUpdate",
        )

        return KubernetesDeployment(
            name=self._runner_name(),
            config=config,
        )

    def _build_kubernetes_service(self, namespace_name: str) -> Service:
        """Build kubernetes/service child resource.

        Uses LoadBalancer when public is true, ClusterIP otherwise.

        Args:
            namespace_name: Resolved namespace name string.

        Returns:
            Kubernetes Service resource ready to apply.
        """
        labels = self._labels()
        service_type = "LoadBalancer" if self.config.public else "ClusterIP"

        config = ServiceConfig(
            config=self.config.config,
            namespace=namespace_name,
            type=service_type,
            selector=labels,
            ports=[
                PortConfig(name="http", port=80, target_port=8000),
            ],
        )

        return Service(
            name=self._service_name(),
            config=config,
        )

    def _build_kubernetes_deployment_for_delete(self, namespace_name: str) -> KubernetesDeployment:
        """Build minimal kubernetes/deployment for deletion.

        Creates a deployment resource with just enough config to call on_delete().
        The actual container/selector config doesn't matter for deletion.

        Args:
            namespace_name: Resolved namespace name string.

        Returns:
            Kubernetes Deployment resource for deletion.
        """
        labels = self._labels()

        container = ContainerConfig(
            name="agno",
            image=self.config.image,
        )

        config = KubernetesDeploymentConfig(
            config=self.config.config,
            namespace=namespace_name,
            replicas=1,
            selector=labels,
            containers=[container],
        )

        return KubernetesDeployment(
            name=self._runner_name(),
            config=config,
        )

    def _build_service_url(self, namespace_name: str) -> str:
        """Build in-cluster service URL.

        Args:
            namespace_name: Resolved namespace name string.

        Returns:
            In-cluster DNS URL for the service.
        """
        return f"http://{self._service_name()}.{namespace_name}.svc.cluster.local"

    def _build_outputs(
        self,
        namespace_name: str,
        agent_specs: list[AgentSpec],
        team_specs: list[TeamSpec],
        ready: bool,
    ) -> RunnerOutputs:
        """Build runner outputs.

        Args:
            namespace_name: Resolved namespace name string.
            agent_specs: Agent specs deployed on this runner.
            team_specs: Team specs deployed on this runner.
            ready: Whether runner is ready.

        Returns:
            RunnerOutputs with spec, url, and ready status.
        """
        runner_spec = RunnerSpec(
            name=self._runner_name(),
            namespace=namespace_name,
            agent_specs=agent_specs,
            team_specs=team_specs,
            workflow_specs=[],
            replicas=self.config.replicas,
            image=self.config.image,
            cpu=self.config.cpu,
            memory=self.config.memory,
            security_key=self.config.security_key is not None,
        )

        return RunnerOutputs(
            spec=runner_spec,
            url=self._build_service_url(namespace_name),
            ready=ready,
        )

    async def _apply_kubernetes_resources(
        self,
        namespace_name: str,
        agent_specs: list[AgentSpec],
        team_specs: list[TeamSpec],
    ) -> None:
        """Apply kubernetes deployment and service as child resources.

        Args:
            namespace_name: Resolved namespace name string.
            agent_specs: Agent specs to deploy on this runner.
            team_specs: Team specs to deploy on this runner.
        """
        kubernetes_deployment = self._build_kubernetes_deployment(namespace_name, agent_specs, team_specs)
        await kubernetes_deployment.apply()

        kubernetes_service = self._build_kubernetes_service(namespace_name)
        await kubernetes_service.apply()

    async def _kubernetes_deployment(self) -> KubernetesDeployment:
        """Get kubernetes deployment resource for current spec.

        Returns:
            Kubernetes Deployment resource configured for current entities.
        """
        namespace_name = await self._resolve_namespace_name()
        agent_specs, team_specs = await self._resolve_entity_specs()

        return self._build_kubernetes_deployment(namespace_name, agent_specs, team_specs)

    async def on_create(self) -> RunnerOutputs:
        """Create Kubernetes Deployment + Service.

        Returns:
            RunnerOutputs with runner details.
        """
        namespace_name = await self._resolve_namespace_name()
        agent_specs, team_specs = await self._resolve_entity_specs()

        await self._apply_kubernetes_resources(namespace_name, agent_specs, team_specs)

        return self._build_outputs(namespace_name, agent_specs, team_specs, ready=True)

    async def on_update(self, previous_config: RunnerConfig) -> RunnerOutputs:
        """Update Kubernetes Deployment + Service.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            RunnerOutputs with updated runner details.

        Raises:
            ValueError: If immutable fields changed.
        """
        if previous_config.config.id != self.config.config.id:
            msg = "Cannot change config; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.namespace.id != self.config.namespace.id:
            msg = "Cannot change namespace; delete and recreate resource"
            raise ValueError(msg)

        namespace_name = await self._resolve_namespace_name()
        agent_specs, team_specs = await self._resolve_entity_specs()

        await self._apply_kubernetes_resources(namespace_name, agent_specs, team_specs)

        return self._build_outputs(namespace_name, agent_specs, team_specs, ready=True)

    async def on_delete(self) -> None:
        """Delete Kubernetes Deployment + Service.

        Explicitly deletes child Kubernetes resources. Once cascade delete
        via owner_references is implemented, this can be simplified.
        """
        namespace_name = await self._resolve_namespace_name()

        kubernetes_service = self._build_kubernetes_service(namespace_name)
        await kubernetes_service.on_delete()

        kubernetes_deployment = self._build_kubernetes_deployment_for_delete(namespace_name)
        await kubernetes_deployment.on_delete()

    async def health(self) -> HealthStatus:
        """Check Runner health by delegating to child kubernetes/deployment.

        Returns:
            HealthStatus indicating healthy/degraded/unhealthy.
        """
        kubernetes_deployment = await self._kubernetes_deployment()
        return await kubernetes_deployment.health()

    async def logs(
        self,
        since: datetime | None = None,
        tail: int = 100,
    ) -> AsyncIterator[LogEntry]:
        """Fetch logs from pods managed by this Runner.

        Args:
            since: Only return logs after this timestamp.
            tail: Maximum number of log lines per pod.

        Yields:
            LogEntry for each log line from pods.
        """
        kubernetes_deployment = await self._kubernetes_deployment()

        async for entry in kubernetes_deployment.logs(since=since, tail=tail):
            yield entry

    @classmethod
    def upgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs

    @classmethod
    def downgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs
