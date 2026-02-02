"""Agno Runner resource - deploys agents or teams to Kubernetes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import ClassVar, Literal

from gcp_provider import GKE
from kubernetes_provider import (
    Deployment as KubernetesDeployment,
)
from kubernetes_provider import (
    DeploymentConfig as KubernetesDeploymentConfig,
)
from kubernetes_provider import (
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
from pragma_sdk import Config, Dependency, HealthStatus, LogEntry, Outputs, Resource
from pydantic import model_validator

from agno_provider.resources.agent import Agent, AgentOutputs, AgentSpec
from agno_provider.resources.base import AgnoSpec
from agno_provider.resources.team import Team, TeamOutputs, TeamSpec


class RunnerSpec(AgnoSpec):
    """Specification for reconstructing a Runner configuration.

    Contains all runner configuration including the nested agent or team spec.
    Used for tracking what was deployed.

    Attributes:
        name: Runner name.
        namespace: Kubernetes namespace.
        agent_spec: Nested agent spec if deploying an agent.
        team_spec: Nested team spec if deploying a team.
        replicas: Number of replicas.
        image: Container image.
        cpu: CPU resource limit.
        memory: Memory resource limit.
    """

    name: str
    namespace: str
    agent_spec: AgentSpec | None = None
    team_spec: TeamSpec | None = None
    replicas: int
    image: str
    cpu: str
    memory: str


class RunnerConfig(Config):
    """Configuration for deploying an Agno agent or team to Kubernetes.

    Exactly one of agent or team must be provided.

    Attributes:
        agent: Agent dependency to deploy.
        team: Team dependency to deploy.
        cluster: GKE cluster dependency providing Kubernetes credentials.
        namespace: Kubernetes namespace for runner.
        replicas: Number of pod replicas.
        image: Container image for running the agent/team.
        cpu: CPU resource limit.
        memory: Memory resource limit.
    """

    agent: Dependency[Agent] | None = None
    team: Dependency[Team] | None = None

    cluster: Dependency[GKE]

    namespace: str = "default"
    replicas: int = 1
    image: str = "ghcr.io/pragmatiks/agno-runner:latest"

    cpu: str = "500m"
    memory: str = "1Gi"

    @model_validator(mode="after")
    def validate_exactly_one_target(self) -> RunnerConfig:
        """Validate that exactly one of agent or team is provided.

        Returns:
            Self after validation.

        Raises:
            ValueError: If neither or both of agent and team are provided.
        """
        has_agent = self.agent is not None
        has_team = self.team is not None

        if not has_agent and not has_team:
            msg = "Either agent or team must be provided"
            raise ValueError(msg)

        if has_agent and has_team:
            msg = "Only one of agent or team can be provided, not both"
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
    """Agno agent/team runner on Kubernetes.

    This is the ONLY agno resource that creates infrastructure. It deploys
    either an agent or a team as a Kubernetes Deployment + Service using
    child kubernetes provider resources.

    The container receives the agent/team spec as JSON environment variables:
    - AGNO_SPEC_TYPE: "agent" or "team"
    - AGNO_SPEC_JSON: JSON-serialized AgentSpec or TeamSpec

    The container image uses these to reconstruct the agent/team at startup.

    Example YAML:
        provider: agno
        resource: runner
        name: my-agent-runner
        config:
          agent:
            $ref: agno/agent/my-agent
          cluster:
            $ref: gcp/gke/my-cluster
          namespace: agents
          replicas: 2

    Lifecycle:
        - on_create: Create child Kubernetes Deployment + Service, wait for ready
        - on_update: Update child Kubernetes resources, wait for ready
        - on_delete: Child resources cascade deleted via owner_references
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "runner"

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

    async def _get_spec_info(self) -> tuple[Literal["agent", "team"], AgentSpec | TeamSpec]:
        """Get spec type and spec from resolved dependency.

        Returns:
            Tuple of (spec_type, spec).

        Raises:
            RuntimeError: If dependency outputs are not available.
        """
        if self.config.agent is not None:
            agent = await self.config.agent.resolve()

            if agent.outputs is None:
                msg = "Agent dependency outputs not available"
                raise RuntimeError(msg)

            assert isinstance(agent.outputs, AgentOutputs)
            return ("agent", agent.outputs.spec)

        if self.config.team is not None:
            team = await self.config.team.resolve()

            if team.outputs is None:
                msg = "Team dependency outputs not available"
                raise RuntimeError(msg)

            assert isinstance(team.outputs, TeamOutputs)
            return ("team", team.outputs.spec)

        msg = "Neither agent nor team dependency is set"
        raise RuntimeError(msg)

    def _build_kubernetes_deployment(
        self,
        spec_type: Literal["agent", "team"],
        spec: AgentSpec | TeamSpec,
    ) -> KubernetesDeployment:
        """Build kubernetes/deployment child resource.

        Args:
            spec_type: Type of spec ("agent" or "team").
            spec: The agent or team spec to deploy.

        Returns:
            Kubernetes Deployment resource ready to apply.
        """
        labels = self._labels()
        spec_json = spec.model_dump_json()

        container = ContainerConfig(
            name="agno",
            image=self.config.image,
            ports=[ContainerPortConfig(container_port=8000, name="http")],
            env={
                "AGNO_SPEC_TYPE": spec_type,
                "AGNO_SPEC_JSON": spec_json,
            },
            resources=ResourceRequirementsConfig(
                cpu=self.config.cpu,
                memory=self.config.memory,
                cpu_limit=self.config.cpu,
                memory_limit=self.config.memory,
            ),
            liveness_probe=ProbeConfig(
                http_get=HttpGetConfig(path="/health", port=8000),
                initial_delay_seconds=10,
                period_seconds=10,
                timeout_seconds=5,
                failure_threshold=3,
            ),
            readiness_probe=ProbeConfig(
                http_get=HttpGetConfig(path="/health", port=8000),
                initial_delay_seconds=5,
                period_seconds=5,
                timeout_seconds=3,
                failure_threshold=3,
            ),
        )

        config = KubernetesDeploymentConfig(
            cluster=self.config.cluster,
            namespace=self.config.namespace,
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

    def _build_kubernetes_service(self) -> Service:
        """Build kubernetes/service child resource (ClusterIP).

        Returns:
            Kubernetes Service resource ready to apply.
        """
        labels = self._labels()

        config = ServiceConfig(
            cluster=self.config.cluster,
            namespace=self.config.namespace,
            type="ClusterIP",
            selector=labels,
            ports=[
                PortConfig(name="http", port=80, target_port=8000),
            ],
        )

        return Service(
            name=self._service_name(),
            config=config,
        )

    def _build_kubernetes_deployment_for_delete(self) -> KubernetesDeployment:
        """Build minimal kubernetes/deployment for deletion.

        Creates a deployment resource with just enough config to call on_delete().
        The actual container/selector config doesn't matter for deletion.

        Returns:
            Kubernetes Deployment resource for deletion.
        """
        labels = self._labels()

        container = ContainerConfig(
            name="agno",
            image=self.config.image,
        )

        config = KubernetesDeploymentConfig(
            cluster=self.config.cluster,
            namespace=self.config.namespace,
            replicas=1,
            selector=labels,
            containers=[container],
        )

        return KubernetesDeployment(
            name=self._runner_name(),
            config=config,
        )

    def _build_service_url(self) -> str:
        """Build in-cluster service URL.

        Returns:
            In-cluster DNS URL for the service.
        """
        return f"http://{self._service_name()}.{self.config.namespace}.svc.cluster.local"

    def _build_outputs(
        self,
        spec: AgentSpec | TeamSpec,
        ready: bool,
    ) -> RunnerOutputs:
        """Build runner outputs.

        Args:
            spec: The agent or team spec deployed.
            ready: Whether runner is ready.

        Returns:
            RunnerOutputs with spec, url, and ready status.
        """
        runner_spec = RunnerSpec(
            name=self._runner_name(),
            namespace=self.config.namespace,
            agent_spec=spec if isinstance(spec, AgentSpec) else None,
            team_spec=spec if isinstance(spec, TeamSpec) else None,
            replicas=self.config.replicas,
            image=self.config.image,
            cpu=self.config.cpu,
            memory=self.config.memory,
        )

        return RunnerOutputs(
            spec=runner_spec,
            url=self._build_service_url(),
            ready=ready,
        )

    async def _apply_kubernetes_resources(
        self,
        spec_type: Literal["agent", "team"],
        spec: AgentSpec | TeamSpec,
    ) -> None:
        """Apply kubernetes deployment and service as child resources.

        Args:
            spec_type: Type of spec ("agent" or "team").
            spec: The agent or team spec to deploy.
        """
        kubernetes_deployment = self._build_kubernetes_deployment(spec_type, spec)
        await kubernetes_deployment.apply()
        await kubernetes_deployment.wait_ready(timeout=300.0)

        kubernetes_service = self._build_kubernetes_service()
        await kubernetes_service.apply()
        await kubernetes_service.wait_ready(timeout=60.0)

    async def _kubernetes_deployment(self) -> KubernetesDeployment:
        """Get kubernetes deployment resource for current spec.

        Returns:
            Kubernetes Deployment resource configured for current agent/team.
        """
        spec_type, spec = await self._get_spec_info()
        return self._build_kubernetes_deployment(spec_type, spec)

    async def on_create(self) -> RunnerOutputs:
        """Create Kubernetes Deployment + Service and wait for ready.

        Returns:
            RunnerOutputs with runner details.
        """
        spec_type, spec = await self._get_spec_info()
        await self._apply_kubernetes_resources(spec_type, spec)

        return self._build_outputs(spec, ready=True)

    async def on_update(self, previous_config: RunnerConfig) -> RunnerOutputs:
        """Update Kubernetes Deployment + Service and wait for ready.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            RunnerOutputs with updated runner details.

        Raises:
            ValueError: If immutable fields changed.
        """
        if previous_config.cluster.id != self.config.cluster.id:
            msg = "Cannot change cluster; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.namespace != self.config.namespace:
            msg = "Cannot change namespace; delete and recreate resource"
            raise ValueError(msg)

        spec_type, spec = await self._get_spec_info()
        await self._apply_kubernetes_resources(spec_type, spec)

        return self._build_outputs(spec, ready=True)

    async def on_delete(self) -> None:
        """Delete Kubernetes Deployment + Service.

        Explicitly deletes child Kubernetes resources. Once cascade delete
        via owner_references is implemented, this can be simplified.
        """
        kubernetes_service = self._build_kubernetes_service()
        await kubernetes_service.on_delete()

        kubernetes_deployment = self._build_kubernetes_deployment_for_delete()
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
