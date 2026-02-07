"""Kubernetes Deployment resource."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import ClassVar, Literal

from gcp_provider import GKE
from lightkube import ApiError
from lightkube.models.apps_v1 import DeploymentSpec, DeploymentStrategy, RollingUpdateDeployment
from lightkube.models.core_v1 import (
    Container,
    ContainerPort,
    EnvVar,
    EnvVarSource,
    HTTPGetAction,
    PodSpec,
    PodTemplateSpec,
    Probe,
    ResourceRequirements,
    SecretKeySelector,
)
from lightkube.models.meta_v1 import LabelSelector, ObjectMeta
from lightkube.resources.apps_v1 import Deployment as K8sDeployment
from lightkube.resources.core_v1 import Pod
from pragma_sdk import Config, Dependency, HealthStatus, LogEntry, Outputs, Resource
from pydantic import BaseModel

from kubernetes_provider.client import create_client_from_gke


_POLL_INTERVAL_SECONDS = 5.0
_DEFAULT_TIMEOUT_SECONDS = 300.0


class HttpGetConfig(BaseModel):
    """HTTP GET probe configuration.

    Attributes:
        path: HTTP path to probe.
        port: Port to probe.
    """

    model_config = {"extra": "forbid"}

    path: str
    port: int


class ProbeConfig(BaseModel):
    """Container probe configuration.

    Attributes:
        http_get: HTTP GET probe settings.
        initial_delay_seconds: Delay before first probe.
        period_seconds: How often to probe.
        timeout_seconds: Probe timeout.
        failure_threshold: Failures before unhealthy.
    """

    model_config = {"extra": "forbid"}

    http_get: HttpGetConfig | None = None
    initial_delay_seconds: int = 0
    period_seconds: int = 10
    timeout_seconds: int = 1
    failure_threshold: int = 3


class ResourceRequirementsConfig(BaseModel):
    """Container resource requirements.

    Attributes:
        cpu: CPU request (e.g., "500m").
        memory: Memory request (e.g., "1Gi").
        cpu_limit: CPU limit.
        memory_limit: Memory limit.
    """

    model_config = {"extra": "forbid"}

    cpu: str | None = None
    memory: str | None = None
    cpu_limit: str | None = None
    memory_limit: str | None = None


class ContainerPortConfig(BaseModel):
    """Container port configuration.

    Attributes:
        container_port: Port number on the container.
        name: Optional name for the port.
        protocol: Protocol (TCP or UDP).
    """

    model_config = {"extra": "forbid"}

    container_port: int
    name: str | None = None
    protocol: Literal["TCP", "UDP"] = "TCP"


class ContainerConfig(BaseModel):
    """Container configuration.

    Attributes:
        name: Container name.
        image: Container image.
        ports: List of port configurations.
        env: Environment variables as key-value pairs.
        env_from_secret: Environment variables from secrets (env_name -> secret_name.key).
        command: Container command.
        args: Container arguments.
        resources: Resource requirements.
        liveness_probe: Liveness probe configuration.
        readiness_probe: Readiness probe configuration.
        startup_probe: Startup probe configuration.
    """

    model_config = {"extra": "forbid"}

    name: str
    image: str
    ports: list[ContainerPortConfig] | None = None
    env: dict[str, str] | None = None
    env_from_secret: dict[str, str] | None = None
    command: list[str] | None = None
    args: list[str] | None = None
    resources: ResourceRequirementsConfig | None = None
    liveness_probe: ProbeConfig | None = None
    readiness_probe: ProbeConfig | None = None
    startup_probe: ProbeConfig | None = None


class DeploymentConfig(Config):
    """Configuration for a Kubernetes Deployment.

    Attributes:
        cluster: GKE cluster dependency providing Kubernetes credentials.
        namespace: Kubernetes namespace for the deployment.
        replicas: Number of pod replicas.
        selector: Label selector for pods.
        labels: Pod labels (defaults to selector).
        containers: List of container specifications.
        strategy: Deployment strategy (RollingUpdate or Recreate).
    """

    cluster: Dependency[GKE]
    namespace: str = "default"
    replicas: int = 1
    selector: dict[str, str]
    labels: dict[str, str] | None = None
    containers: list[ContainerConfig]
    strategy: Literal["RollingUpdate", "Recreate"] = "RollingUpdate"


class DeploymentOutputs(Outputs):
    """Outputs from Kubernetes Deployment creation.

    Attributes:
        name: Deployment name.
        namespace: Kubernetes namespace.
        replicas: Desired replicas.
        ready_replicas: Current ready replicas.
        available_replicas: Current available replicas.
    """

    name: str
    namespace: str
    replicas: int
    ready_replicas: int
    available_replicas: int


class Deployment(Resource[DeploymentConfig, DeploymentOutputs]):
    """Kubernetes Deployment resource.

    Creates and manages Kubernetes Deployments using lightkube.
    Waits for all replicas to be ready before returning.

    Lifecycle:
        - on_create: Apply deployment, wait for ready
        - on_update: Apply updated deployment, wait for ready
        - on_delete: Delete the deployment
    """

    provider: ClassVar[str] = "kubernetes"
    resource: ClassVar[str] = "deployment"

    async def _get_client(self):
        """Get lightkube client from GKE cluster credentials.

        Returns:
            Lightkube async client configured for the GKE cluster.

        Raises:
            RuntimeError: If GKE cluster outputs are not available.
        """
        cluster = await self.config.cluster.resolve()
        outputs = cluster.outputs

        if outputs is None:
            msg = "GKE cluster outputs not available"
            raise RuntimeError(msg)

        creds = cluster.config.credentials

        return create_client_from_gke(outputs, creds)

    def _build_probe(self, config: ProbeConfig) -> Probe | None:
        """Build probe from config.

        Returns:
            Kubernetes Probe object or None if http_get not configured.
        """
        if config.http_get is None:
            return None

        return Probe(
            httpGet=HTTPGetAction(
                path=config.http_get.path,
                port=config.http_get.port,
            ),
            initialDelaySeconds=config.initial_delay_seconds,
            periodSeconds=config.period_seconds,
            timeoutSeconds=config.timeout_seconds,
            failureThreshold=config.failure_threshold,
        )

    def _build_container(self, config: ContainerConfig) -> Container:
        """Build container from config.

        Args:
            config: Container configuration.

        Returns:
            Kubernetes Container object.

        Raises:
            ValueError: If env_from_secret has invalid format (missing period).
        """
        container = Container(
            name=config.name,
            image=config.image,
        )

        if config.ports:
            container.ports = [
                ContainerPort(
                    name=p.name,
                    containerPort=p.container_port,
                    protocol=p.protocol,
                )
                for p in config.ports
            ]

        env_vars: list[EnvVar] = []

        if config.env:
            env_vars.extend([EnvVar(name=k, value=v) for k, v in config.env.items()])

        if config.env_from_secret:
            for env_name, secret_ref in config.env_from_secret.items():
                parts = secret_ref.rsplit(".", 1)

                if len(parts) != 2:
                    msg = f"Invalid secret reference '{secret_ref}': expected 'secret_name.key'"
                    raise ValueError(msg)

                secret_name, key = parts
                env_vars.append(
                    EnvVar(
                        name=env_name,
                        valueFrom=EnvVarSource(
                            secretKeyRef=SecretKeySelector(
                                name=secret_name,
                                key=key,
                            ),
                        ),
                    )
                )

        if env_vars:
            container.env = env_vars

        if config.command:
            container.command = config.command

        if config.args:
            container.args = config.args

        if config.resources:
            requests: dict[str, str] = {}
            limits: dict[str, str] = {}

            if config.resources.cpu:
                requests["cpu"] = config.resources.cpu

            if config.resources.memory:
                requests["memory"] = config.resources.memory

            if config.resources.cpu_limit:
                limits["cpu"] = config.resources.cpu_limit

            if config.resources.memory_limit:
                limits["memory"] = config.resources.memory_limit

            container.resources = ResourceRequirements(
                requests=requests or None,
                limits=limits or None,
            )

        if config.liveness_probe:
            container.livenessProbe = self._build_probe(config.liveness_probe)

        if config.readiness_probe:
            container.readinessProbe = self._build_probe(config.readiness_probe)

        if config.startup_probe:
            container.startupProbe = self._build_probe(config.startup_probe)

        return container

    def _build_deployment(self) -> K8sDeployment:
        """Build Kubernetes Deployment object from config.

        Returns:
            Kubernetes Deployment object ready to apply.
        """
        labels = self.config.labels or self.config.selector
        containers = [self._build_container(c) for c in self.config.containers]

        strategy = DeploymentStrategy(type=self.config.strategy)

        if self.config.strategy == "RollingUpdate":
            strategy.rollingUpdate = RollingUpdateDeployment(
                maxSurge="25%",
                maxUnavailable="25%",
            )

        spec = DeploymentSpec(
            replicas=self.config.replicas,
            selector=LabelSelector(matchLabels=self.config.selector),
            template=PodTemplateSpec(
                metadata=ObjectMeta(labels=labels),
                spec=PodSpec(containers=containers),
            ),
            strategy=strategy,
        )

        return K8sDeployment(
            metadata=ObjectMeta(
                name=self.name,
                namespace=self.config.namespace,
            ),
            spec=spec,
        )

    def _build_outputs(self, deployment: K8sDeployment) -> DeploymentOutputs:
        """Build outputs from Kubernetes Deployment object.

        Returns:
            DeploymentOutputs with deployment details.
        """
        ready = 0
        available = 0

        if deployment.status:
            if deployment.status.readyReplicas:
                ready = deployment.status.readyReplicas

            if deployment.status.availableReplicas:
                available = deployment.status.availableReplicas

        assert deployment.metadata is not None
        assert deployment.metadata.name is not None
        assert deployment.metadata.namespace is not None
        assert deployment.spec is not None

        return DeploymentOutputs(
            name=deployment.metadata.name,
            namespace=deployment.metadata.namespace,
            replicas=deployment.spec.replicas or 0,
            ready_replicas=ready,
            available_replicas=available,
        )

    async def _wait_for_ready(
        self,
        client,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> K8sDeployment:
        """Poll until Deployment has all replicas ready.

        Args:
            client: Lightkube async client.
            timeout: Maximum seconds to wait for ready state.

        Returns:
            Deployment with ready replicas.

        Raises:
            TimeoutError: If replicas don't become ready in time.
        """
        max_attempts = int(timeout / _POLL_INTERVAL_SECONDS)

        for _ in range(max_attempts):
            deployment = await client.get(
                K8sDeployment,
                name=self.name,
                namespace=self.config.namespace,
            )

            ready = 0

            if deployment.status and deployment.status.readyReplicas:
                ready = deployment.status.readyReplicas

            if ready >= self.config.replicas:
                return deployment

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        msg = f"Deployment {self.name} did not become ready within {timeout}s"
        raise TimeoutError(msg)

    async def on_create(self) -> DeploymentOutputs:
        """Create Kubernetes Deployment and wait for ready.

        Idempotent: Uses apply() which handles both create and update.

        Returns:
            DeploymentOutputs with deployment details.
        """
        client = await self._get_client()
        deployment = self._build_deployment()

        await client.apply(deployment, field_manager="pragma-kubernetes")

        result = await self._wait_for_ready(client)

        return self._build_outputs(result)

    async def on_update(self, previous_config: DeploymentConfig) -> DeploymentOutputs:
        """Update Kubernetes Deployment and wait for ready.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            DeploymentOutputs with updated deployment details.

        Raises:
            ValueError: If immutable fields changed.
        """
        if previous_config.cluster.id != self.config.cluster.id:
            msg = "Cannot change cluster; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.namespace != self.config.namespace:
            msg = "Cannot change namespace; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.selector != self.config.selector:
            msg = "Cannot change selector; delete and recreate resource"
            raise ValueError(msg)

        client = await self._get_client()
        deployment = self._build_deployment()

        await client.apply(deployment, field_manager="pragma-kubernetes")

        result = await self._wait_for_ready(client)

        return self._build_outputs(result)

    async def on_delete(self) -> None:
        """Delete Kubernetes Deployment.

        Idempotent: Succeeds if deployment doesn't exist.

        Raises:
            ApiError: If deletion fails for reasons other than not found.
        """
        client = await self._get_client()

        try:
            await client.delete(
                K8sDeployment,
                name=self.name,
                namespace=self.config.namespace,
            )
        except ApiError as e:
            if e.status.code != 404:
                raise

    async def health(self) -> HealthStatus:
        """Check Deployment health by comparing ready replicas to desired.

        Returns:
            HealthStatus indicating healthy/degraded/unhealthy.

        Raises:
            ApiError: If health check fails for reasons other than not found.
        """
        client = await self._get_client()

        try:
            deployment = await client.get(
                K8sDeployment,
                name=self.name,
                namespace=self.config.namespace,
            )
        except ApiError as e:
            if e.status.code == 404:
                return HealthStatus(
                    status="unhealthy",
                    message="Deployment not found",
                )
            raise

        ready = 0

        if deployment.status and deployment.status.readyReplicas:
            ready = deployment.status.readyReplicas

        desired = deployment.spec.replicas or 0

        if ready >= desired and desired > 0:
            return HealthStatus(
                status="healthy",
                message=f"All {ready} replicas ready",
                details={"ready_replicas": ready, "desired_replicas": desired},
            )

        if ready > 0:
            return HealthStatus(
                status="degraded",
                message=f"{ready}/{desired} replicas ready",
                details={"ready_replicas": ready, "desired_replicas": desired},
            )

        return HealthStatus(
            status="unhealthy",
            message=f"No replicas ready (desired: {desired})",
            details={"ready_replicas": 0, "desired_replicas": desired},
        )

    async def logs(
        self,
        since: datetime | None = None,
        tail: int = 100,
    ) -> AsyncIterator[LogEntry]:
        """Fetch logs from pods managed by this Deployment.

        Args:
            since: Only return logs after this timestamp.
            tail: Maximum number of log lines per pod.

        Yields:
            LogEntry for each log line from pods.
        """
        client = await self._get_client()
        label_selector = ",".join(f"{k}={v}" for k, v in self.config.selector.items())

        pods = client.list(
            Pod,
            namespace=self.config.namespace,
            labels=label_selector,
        )

        async for pod in pods:
            pod_name = pod.metadata.name

            try:
                since_seconds = None

                if since:
                    delta = datetime.now(UTC) - since
                    since_seconds = max(1, int(delta.total_seconds()))

                log_lines = await client.request(
                    "GET",
                    f"/api/v1/namespaces/{self.config.namespace}/pods/{pod_name}/log",
                    params={
                        "tailLines": tail,
                        **({"sinceSeconds": since_seconds} if since_seconds else {}),
                    },
                    response_type=str,
                )

                for line in log_lines.strip().split("\n"):
                    if line:
                        yield LogEntry(
                            timestamp=datetime.now(UTC),
                            level="info",
                            message=line,
                            metadata={"pod": pod_name},
                        )

            except ApiError:
                yield LogEntry(
                    timestamp=datetime.now(UTC),
                    level="warn",
                    message=f"Failed to fetch logs from pod {pod_name}",
                    metadata={"pod": pod_name},
                )
