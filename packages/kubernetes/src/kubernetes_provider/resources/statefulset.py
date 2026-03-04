"""Kubernetes StatefulSet resource."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import ClassVar, Literal

from gcp_provider import GKE
from lightkube import ApiError
from lightkube.core.client import CascadeType
from lightkube.models.apps_v1 import StatefulSetSpec
from lightkube.models.core_v1 import (
    Container,
    ContainerPort,
    EnvVar,
    PersistentVolumeClaim,
    PersistentVolumeClaimSpec,
    PodSpec,
    PodTemplateSpec,
    Probe,
    ResourceRequirements,
    TCPSocketAction,
    VolumeMount,
    VolumeResourceRequirements,
)
from lightkube.models.meta_v1 import LabelSelector, ObjectMeta
from lightkube.resources.apps_v1 import StatefulSet as K8sStatefulSet
from lightkube.resources.core_v1 import Pod
from pragma_sdk import Config, Field, HealthStatus, ImmutableDependency, ImmutableField, LogEntry, Outputs, Resource
from pydantic import BaseModel
from pydantic import Field as PydanticField

from kubernetes_provider.client import create_client_from_gke


_POLL_INTERVAL_SECONDS = 5
_MAX_POLL_ATTEMPTS = 60


class ContainerPortConfig(BaseModel):
    """Container port configuration for StatefulSet pods.

    Attributes:
        name: Optional port name for service discovery.
        container_port: Port number exposed by the container.
        protocol: Network protocol (TCP or UDP).
    """

    model_config = {"extra": "forbid"}

    name: str | None = PydanticField(default=None, description="Optional port name for service discovery.")
    container_port: int = PydanticField(description="Port number exposed by the container.")
    protocol: Literal["TCP", "UDP"] = PydanticField(default="TCP", description="Network protocol (TCP or UDP).")


class EnvVarConfig(BaseModel):
    """Environment variable injected into a container.

    Attributes:
        name: Environment variable name.
        value: Environment variable value.
    """

    model_config = {"extra": "forbid"}

    name: str = PydanticField(description="Environment variable name.")
    value: str = PydanticField(description="Environment variable value.")


class VolumeMountConfig(BaseModel):
    """Volume mount attaching a PVC or volume to a container path.

    Attributes:
        name: Name of the volume (must match a volume_claim_template name).
        mount_path: Filesystem path inside the container where the volume is mounted.
        sub_path: Sub-path within the volume to mount.
        read_only: Whether the mount is read-only.
    """

    model_config = {"extra": "forbid"}

    name: str = PydanticField(description="Name of the volume (must match a volume_claim_template name).")
    mount_path: str = PydanticField(description="Filesystem path inside the container.")
    sub_path: str | None = PydanticField(default=None, description="Sub-path within the volume to mount.")
    read_only: bool = PydanticField(default=False, description="Whether the mount is read-only.")


class ResourcesConfig(BaseModel):
    """Container CPU and memory resource requirements.

    Attributes:
        requests: Resource requests (e.g., {"cpu": "500m", "memory": "1Gi"}).
        limits: Resource limits (e.g., {"cpu": "2000m", "memory": "4Gi"}).
    """

    model_config = {"extra": "forbid"}

    requests: dict[str, str] | None = PydanticField(
        default=None, description='Resource requests (e.g., {"cpu": "500m", "memory": "1Gi"}).'
    )
    limits: dict[str, str] | None = PydanticField(
        default=None, description='Resource limits (e.g., {"cpu": "2000m", "memory": "4Gi"}).'
    )


class ProbeConfig(BaseModel):
    """Container health probe using TCP socket checks.

    Attributes:
        tcp_socket_port: Port to probe via TCP connection.
        initial_delay_seconds: Delay before the first probe after container start.
        period_seconds: Interval between probes.
        timeout_seconds: Timeout for each probe attempt.
        failure_threshold: Consecutive failures before marking unhealthy.
    """

    model_config = {"extra": "forbid"}

    tcp_socket_port: int | None = PydanticField(default=None, description="Port to probe via TCP connection.")
    initial_delay_seconds: int = PydanticField(default=10, description="Delay in seconds before the first probe.")
    period_seconds: int = PydanticField(default=10, description="Interval in seconds between probes.")
    timeout_seconds: int = PydanticField(default=5, description="Timeout in seconds for each probe attempt.")
    failure_threshold: int = PydanticField(default=3, description="Consecutive failures before marking unhealthy.")


class ContainerConfig(BaseModel):
    """Container specification for a StatefulSet pod.

    Attributes:
        name: Container name (unique within the pod).
        image: Container image including tag (e.g., "postgres:16").
        ports: Ports exposed by the container.
        env: Environment variables as name-value pairs.
        volume_mounts: Volume mounts attaching PVCs to container paths.
        resources: CPU and memory resource requests and limits.
        command: Override the container entrypoint.
        args: Arguments passed to the entrypoint.
        liveness_probe: Probe to detect if the container is alive.
        readiness_probe: Probe to detect if the container is ready for traffic.
    """

    model_config = {"extra": "forbid"}

    name: str = PydanticField(description="Container name (unique within the pod).")
    image: str = PydanticField(description='Container image including tag (e.g., "postgres:16").')
    ports: list[ContainerPortConfig] | None = PydanticField(default=None, description="Ports exposed by the container.")
    env: list[EnvVarConfig] | None = PydanticField(
        default=None, description="Environment variables as name-value pairs."
    )
    volume_mounts: list[VolumeMountConfig] | None = PydanticField(
        default=None, description="Volume mounts attaching PVCs to container paths."
    )
    resources: ResourcesConfig | None = PydanticField(
        default=None, description="CPU and memory resource requests and limits."
    )
    command: list[str] | None = PydanticField(default=None, description="Override the container entrypoint.")
    args: list[str] | None = PydanticField(default=None, description="Arguments passed to the entrypoint.")
    liveness_probe: ProbeConfig | None = PydanticField(
        default=None, description="Probe to detect if the container is alive."
    )
    readiness_probe: ProbeConfig | None = PydanticField(
        default=None, description="Probe to detect if the container is ready for traffic."
    )


class VolumeClaimTemplateConfig(BaseModel):
    """PersistentVolumeClaim template for StatefulSet persistent storage.

    Each pod replica gets its own PVC from this template, providing stable
    storage that survives pod restarts.

    Attributes:
        name: PVC name (referenced by volume_mounts in containers).
        storage_class: Kubernetes StorageClass name (e.g., "premium-rwo").
        access_modes: Volume access modes.
        storage: Storage capacity (e.g., "10Gi", "50Gi").
    """

    model_config = {"extra": "forbid"}

    name: str = PydanticField(description="PVC name (referenced by volume_mounts in containers).")
    storage_class: str | None = PydanticField(
        default=None, description='Kubernetes StorageClass name (e.g., "premium-rwo").'
    )
    access_modes: list[str] = PydanticField(
        default_factory=lambda: ["ReadWriteOnce"], description="Volume access modes."
    )
    storage: str = PydanticField(default="10Gi", description='Storage capacity (e.g., "10Gi", "50Gi").')


class StatefulSetConfig(Config):
    """Configuration for a Kubernetes StatefulSet.

    Attributes:
        cluster: GKE cluster dependency providing Kubernetes credentials.
        namespace: Kubernetes namespace (immutable after creation).
        replicas: Number of pod replicas to maintain.
        service_name: Name of the headless service for stable pod DNS (immutable after creation).
        selector: Label selector for pods; defaults to ``{"app": "<name>"}`` if not set.
        containers: List of container specifications defining the pod template.
        volume_claim_templates: PVC templates for persistent storage per replica.
    """

    cluster: ImmutableDependency[GKE]
    namespace: ImmutableField[str] = "default"
    replicas: Field[int] = 1
    service_name: ImmutableField[str]
    selector: Field[dict[str, str]] | None = None
    containers: Field[list[ContainerConfig]]
    volume_claim_templates: Field[list[VolumeClaimTemplateConfig]] | None = None


class StatefulSetOutputs(Outputs):
    """Outputs from Kubernetes StatefulSet creation.

    Attributes:
        name: StatefulSet name as created in the cluster.
        namespace: Kubernetes namespace containing the statefulset.
        replicas: Desired number of pod replicas.
        ready_replicas: Number of pods that have passed readiness checks.
        service_name: Associated headless service name for pod DNS.
    """

    name: str
    namespace: str
    replicas: int
    ready_replicas: int
    service_name: str


class StatefulSet(Resource[StatefulSetConfig, StatefulSetOutputs]):
    """Kubernetes StatefulSet resource.

    Manages stateful workloads with stable pod identity, persistent storage
    via PVC templates, and ordered deployment. Each pod gets a predictable
    hostname (e.g., ``postgres-0``, ``postgres-1``) and its own PersistentVolumeClaim
    that survives pod restarts.

    Requires a headless Service (``service_name``) for DNS-based pod discovery.
    Waits for all replicas to reach ready state before reporting success
    (polls every 5s, max 300s).

    Uses server-side apply with ``field_manager="pragma-kubernetes"`` for
    idempotent create and update operations. Deletes use background cascade
    to clean up owned pods and PVCs.

    Lifecycle:
        - on_create: Apply statefulset, wait for ready
        - on_update: Apply updated statefulset, wait for ready
        - on_delete: Delete statefulset with background cascade
    """

    provider: ClassVar[str] = "kubernetes"
    resource: ClassVar[str] = "statefulset"

    @asynccontextmanager
    async def _get_client(self):
        """Get lightkube client from GKE cluster credentials.

        Yields:
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
        client = create_client_from_gke(outputs, creds)

        try:
            yield client
        finally:
            await client.close()

    def _build_probe(self, config: ProbeConfig) -> Probe | None:
        """Build probe from config.

        Returns:
            Kubernetes Probe object or None if tcp_socket_port not configured.
        """
        if config.tcp_socket_port is None:
            return None

        return Probe(
            tcpSocket=TCPSocketAction(port=config.tcp_socket_port),
            initialDelaySeconds=config.initial_delay_seconds,
            periodSeconds=config.period_seconds,
            timeoutSeconds=config.timeout_seconds,
            failureThreshold=config.failure_threshold,
        )

    def _build_container(self, config: ContainerConfig) -> Container:
        """Build container from config.

        Returns:
            Kubernetes Container object.
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

        if config.env:
            container.env = [EnvVar(name=e.name, value=e.value) for e in config.env]

        if config.volume_mounts:
            container.volumeMounts = [
                VolumeMount(
                    name=vm.name,
                    mountPath=vm.mount_path,
                    subPath=vm.sub_path,
                    readOnly=vm.read_only,
                )
                for vm in config.volume_mounts
            ]

        if config.resources:
            container.resources = ResourceRequirements(
                requests=config.resources.requests,
                limits=config.resources.limits,
            )

        if config.command:
            container.command = config.command

        if config.args:
            container.args = config.args

        if config.liveness_probe:
            container.livenessProbe = self._build_probe(config.liveness_probe)

        if config.readiness_probe:
            container.readinessProbe = self._build_probe(config.readiness_probe)

        return container

    def _build_pvc_template(self, config: VolumeClaimTemplateConfig) -> PersistentVolumeClaim:
        """Build PVC template from config.

        Returns:
            Kubernetes PersistentVolumeClaim object.
        """
        return PersistentVolumeClaim(
            metadata=ObjectMeta(name=config.name),
            spec=PersistentVolumeClaimSpec(
                storageClassName=config.storage_class,
                accessModes=config.access_modes,
                resources=VolumeResourceRequirements(
                    requests={"storage": config.storage},
                ),
            ),
        )

    def _build_statefulset(self) -> K8sStatefulSet:
        """Build Kubernetes StatefulSet object from config.

        Returns:
            Kubernetes StatefulSet object ready to apply.
        """
        labels = self.config.selector or {"app": self.name}

        containers = [self._build_container(c) for c in self.config.containers]

        spec = StatefulSetSpec(
            replicas=self.config.replicas,
            serviceName=self.config.service_name,
            selector=LabelSelector(matchLabels=labels),
            template=PodTemplateSpec(
                metadata=ObjectMeta(labels=labels),
                spec=PodSpec(containers=containers),
            ),
        )

        if self.config.volume_claim_templates:
            spec.volumeClaimTemplates = [self._build_pvc_template(t) for t in self.config.volume_claim_templates]

        return K8sStatefulSet(
            metadata=ObjectMeta(
                name=self.name,
                namespace=self.config.namespace,
            ),
            spec=spec,
        )

    def _build_outputs(self, sts: K8sStatefulSet) -> StatefulSetOutputs:
        """Build outputs from Kubernetes StatefulSet object.

        Returns:
            StatefulSetOutputs with statefulset details.
        """
        ready = 0

        if sts.status and sts.status.readyReplicas:
            ready = sts.status.readyReplicas

        assert sts.metadata is not None
        assert sts.metadata.name is not None
        assert sts.metadata.namespace is not None
        assert sts.spec is not None
        assert sts.spec.serviceName is not None

        return StatefulSetOutputs(
            name=sts.metadata.name,
            namespace=sts.metadata.namespace,
            replicas=sts.spec.replicas or 0,
            ready_replicas=ready,
            service_name=sts.spec.serviceName,
        )

    async def _wait_for_ready(self, client) -> K8sStatefulSet:
        """Poll until StatefulSet has all replicas ready.

        Args:
            client: Lightkube async client.

        Returns:
            StatefulSet with ready replicas.

        Raises:
            TimeoutError: If replicas don't become ready in time.
        """
        for _ in range(_MAX_POLL_ATTEMPTS):
            sts = await client.get(
                K8sStatefulSet,
                name=self.name,
                namespace=self.config.namespace,
            )

            ready = 0
            if sts.status and sts.status.readyReplicas:
                ready = sts.status.readyReplicas

            if ready >= self.config.replicas:
                return sts

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        msg = f"StatefulSet {self.name} did not become ready within {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS}s"
        raise TimeoutError(msg)

    async def on_create(self) -> StatefulSetOutputs:
        """Create Kubernetes StatefulSet and wait for ready.

        Idempotent: Uses apply() which handles both create and update.

        Returns:
            StatefulSetOutputs with statefulset details.
        """
        async with self._get_client() as client:
            sts = self._build_statefulset()

            await client.apply(sts, field_manager="pragma-kubernetes")

            result = await self._wait_for_ready(client)

            return self._build_outputs(result)

    async def on_update(self, previous_config: StatefulSetConfig) -> StatefulSetOutputs:
        """Update Kubernetes StatefulSet and wait for ready.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            StatefulSetOutputs with updated statefulset details.
        """
        async with self._get_client() as client:
            sts = self._build_statefulset()

            await client.apply(sts, field_manager="pragma-kubernetes")

            result = await self._wait_for_ready(client)

            return self._build_outputs(result)

    async def on_delete(self) -> None:
        """Delete Kubernetes StatefulSet with cascade.

        Idempotent: Succeeds if statefulset doesn't exist.

        Raises:
            ApiError: If deletion fails for reasons other than not found.
        """
        async with self._get_client() as client:
            try:
                await client.delete(
                    K8sStatefulSet,
                    name=self.name,
                    namespace=self.config.namespace,
                    cascade=CascadeType.BACKGROUND,
                )
            except ApiError as e:
                if e.status.code != 404:
                    raise

    async def health(self) -> HealthStatus:
        """Check StatefulSet health by comparing ready replicas to desired.

        Returns:
            HealthStatus indicating healthy/degraded/unhealthy.

        Raises:
            ApiError: If health check fails for reasons other than not found.
        """
        async with self._get_client() as client:
            try:
                sts = await client.get(
                    K8sStatefulSet,
                    name=self.name,
                    namespace=self.config.namespace,
                )
            except ApiError as e:
                if e.status.code == 404:
                    return HealthStatus(
                        status="unhealthy",
                        message="StatefulSet not found",
                    )
                raise

            ready = 0

            if sts.status and sts.status.readyReplicas:
                ready = sts.status.readyReplicas

            desired = sts.spec.replicas or 0

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
        """Fetch logs from pods managed by this StatefulSet.

        Args:
            since: Only return logs after this timestamp.
            tail: Maximum number of log lines per pod.

        Yields:
            LogEntry for each log line from pods.
        """
        async with self._get_client() as client:
            labels = self.config.selector or {"app": self.name}
            label_selector = ",".join(f"{k}={v}" for k, v in labels.items())

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
