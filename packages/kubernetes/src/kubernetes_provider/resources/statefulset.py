"""Kubernetes StatefulSet resource."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

from lightkube import ApiError
from lightkube.models.apps_v1 import StatefulSetPersistentVolumeClaimRetentionPolicy, StatefulSetSpec
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

from kubernetes_provider.objects import delete_object, fetch_object, wait_for_absence
from kubernetes_provider.resources.config import KubernetesConfig
from kubernetes_provider.rollout import build_replica_health, wait_for_rollout


class ContainerPortConfig(BaseModel):
    """Container port configuration for StatefulSet pods.

    Attributes:
        name: Optional port name for service discovery.
        container_port: Port number exposed by the container.
        protocol: Network protocol (TCP or UDP).
    """

    model_config = {"extra": "forbid"}

    name: str | None = None
    container_port: int
    protocol: Literal["TCP", "UDP"] = "TCP"


class EnvVarConfig(BaseModel):
    """Environment variable injected into a container.

    Attributes:
        name: Environment variable name.
        value: Environment variable value.
    """

    model_config = {"extra": "forbid"}

    name: str
    value: str


class VolumeMountConfig(BaseModel):
    """Volume mount attaching a PVC or volume to a container path.

    Attributes:
        name: Name of the volume (must match a volume_claim_template name).
        mount_path: Filesystem path inside the container where the volume is mounted.
        sub_path: Sub-path within the volume to mount.
        read_only: Whether the mount is read-only.
    """

    model_config = {"extra": "forbid"}

    name: str
    mount_path: str
    sub_path: str | None = None
    read_only: bool = False


class ResourcesConfig(BaseModel):
    """Container CPU and memory resource requirements.

    Attributes:
        requests: Resource requests (e.g., {"cpu": "500m", "memory": "1Gi"}).
        limits: Resource limits (e.g., {"cpu": "2000m", "memory": "4Gi"}).
    """

    model_config = {"extra": "forbid"}

    requests: dict[str, str] | None = None
    limits: dict[str, str] | None = None


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

    tcp_socket_port: int | None = None
    initial_delay_seconds: int = 10
    period_seconds: int = 10
    timeout_seconds: int = 5
    failure_threshold: int = 3


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

    name: str
    image: str
    ports: list[ContainerPortConfig] | None = None
    env: list[EnvVarConfig] | None = None
    volume_mounts: list[VolumeMountConfig] | None = None
    resources: ResourcesConfig | None = None
    command: list[str] | None = None
    args: list[str] | None = None
    liveness_probe: ProbeConfig | None = None
    readiness_probe: ProbeConfig | None = None


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

    name: str
    storage_class: str | None = None
    access_modes: list[str] = PydanticField(default_factory=lambda: ["ReadWriteOnce"])
    storage: str = "10Gi"


class StatefulSetConfig(Config):
    """Configuration for a Kubernetes StatefulSet.

    Attributes:
        config: Kubernetes config dependency providing cluster access.
        namespace: Kubernetes namespace (immutable after creation).
        replicas: Number of pod replicas to maintain.
        service_name: Name of the headless service for stable pod DNS (immutable after creation).
        selector: Label selector for pods; defaults to ``{"app": "<name>"}`` if not set (immutable after creation).
        containers: List of container specifications defining the pod template.
        volume_claim_templates: PVC templates for persistent storage per replica (immutable after creation).
    """

    config: ImmutableDependency[KubernetesConfig]
    namespace: ImmutableField[str] = "default"
    replicas: Field[int] = 1
    service_name: ImmutableField[str]
    selector: ImmutableField[dict[str, str]] | None = None
    containers: Field[list[ContainerConfig]]
    volume_claim_templates: ImmutableField[list[VolumeClaimTemplateConfig]] | None = None


class StatefulSetOutputs(Outputs):
    """Outputs from Kubernetes StatefulSet creation.

    Attributes:
        name: StatefulSet name as created in the cluster.
        namespace: Kubernetes namespace containing the statefulset.
        replicas: Desired number of pod replicas.
        service_name: Associated headless service name for pod DNS.
    """

    name: str
    namespace: str
    replicas: int
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
    to clean up owned pods, and the cluster then deletes the PersistentVolumeClaims
    created from ``volume_claim_templates``, with their data.

    Lifecycle:
        - on_create: Apply statefulset, wait for ready
        - on_observe: Look up the statefulset by name and namespace
        - on_update: Apply updated statefulset, wait for ready
        - on_delete: Delete statefulset with background cascade and wait until it is gone
    """

    @asynccontextmanager
    async def _get_client(self):
        """Yield lightkube client from the kubernetes config dependency.

        Yields:
            Lightkube async client configured for the target cluster.
        """
        cluster_config = await self.config.config.resolve()

        async with cluster_config.build_client() as client:
            yield client

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
            persistentVolumeClaimRetentionPolicy=StatefulSetPersistentVolumeClaimRetentionPolicy(whenDeleted="Delete"),
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

    def build_outputs(self, statefulset: K8sStatefulSet) -> StatefulSetOutputs:
        """Build outputs from the StatefulSet as read from the cluster.

        Args:
            statefulset: StatefulSet returned by the cluster.

        Returns:
            StatefulSetOutputs with statefulset details.
        """
        assert statefulset.metadata is not None
        assert statefulset.metadata.name is not None
        assert statefulset.metadata.namespace is not None
        assert statefulset.spec is not None
        assert statefulset.spec.serviceName is not None

        return StatefulSetOutputs(
            name=statefulset.metadata.name,
            namespace=statefulset.metadata.namespace,
            replicas=statefulset.spec.replicas or 0,
            service_name=statefulset.spec.serviceName,
        )

    async def on_create(self) -> StatefulSetOutputs:
        """Create Kubernetes StatefulSet and wait for ready.

        Idempotent: Uses apply() which handles both create and update.

        Returns:
            StatefulSetOutputs with statefulset details.

        Raises:
            TimeoutError: If the rollout does not complete within the rollout timeout.
        """
        async with self._get_client() as client:
            await client.apply(self._build_statefulset(), field_manager="pragma-kubernetes")

            statefulset = await wait_for_rollout(client, K8sStatefulSet, self.name, self.config.namespace)

        return self.build_outputs(statefulset)

    async def on_observe(self) -> StatefulSetOutputs | None:
        """Look up the Kubernetes StatefulSet named after this resource.

        Returns:
            StatefulSetOutputs, or ``None`` if the statefulset does not exist.
        """
        async with self._get_client() as client:
            statefulset = await fetch_object(client, K8sStatefulSet, self.name, self.config.namespace)

        if statefulset is None:
            return None

        return self.build_outputs(statefulset)

    async def on_update(self, previous_config: StatefulSetConfig | None) -> StatefulSetOutputs:
        """Apply the full desired statefulset configuration and wait for ready.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            StatefulSetOutputs with updated statefulset details.
        """
        return await self.on_create()

    async def on_delete(self) -> None:
        """Delete Kubernetes StatefulSet with background cascade and wait until it is gone.

        Idempotent: Succeeds if statefulset doesn't exist. The cluster then deletes the
        PersistentVolumeClaims created from ``volume_claim_templates``, with their data.

        Raises:
            TimeoutError: If the statefulset is still present after the deletion timeout.
        """
        async with self._get_client() as client:
            await delete_object(client, K8sStatefulSet, self.name, self.config.namespace)
            await wait_for_absence(client, K8sStatefulSet, self.name, self.config.namespace)

    async def health(self) -> HealthStatus:
        """Check StatefulSet health by comparing ready replicas to desired.

        Returns:
            HealthStatus indicating healthy/degraded/unhealthy.
        """
        async with self._get_client() as client:
            statefulset = await fetch_object(client, K8sStatefulSet, self.name, self.config.namespace)

        if statefulset is None:
            return HealthStatus(
                status="unhealthy",
                message=f"StatefulSet {self.name} not found",
            )

        return build_replica_health(statefulset)

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
