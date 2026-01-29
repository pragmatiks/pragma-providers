"""Kubernetes StatefulSet resource."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
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
)
from lightkube.models.meta_v1 import LabelSelector, ObjectMeta
from lightkube.resources.apps_v1 import StatefulSet as K8sStatefulSet
from lightkube.resources.core_v1 import Pod
from pydantic import BaseModel, Field
from pragma_sdk import Config, Dependency, HealthStatus, LogEntry, Outputs, Resource

from kubernetes_provider.client import create_client_from_gke


_POLL_INTERVAL_SECONDS = 5
_MAX_POLL_ATTEMPTS = 60


class ContainerPortConfig(BaseModel):
    """Container port configuration."""

    model_config = {"extra": "forbid"}

    name: str | None = None
    container_port: int
    protocol: Literal["TCP", "UDP"] = "TCP"


class EnvVarConfig(BaseModel):
    """Environment variable configuration."""

    model_config = {"extra": "forbid"}

    name: str
    value: str


class VolumeMountConfig(BaseModel):
    """Volume mount configuration."""

    model_config = {"extra": "forbid"}

    name: str
    mount_path: str
    sub_path: str | None = None
    read_only: bool = False


class ResourcesConfig(BaseModel):
    """Container resource requirements."""

    model_config = {"extra": "forbid"}

    requests: dict[str, str] | None = None
    limits: dict[str, str] | None = None


class ProbeConfig(BaseModel):
    """Container probe configuration."""

    model_config = {"extra": "forbid"}

    tcp_socket_port: int | None = None
    initial_delay_seconds: int = 10
    period_seconds: int = 10
    timeout_seconds: int = 5
    failure_threshold: int = 3


class ContainerConfig(BaseModel):
    """Container specification."""

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
    """PersistentVolumeClaim template for StatefulSet."""

    model_config = {"extra": "forbid"}

    name: str
    storage_class: str | None = None
    access_modes: list[str] = Field(default_factory=lambda: ["ReadWriteOnce"])
    storage: str = "10Gi"


class StatefulSetConfig(Config):
    """Configuration for a Kubernetes StatefulSet.

    Attributes:
        cluster: GKE cluster dependency providing Kubernetes credentials.
        namespace: Kubernetes namespace.
        replicas: Number of pod replicas.
        service_name: Name of the headless service for pod DNS.
        selector: Label selector for pods (defaults to app: {name}).
        containers: List of container specifications.
        volume_claim_templates: PVC templates for persistent storage.
    """

    cluster: Dependency[GKE]
    namespace: str = "default"
    replicas: int = 1
    service_name: str
    selector: dict[str, str] | None = None
    containers: list[ContainerConfig]
    volume_claim_templates: list[VolumeClaimTemplateConfig] | None = None


class StatefulSetOutputs(Outputs):
    """Outputs from Kubernetes StatefulSet creation.

    Attributes:
        name: StatefulSet name.
        namespace: Kubernetes namespace.
        replicas: Desired replicas.
        ready_replicas: Current ready replicas.
        service_name: Associated headless service name.
    """

    name: str
    namespace: str
    replicas: int
    ready_replicas: int
    service_name: str


class StatefulSet(Resource[StatefulSetConfig, StatefulSetOutputs]):
    """Kubernetes StatefulSet resource.

    Creates and manages Kubernetes StatefulSets with persistent storage.
    Waits for all replicas to be ready before returning.

    Lifecycle:
        - on_create: Apply statefulset, wait for ready
        - on_update: Apply updated statefulset, wait for ready
        - on_delete: Delete statefulset with cascade
    """

    provider: ClassVar[str] = "kubernetes"
    resource: ClassVar[str] = "statefulset"

    async def _get_client(self):
        """Get lightkube client from GKE cluster credentials."""
        cluster = await self.config.cluster.resolve()
        outputs = cluster.outputs

        if outputs is None:
            msg = "GKE cluster outputs not available"
            raise RuntimeError(msg)

        creds = cluster.config.credentials

        return create_client_from_gke(outputs, creds)

    def _build_probe(self, config: ProbeConfig) -> Probe | None:
        """Build probe from config."""
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
        """Build container from config."""
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
        """Build PVC template from config."""
        return PersistentVolumeClaim(
            metadata=ObjectMeta(name=config.name),
            spec=PersistentVolumeClaimSpec(
                storageClassName=config.storage_class,
                accessModes=config.access_modes,
                resources=ResourceRequirements(
                    requests={"storage": config.storage},
                ),
            ),
        )

    def _build_statefulset(self) -> K8sStatefulSet:
        """Build Kubernetes StatefulSet object from config."""
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
        """Build outputs from Kubernetes StatefulSet object."""
        ready = 0

        if sts.status and sts.status.readyReplicas:
            ready = sts.status.readyReplicas

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
        client = await self._get_client()
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

        Raises:
            ValueError: If immutable fields changed.
        """
        if previous_config.cluster.id != self.config.cluster.id:
            msg = "Cannot change cluster; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.namespace != self.config.namespace:
            msg = "Cannot change namespace; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.service_name != self.config.service_name:
            msg = "Cannot change service_name; delete and recreate resource"
            raise ValueError(msg)

        client = await self._get_client()
        sts = self._build_statefulset()

        await client.apply(sts, field_manager="pragma-kubernetes")

        result = await self._wait_for_ready(client)

        return self._build_outputs(result)

    async def on_delete(self) -> None:
        """Delete Kubernetes StatefulSet with cascade.

        Idempotent: Succeeds if statefulset doesn't exist.
        """
        client = await self._get_client()

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
        """
        client = await self._get_client()

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
        client = await self._get_client()
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
                    delta = datetime.now(timezone.utc) - since
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
                            timestamp=datetime.now(timezone.utc),
                            level="info",
                            message=line,
                            metadata={"pod": pod_name},
                        )

            except ApiError:
                yield LogEntry(
                    timestamp=datetime.now(timezone.utc),
                    level="warn",
                    message=f"Failed to fetch logs from pod {pod_name}",
                    metadata={"pod": pod_name},
                )
