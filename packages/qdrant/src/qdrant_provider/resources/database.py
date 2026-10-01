"""Qdrant Database resource - deploys Qdrant to Kubernetes using Kubernetes resources."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from kubernetes_provider import (
    KubernetesConfig,
    Service,
    ServiceConfig,
    StatefulSet,
    StatefulSetConfig,
)
from kubernetes_provider.resources.service import PortConfig
from kubernetes_provider.resources.statefulset import (
    ContainerConfig,
    ContainerPortConfig,
    EnvVarConfig,
    ProbeConfig,
    ResourcesConfig,
    VolumeClaimTemplateConfig,
    VolumeMountConfig,
)
from lightkube import ApiError
from lightkube.resources.core_v1 import Service as K8sService
from pragma_sdk import Config, Field, HealthStatus, ImmutableDependency, LogEntry, Outputs, Resource
from pydantic import BaseModel, model_validator
from pydantic import Field as PydanticField


LOAD_BALANCER_POLL_INTERVAL_SECONDS = 5
OPERATION_BUDGET_SECONDS = 1140


class StorageConfig(BaseModel):
    """Storage configuration for Qdrant database.

    Attributes:
        size: Persistent volume size (e.g., "10Gi").
        class_: Kubernetes storage class name.
    """

    model_config = {"populate_by_name": True}

    size: str = "10Gi"
    class_: str = PydanticField(default="standard-rwo", alias="class")


class ResourceConfig(BaseModel):
    """Resource limits for Qdrant database.

    Attributes:
        memory: Memory limit (e.g., "2Gi").
        cpu: CPU limit (e.g., "1" or "500m").
    """

    memory: str = "2Gi"
    cpu: str = "1"


class DatabaseConfig(Config):
    """Configuration for a Qdrant database deployment.

    Attributes:
        config: Kubernetes config dependency providing cluster access.
        replicas: Number of Qdrant replicas (StatefulSet pods).
        image: Docker image for Qdrant (default: qdrant/qdrant:latest).
        storage: Persistent storage configuration.
        resources: CPU and memory limits.
        api_key: API key for Qdrant authentication. If provided, used directly.
        generate_api_key: If True and api_key is None, generates a secure 32-char key.
    """

    config: ImmutableDependency[KubernetesConfig]
    replicas: Field[int] = 1
    image: Field[str] = "qdrant/qdrant:latest"
    storage: Field[StorageConfig] | None = None
    resources: Field[ResourceConfig] | None = None
    api_key: Field[str] | None = None
    generate_api_key: Field[bool] = False

    @model_validator(mode="after")
    def validate_api_key_options(self) -> DatabaseConfig:
        """Validate that api_key and generate_api_key are mutually exclusive.

        Returns:
            The validated config.

        Raises:
            ValueError: If both api_key and generate_api_key are set.
        """
        if self.api_key is not None and self.generate_api_key:
            msg = "Cannot set both 'api_key' and 'generate_api_key'; use one or the other"
            raise ValueError(msg)

        return self


class DatabaseOutputs(Outputs):
    """Outputs from Qdrant database deployment.

    Attributes:
        url: HTTP endpoint for Qdrant REST API (external LoadBalancer URL).
        grpc_url: gRPC endpoint for Qdrant (external LoadBalancer URL).
        api_key: The API key for authentication (if configured).
    """

    url: str
    grpc_url: str
    api_key: str | None


class Database(Resource[DatabaseConfig, DatabaseOutputs]):
    """Qdrant database deployed to Kubernetes via child resources.

    Creates child resources:
    - Headless Service for pod DNS
    - StatefulSet with persistent storage
    - Client Service for cluster access

    Lifecycle:
        - on_create: Create child resources, wait for ready
        - on_update: Update child resources
        - on_delete: Delete child resources and wait until each is gone
    """

    computed = True

    _resolved_api_key: str | None = None

    def _resolve_api_key(self) -> str | None:
        """Resolve the API key from config.

        Returns:
            - The provided api_key if set
            - A generated 32-char hex key if generate_api_key is True
            - None otherwise
        """
        if self._resolved_api_key is not None:
            return self._resolved_api_key

        if self.config.api_key is not None:
            self._resolved_api_key = self.config.api_key
            return self._resolved_api_key

        if self.config.generate_api_key:
            self._resolved_api_key = secrets.token_hex(16)
            return self._resolved_api_key

        return None

    def _headless_service_name(self) -> str:
        """Get headless service name for pod DNS.

        Returns:
            The headless service name.
        """
        return f"qdrant-{self.name}-headless"

    def _client_service_name(self) -> str:
        """Get client service name for cluster access.

        Returns:
            The client service name.
        """
        return f"qdrant-{self.name}"

    def _statefulset_name(self) -> str:
        """Get StatefulSet name.

        Returns:
            The StatefulSet name.
        """
        return f"qdrant-{self.name}"

    def _namespace(self) -> str:
        """Get Kubernetes namespace.

        Returns:
            The Kubernetes namespace.
        """
        return "default"

    def _labels(self) -> dict[str, str]:
        """Get labels for Kubernetes resources.

        Returns:
            Dictionary of Kubernetes labels.
        """
        return {
            "app": "qdrant",
            "app.kubernetes.io/name": "qdrant",
            "app.kubernetes.io/instance": self.name,
        }

    @asynccontextmanager
    async def _get_client(self):
        """Yield lightkube client from the kubernetes config dependency.

        Yields:
            Lightkube async client, closed on exit.
        """
        cluster_config = await self.config.config.resolve()

        async with cluster_config.build_client() as client:
            yield client

    async def wait_for_load_balancer_address(self, deadline: float) -> str:
        """Wait for the client Service's LoadBalancer to receive an external address.

        Args:
            deadline: Time on the ``time.monotonic`` clock by which the address must be assigned.

        Returns:
            External IP address, or hostname when the load balancer reports no IP.

        Raises:
            ApiError: If reading the Service fails for any reason other than not found.
            TimeoutError: If no address is assigned by the deadline; the message names the Service.
        """
        namespace = self._namespace()
        service_name = self._client_service_name()

        async with self._get_client() as client:
            while True:
                try:
                    service = await client.get(K8sService, name=service_name, namespace=namespace)
                except ApiError as error:
                    if error.status.code != 404:
                        raise
                    service = None

                if service is None:
                    last_status = "service not found"
                elif service.status and service.status.loadBalancer and service.status.loadBalancer.ingress:
                    ingress = service.status.loadBalancer.ingress[0]

                    if ingress.ip:
                        return ingress.ip

                    if ingress.hostname:
                        return ingress.hostname

                    last_status = "ingress without ip or hostname"
                else:
                    last_status = "no ingress"

                if time.monotonic() >= deadline:
                    break

                await asyncio.sleep(LOAD_BALANCER_POLL_INTERVAL_SECONDS)

        message = (
            f"LoadBalancer Service {namespace}/{service_name} did not receive an external address "
            f"by the operation deadline; last status: {last_status}"
        )
        raise TimeoutError(message)

    def _build_headless_service(self) -> Service:
        """Build headless service for pod DNS.

        Returns:
            Configured Service resource.
        """
        config = ServiceConfig(
            config=self.config.config,
            namespace=self._namespace(),
            type="Headless",
            selector=self._labels(),
            ports=[
                PortConfig(name="rest", port=6333, target_port=6333),
                PortConfig(name="grpc", port=6334, target_port=6334),
            ],
        )

        return Service(
            project_id=self.project_id,
            name=self._headless_service_name(),
            config=config,
        )

    def _build_client_service(self) -> Service:
        """Build client service with LoadBalancer for external access.

        Returns:
            Configured Service resource.
        """
        config = ServiceConfig(
            config=self.config.config,
            namespace=self._namespace(),
            type="LoadBalancer",
            selector=self._labels(),
            ports=[
                PortConfig(name="rest", port=6333, target_port=6333),
                PortConfig(name="grpc", port=6334, target_port=6334),
            ],
        )

        return Service(
            project_id=self.project_id,
            name=self._client_service_name(),
            config=config,
        )

    def _build_statefulset(self) -> StatefulSet:
        """Build StatefulSet for Qdrant pods.

        Returns:
            Configured StatefulSet resource.
        """
        labels = self._labels()

        resources_config = None
        if self.config.resources:
            resources_config = ResourcesConfig(
                requests={
                    "memory": self.config.resources.memory,
                    "cpu": self.config.resources.cpu,
                },
                limits={
                    "memory": self.config.resources.memory,
                    "cpu": self.config.resources.cpu,
                },
            )

        volume_mounts = [
            VolumeMountConfig(name="qdrant-storage", mount_path="/qdrant/storage"),
        ]

        env_vars = None
        api_key = self._resolve_api_key()
        if api_key:
            env_vars = [
                EnvVarConfig(name="QDRANT__SERVICE__API_KEY", value=api_key),
            ]

        container = ContainerConfig(
            name="qdrant",
            image=self.config.image,
            ports=[
                ContainerPortConfig(name="rest", container_port=6333),
                ContainerPortConfig(name="grpc", container_port=6334),
            ],
            env=env_vars,
            volume_mounts=volume_mounts,
            resources=resources_config,
            readiness_probe=ProbeConfig(tcp_socket_port=6333),
            liveness_probe=ProbeConfig(tcp_socket_port=6333, initial_delay_seconds=30),
        )

        storage_class = None
        storage_size = "10Gi"

        if self.config.storage:
            storage_class = self.config.storage.class_
            storage_size = self.config.storage.size

        pvc_template = VolumeClaimTemplateConfig(
            name="qdrant-storage",
            storage_class=storage_class,
            storage=storage_size,
        )

        config = StatefulSetConfig(
            config=self.config.config,
            namespace=self._namespace(),
            replicas=self.config.replicas,
            service_name=self._headless_service_name(),
            selector=labels,
            containers=[container],
            volume_claim_templates=[pvc_template],
        )

        return StatefulSet(
            project_id=self.project_id,
            name=self._statefulset_name(),
            config=config,
        )

    async def fetch_outputs(self, deadline: float) -> DatabaseOutputs:
        """Wait for the client Service's LoadBalancer address and return the outputs.

        Args:
            deadline: Time on the ``time.monotonic`` clock by which the address must be assigned.

        Returns:
            DatabaseOutputs with external URLs and API key.

        Raises:
            TimeoutError: If the LoadBalancer gets no external address by the deadline.
        """
        external_address = await self.wait_for_load_balancer_address(deadline)

        url = f"http://{external_address}:6333"
        grpc_url = f"http://{external_address}:6334"

        return DatabaseOutputs(
            url=url,
            grpc_url=grpc_url,
            api_key=self._resolve_api_key(),
        )

    def is_statefulset_changed(self, previous_config: DatabaseConfig) -> bool:
        """Tell whether the StatefulSet built from this configuration differs from the previous one.

        Args:
            previous_config: The configuration last applied.

        Returns:
            ``True`` when a StatefulSet setting changed or the API key is generated anew on every apply.
        """
        if self.config.generate_api_key:
            return True

        return previous_config.model_dump(exclude={"config"}) != self.config.model_dump(exclude={"config"})

    async def apply_child(self, child: Service | StatefulSet) -> None:
        """Apply a child resource owned by this database.

        Args:
            child: The child resource to apply.
        """
        child.set_owner(self)
        await child.apply()

    async def wait_for_child(self, child: Service | StatefulSet, deadline: float) -> None:
        """Wait for an applied child resource to become ready by the operation deadline.

        Args:
            child: The applied child resource.
            deadline: Time on the ``time.monotonic`` clock by which the child must be ready.

        Raises:
            TimeoutError: If the child is not ready by the deadline; the message names the child.
        """
        try:
            await child.wait_ready(timeout=max(deadline - time.monotonic(), 0.0))
        except TimeoutError as error:
            message = f"{type(child).__name__} {child.name} did not become ready by the operation deadline"
            raise TimeoutError(message) from error

    async def on_create(self) -> DatabaseOutputs:
        """Deploy Qdrant using child Kubernetes resources.

        Creates headless service (for pod DNS), StatefulSet, and client service, waiting for each
        to be ready and for the LoadBalancer address, all within one ``OPERATION_BUDGET_SECONDS``
        budget.

        Returns:
            DatabaseOutputs with external LoadBalancer URLs.

        Raises:
            TimeoutError: If a child or the LoadBalancer address is not ready within the budget; the
                message names what was waited for.
        """
        deadline = time.monotonic() + OPERATION_BUDGET_SECONDS

        for child in (self._build_headless_service(), self._build_statefulset(), self._build_client_service()):
            await self.apply_child(child)
            await self.wait_for_child(child, deadline)

        return await self.fetch_outputs(deadline)

    async def on_update(self, previous_config: DatabaseConfig | None) -> DatabaseOutputs:
        """Reapply every child resource with the desired configuration.

        Without a previous configuration, behaves as ``on_create``. Otherwise waits only for the
        StatefulSet, and only when its settings changed, before reapplying the client Service. All
        waits share one ``OPERATION_BUDGET_SECONDS`` budget.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            DatabaseOutputs with external LoadBalancer URLs.

        Raises:
            TimeoutError: If a waited-for child or the LoadBalancer address is not ready within the
                budget; the message names what was waited for.
        """
        if previous_config is None:
            return await self.on_create()

        deadline = time.monotonic() + OPERATION_BUDGET_SECONDS
        statefulset = self._build_statefulset()

        await self.apply_child(self._build_headless_service())
        await self.apply_child(statefulset)

        if self.is_statefulset_changed(previous_config):
            await self.wait_for_child(statefulset, deadline)

        await self.apply_child(self._build_client_service())

        return await self.fetch_outputs(deadline)

    async def on_delete(self) -> None:
        """Delete the client Service, the StatefulSet and the headless Service, and wait until each is gone.

        Note:
            The stored vectors live on the StatefulSet's PersistentVolumeClaims; whether they are
            deleted with the StatefulSet follows the kubernetes provider's StatefulSet.
        """
        client_svc = self._build_client_service()
        await client_svc.on_delete()

        statefulset = self._build_statefulset()
        await statefulset.on_delete()

        headless_svc = self._build_headless_service()
        await headless_svc.on_delete()

    async def health(self) -> HealthStatus:
        """Check Qdrant database health via StatefulSet status.

        Returns:
            HealthStatus indicating healthy/degraded/unhealthy.
        """
        statefulset = self._build_statefulset()

        return await statefulset.health()

    async def logs(
        self,
        since: datetime | None = None,
        tail: int = 100,
    ) -> AsyncIterator[LogEntry]:
        """Fetch logs from Qdrant pods.

        Delegates to the underlying StatefulSet.

        Args:
            since: Only return logs after this timestamp.
            tail: Maximum number of log lines per pod.

        Yields:
            LogEntry for each log line from Qdrant pods.
        """
        statefulset = self._build_statefulset()

        async for entry in statefulset.logs(since=since, tail=tail):
            yield entry
