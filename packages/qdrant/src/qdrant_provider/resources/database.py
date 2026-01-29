"""Qdrant Database resource - deploys Qdrant to GKE using Kubernetes resources."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator
from datetime import datetime
from typing import ClassVar

from lightkube.resources.core_v1 import Service as K8sService
from pydantic import BaseModel, Field as PydanticField
from pragma_sdk import Config, Dependency, HealthStatus, LogEntry, Outputs, Resource

from gcp_provider import GKE
from kubernetes_provider import (
    Service,
    ServiceConfig,
    StatefulSet,
    StatefulSetConfig,
)
from kubernetes_provider.client import create_client_from_gke
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


_LB_POLL_INTERVAL_SECONDS = 5
_LB_MAX_POLL_ATTEMPTS = 60


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
        cluster: GKE cluster dependency providing Kubernetes credentials.
        replicas: Number of Qdrant replicas (StatefulSet pods).
        storage: Persistent storage configuration.
        resources: CPU and memory limits.
        api_key: API key for Qdrant authentication. If provided, used directly.
        generate_api_key: If True and api_key is None, generates a secure 32-char key.
    """

    cluster: Dependency[GKE]
    replicas: int = 1
    storage: StorageConfig | None = None
    resources: ResourceConfig | None = None
    api_key: str | None = None
    generate_api_key: bool = False


class DatabaseOutputs(Outputs):
    """Outputs from Qdrant database deployment.

    Attributes:
        url: HTTP endpoint for Qdrant REST API (external LoadBalancer URL).
        grpc_url: gRPC endpoint for Qdrant (external LoadBalancer URL).
        api_key: The API key for authentication (if configured).
        ready: Whether the StatefulSet is ready.
    """

    url: str
    grpc_url: str
    api_key: str | None
    ready: bool


class Database(Resource[DatabaseConfig, DatabaseOutputs]):
    """Qdrant database deployed to GKE using Kubernetes resources.

    Creates child resources:
    - Headless Service for pod DNS
    - StatefulSet with persistent storage
    - Client Service for cluster access

    Lifecycle:
        - on_create: Create child resources, wait for ready
        - on_update: Update child resources
        - on_delete: Child resources cascade deleted via owner_references
    """

    provider: ClassVar[str] = "qdrant"
    resource: ClassVar[str] = "database"

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
        """Get headless service name for pod DNS."""
        return f"qdrant-{self.name}-headless"

    def _client_service_name(self) -> str:
        """Get client service name for cluster access."""
        return f"qdrant-{self.name}"

    def _statefulset_name(self) -> str:
        """Get StatefulSet name."""
        return f"qdrant-{self.name}"

    def _namespace(self) -> str:
        """Get Kubernetes namespace."""
        return "default"

    def _labels(self) -> dict[str, str]:
        """Get labels for Kubernetes resources."""
        return {
            "app": "qdrant",
            "app.kubernetes.io/name": "qdrant",
            "app.kubernetes.io/instance": self.name,
        }

    async def _get_client(self):
        """Get lightkube client from GKE cluster credentials."""
        cluster = await self.config.cluster.resolve()
        outputs = cluster.outputs

        if outputs is None:
            msg = "GKE cluster outputs not available"
            raise RuntimeError(msg)

        creds = cluster.config.credentials

        return create_client_from_gke(outputs, creds)

    async def _wait_for_load_balancer_ip(self, timeout: float = 300.0) -> str:
        """Wait for LoadBalancer external IP to be assigned.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            External IP address.

        Raises:
            TimeoutError: If IP not assigned within timeout.
        """
        client = await self._get_client()
        namespace = self._namespace()
        service_name = self._client_service_name()
        max_attempts = int(timeout / _LB_POLL_INTERVAL_SECONDS)

        for _ in range(max_attempts):
            svc = await client.get(K8sService, name=service_name, namespace=namespace)

            if svc.status and svc.status.loadBalancer and svc.status.loadBalancer.ingress:
                ingress = svc.status.loadBalancer.ingress[0]

                if ingress.ip:
                    return ingress.ip

                if ingress.hostname:
                    return ingress.hostname

            await asyncio.sleep(_LB_POLL_INTERVAL_SECONDS)

        msg = f"LoadBalancer {service_name} did not receive external IP within {timeout}s"
        raise TimeoutError(msg)

    def _build_headless_service(self) -> Service:
        """Build headless service for pod DNS."""
        config = ServiceConfig(
            cluster=self.config.cluster,
            namespace=self._namespace(),
            type="Headless",
            selector=self._labels(),
            ports=[
                PortConfig(name="rest", port=6333, target_port=6333),
                PortConfig(name="grpc", port=6334, target_port=6334),
            ],
        )

        return Service(
            name=self._headless_service_name(),
            config=config,
        )

    def _build_client_service(self) -> Service:
        """Build client service with LoadBalancer for external access."""
        config = ServiceConfig(
            cluster=self.config.cluster,
            namespace=self._namespace(),
            type="LoadBalancer",
            selector=self._labels(),
            ports=[
                PortConfig(name="rest", port=6333, target_port=6333),
                PortConfig(name="grpc", port=6334, target_port=6334),
            ],
        )

        return Service(
            name=self._client_service_name(),
            config=config,
        )

    def _build_statefulset(self) -> StatefulSet:
        """Build StatefulSet for Qdrant pods."""
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
            image="qdrant/qdrant:latest",
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
            cluster=self.config.cluster,
            namespace=self._namespace(),
            replicas=self.config.replicas,
            service_name=self._headless_service_name(),
            selector=labels,
            containers=[container],
            volume_claim_templates=[pvc_template],
        )

        return StatefulSet(
            name=self._statefulset_name(),
            config=config,
        )

    async def _build_outputs(self) -> DatabaseOutputs:
        """Build outputs with external LoadBalancer URLs."""
        external_ip = await self._wait_for_load_balancer_ip()

        url = f"http://{external_ip}:6333"
        grpc_url = f"http://{external_ip}:6334"

        return DatabaseOutputs(
            url=url,
            grpc_url=grpc_url,
            api_key=self._resolve_api_key(),
            ready=True,
        )

    async def on_create(self) -> DatabaseOutputs:
        """Deploy Qdrant to GKE using child Kubernetes resources.

        Creates headless service (for pod DNS), StatefulSet, and client service.
        Waits for all resources to be ready and LoadBalancer IP to be assigned.

        Returns:
            DatabaseOutputs with external LoadBalancer URLs.
        """
        headless_svc = self._build_headless_service()
        headless_svc.set_owner(self)
        await headless_svc.apply()
        await headless_svc.wait_ready(timeout=60.0)

        statefulset = self._build_statefulset()
        statefulset.set_owner(self)
        await statefulset.apply()
        await statefulset.wait_ready(timeout=300.0)

        client_svc = self._build_client_service()
        client_svc.set_owner(self)
        await client_svc.apply()
        await client_svc.wait_ready(timeout=60.0)

        return await self._build_outputs()

    async def on_update(self, previous_config: DatabaseConfig) -> DatabaseOutputs:
        """Update Qdrant deployment.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            DatabaseOutputs with external LoadBalancer URLs.

        Raises:
            ValueError: If cluster changed (requires delete + create).
        """
        if previous_config.cluster.id != self.config.cluster.id:
            msg = "Cannot change cluster; delete and recreate resource"
            raise ValueError(msg)

        if self.outputs is not None:
            previous_dict = previous_config.model_dump(exclude={"cluster"})
            current_dict = self.config.model_dump(exclude={"cluster"})

            if previous_dict == current_dict:
                return self.outputs

        headless_svc = self._build_headless_service()
        headless_svc.set_owner(self)
        await headless_svc.apply()
        await headless_svc.wait_ready(timeout=60.0)

        statefulset = self._build_statefulset()
        statefulset.set_owner(self)
        await statefulset.apply()
        await statefulset.wait_ready(timeout=300.0)

        client_svc = self._build_client_service()
        client_svc.set_owner(self)
        await client_svc.apply()
        await client_svc.wait_ready(timeout=60.0)

        return await self._build_outputs()

    async def on_delete(self) -> None:
        """Delete Qdrant deployment.

        Explicitly deletes child Kubernetes resources. Once PRA-137 is implemented,
        this will be handled automatically via owner_references cascading deletes.
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
