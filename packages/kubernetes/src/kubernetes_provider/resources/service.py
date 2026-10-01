"""Kubernetes Service resource."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

from lightkube.models.core_v1 import ServicePort, ServiceSpec
from lightkube.models.meta_v1 import ObjectMeta
from lightkube.resources.core_v1 import Endpoints
from lightkube.resources.core_v1 import Service as K8sService
from pragma_sdk import Config, Field, HealthStatus, ImmutableDependency, ImmutableField, LogEntry, Outputs, Resource
from pydantic import BaseModel

from kubernetes_provider.objects import delete_object, fetch_object, wait_for_absence
from kubernetes_provider.resources.config import KubernetesConfig


class PortConfig(BaseModel):
    """Service port mapping between the service and target pods.

    Attributes:
        name: Optional port name (required when exposing multiple ports).
        port: Port number exposed by the service.
        target_port: Port on the target pods; defaults to ``port`` if not set.
        protocol: Network protocol (TCP or UDP).
    """

    model_config = {"extra": "forbid"}

    name: str | None = None
    port: int
    target_port: int | None = None
    protocol: Literal["TCP", "UDP"] = "TCP"


class ServiceConfig(Config):
    """Configuration for a Kubernetes Service.

    Attributes:
        config: Kubernetes config dependency providing cluster access.
        namespace: Kubernetes namespace for the service (immutable after creation).
        type: Service type (ClusterIP, NodePort, LoadBalancer, or Headless).
        selector: Label selector matching the target pods.
        ports: List of port mappings between the service and target pods.
        cluster_ip: Explicit cluster IP; use ``"None"`` for headless services (immutable after creation).
    """

    config: ImmutableDependency[KubernetesConfig]
    namespace: ImmutableField[str] = "default"
    type: Field[Literal["ClusterIP", "NodePort", "LoadBalancer", "Headless"]] = "ClusterIP"
    selector: Field[dict[str, str]]
    ports: Field[list[PortConfig]]
    cluster_ip: ImmutableField[str] | None = None


class ServiceOutputs(Outputs):
    """Outputs from Kubernetes Service creation.

    Attributes:
        name: Service name as created in the cluster.
        namespace: Kubernetes namespace containing the service.
        cluster_ip: Assigned cluster IP; ``"None"`` for headless services, empty string when the
            cluster reports none.
        type: Kubernetes service type after apply.
    """

    name: str
    namespace: str
    cluster_ip: str
    type: str


class Service(Resource[ServiceConfig, ServiceOutputs]):
    """Kubernetes Service resource.

    Exposes workloads via ClusterIP, NodePort, LoadBalancer, or Headless
    service types. Headless services (``type: "Headless"``) automatically
    set ``clusterIP: None`` for DNS-based pod discovery.

    Services are immediately ready after apply (no polling needed). Health
    checks verify both the Service existence and the presence of backend
    endpoints.

    Uses server-side apply with ``field_manager="pragma-kubernetes"`` for
    idempotent operations.

    Lifecycle:
        - on_create: Apply service configuration
        - on_observe: Look up the service by name and namespace
        - on_update: Apply updated service configuration
        - on_delete: Delete the service and wait until it is gone (idempotent)
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

    def _build_service(self) -> K8sService:
        """Build Kubernetes Service object from config.

        Returns:
            Kubernetes Service object ready to apply.
        """
        ports = [
            ServicePort(
                name=p.name,
                port=p.port,
                targetPort=p.target_port or p.port,
                protocol=p.protocol,
            )
            for p in self.config.ports
        ]

        service_type = self.config.type
        cluster_ip = self.config.cluster_ip

        if service_type == "Headless":
            service_type = "ClusterIP"
            cluster_ip = "None"

        spec = ServiceSpec(
            type=service_type,
            selector=self.config.selector,
            ports=ports,
        )

        if cluster_ip:
            spec.clusterIP = cluster_ip

        return K8sService(
            metadata=ObjectMeta(
                name=self.name,
                namespace=self.config.namespace,
            ),
            spec=spec,
        )

    def build_outputs(self, service: K8sService) -> ServiceOutputs:
        """Build outputs from the Service as read from the cluster.

        Args:
            service: Service returned by the cluster.

        Returns:
            ServiceOutputs with service details.
        """
        assert service.metadata is not None
        assert service.metadata.name is not None
        assert service.metadata.namespace is not None
        assert service.spec is not None
        assert service.spec.type is not None

        return ServiceOutputs(
            name=service.metadata.name,
            namespace=service.metadata.namespace,
            cluster_ip=service.spec.clusterIP or "",
            type=service.spec.type,
        )

    async def on_create(self) -> ServiceOutputs:
        """Create or update Kubernetes Service.

        Idempotent: Uses apply() which handles both create and update.

        Returns:
            ServiceOutputs with service details.
        """
        async with self._get_client() as client:
            service = await client.apply(self._build_service(), field_manager="pragma-kubernetes")

        return self.build_outputs(service)

    async def on_observe(self) -> ServiceOutputs | None:
        """Look up the Kubernetes Service named after this resource.

        Returns:
            ServiceOutputs, or ``None`` if the service does not exist.
        """
        async with self._get_client() as client:
            service = await fetch_object(client, K8sService, self.name, self.config.namespace)

        if service is None:
            return None

        return self.build_outputs(service)

    async def on_update(self, previous_config: ServiceConfig | None) -> ServiceOutputs:
        """Apply the full desired service configuration, refusing a headless switch against the live service.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            ServiceOutputs with updated service details.

        Raises:
            ValueError: If the service in the cluster is headless and the desired one is not, or
                the reverse; the cluster IP of an existing service cannot change.
        """
        live_service = await self.on_observe()
        live_headless = live_service is not None and live_service.cluster_ip == "None"
        desired_headless = self.config.type == "Headless" or self.config.cluster_ip == "None"

        if live_service is not None and live_headless != desired_headless:
            requested_change = "become headless" if desired_headless else "stop being headless"
            message = (
                f"Service {self.name} cannot {requested_change}: the cluster IP of the service in the "
                "cluster cannot change"
            )
            raise ValueError(message)

        return await self.on_create()

    async def on_delete(self) -> None:
        """Delete Kubernetes Service and wait until it is gone.

        Idempotent: Succeeds if service doesn't exist.

        Raises:
            TimeoutError: If the service is still present after the deletion timeout.
        """
        async with self._get_client() as client:
            await delete_object(client, K8sService, self.name, self.config.namespace)
            await wait_for_absence(client, K8sService, self.name, self.config.namespace)

    async def health(self) -> HealthStatus:
        """Check Service health by verifying existence and endpoints.

        Returns:
            HealthStatus indicating healthy/degraded/unhealthy.
        """
        async with self._get_client() as client:
            service = await fetch_object(client, K8sService, self.name, self.config.namespace)
            endpoints = await fetch_object(client, Endpoints, self.name, self.config.namespace)

        if service is None:
            return HealthStatus(
                status="unhealthy",
                message=f"Service {self.name} not found",
            )

        if endpoints is None:
            return HealthStatus(
                status="degraded",
                message="Service exists but endpoints not found",
            )

        endpoint_count = sum(len(subset.addresses or []) for subset in endpoints.subsets or [])

        if endpoint_count:
            return HealthStatus(
                status="healthy",
                message=f"Service has {endpoint_count} endpoint(s)",
                details={"endpoint_count": endpoint_count},
            )

        return HealthStatus(
            status="degraded",
            message="Service exists but has no endpoints",
        )

    async def logs(
        self,
        since: datetime | None = None,
        tail: int = 100,
    ) -> AsyncIterator[LogEntry]:
        """Services do not produce logs.

        This method exists for interface compatibility and yields a single
        informational entry.

        Args:
            since: Ignored for services.
            tail: Ignored for services.

        Yields:
            A single informational LogEntry.
        """
        yield LogEntry(
            timestamp=datetime.now(UTC),
            level="info",
            message="Services do not produce logs",
        )
        return
