"""Kubernetes Service resource."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import ClassVar, Literal

from gcp_provider import GKE
from lightkube import ApiError
from lightkube.models.core_v1 import ServicePort, ServiceSpec
from lightkube.models.meta_v1 import ObjectMeta
from lightkube.resources.core_v1 import Endpoints
from lightkube.resources.core_v1 import Service as K8sService
from pragma_sdk import Config, Field, HealthStatus, ImmutableDependency, ImmutableField, LogEntry, Outputs, Resource
from pydantic import BaseModel
from pydantic import Field as PydanticField

from kubernetes_provider.client import create_client_from_gke


class PortConfig(BaseModel):
    """Service port mapping between the service and target pods.

    Attributes:
        name: Optional port name (required when exposing multiple ports).
        port: Port number exposed by the service.
        target_port: Port on the target pods; defaults to ``port`` if not set.
        protocol: Network protocol (TCP or UDP).
    """

    model_config = {"extra": "forbid"}

    name: str | None = PydanticField(
        default=None, description="Optional port name (required when exposing multiple ports)."
    )
    port: int = PydanticField(description="Port number exposed by the service.")
    target_port: int | None = PydanticField(
        default=None, description="Port on the target pods; defaults to port if not set."
    )
    protocol: Literal["TCP", "UDP"] = PydanticField(default="TCP", description="Network protocol (TCP or UDP).")


class ServiceConfig(Config):
    """Configuration for a Kubernetes Service.

    Attributes:
        cluster: GKE cluster dependency providing Kubernetes credentials.
        namespace: Kubernetes namespace for the service (immutable after creation).
        type: Service type (ClusterIP, NodePort, LoadBalancer, or Headless).
        selector: Label selector matching the target pods.
        ports: List of port mappings between the service and target pods.
        cluster_ip: Explicit cluster IP; use ``"None"`` for headless services.
    """

    cluster: ImmutableDependency[GKE]
    namespace: ImmutableField[str] = "default"
    type: Field[Literal["ClusterIP", "NodePort", "LoadBalancer", "Headless"]] = "ClusterIP"
    selector: Field[dict[str, str]]
    ports: Field[list[PortConfig]]
    cluster_ip: Field[str] | None = None


class ServiceOutputs(Outputs):
    """Outputs from Kubernetes Service creation.

    Attributes:
        name: Service name as created in the cluster.
        namespace: Kubernetes namespace containing the service.
        cluster_ip: Assigned cluster IP (empty string for headless services).
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
        - on_update: Apply updated service configuration
        - on_delete: Delete the service (idempotent)
    """

    provider: ClassVar[str] = "kubernetes"
    resource: ClassVar[str] = "service"
    description = "Manages Kubernetes services for network access to pods."

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

    def _build_outputs(self, service: K8sService) -> ServiceOutputs:
        """Build outputs from Kubernetes Service object.

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
            service = self._build_service()

            await client.apply(service, field_manager="pragma-kubernetes")

            result = await client.get(
                K8sService,
                name=self.name,
                namespace=self.config.namespace,
            )

            return self._build_outputs(result)

    async def on_update(self, previous_config: ServiceConfig) -> ServiceOutputs:
        """Update Kubernetes Service.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            ServiceOutputs with updated service details.
        """
        async with self._get_client() as client:
            service = self._build_service()

            await client.apply(service, field_manager="pragma-kubernetes")

            result = await client.get(
                K8sService,
                name=self.name,
                namespace=self.config.namespace,
            )

            return self._build_outputs(result)

    async def on_delete(self) -> None:
        """Delete Kubernetes Service.

        Idempotent: Succeeds if service doesn't exist.

        Raises:
            ApiError: If deletion fails for reasons other than not found.
        """
        async with self._get_client() as client:
            try:
                await client.delete(
                    K8sService,
                    name=self.name,
                    namespace=self.config.namespace,
                )
            except ApiError as e:
                if e.status.code != 404:
                    raise

    async def health(self) -> HealthStatus:
        """Check Service health by verifying existence and endpoints.

        Returns:
            HealthStatus indicating healthy/degraded/unhealthy.

        Raises:
            ApiError: If health check fails for reasons other than not found.
        """
        async with self._get_client() as client:
            try:
                await client.get(
                    K8sService,
                    name=self.name,
                    namespace=self.config.namespace,
                )
            except ApiError as e:
                if e.status.code == 404:
                    return HealthStatus(
                        status="unhealthy",
                        message="Service not found",
                    )
                raise

            try:
                endpoints = await client.get(
                    Endpoints,
                    name=self.name,
                    namespace=self.config.namespace,
                )

                has_endpoints = False
                endpoint_count = 0

                if endpoints.subsets:
                    for subset in endpoints.subsets:
                        if subset.addresses:
                            has_endpoints = True
                            endpoint_count += len(subset.addresses)

                if has_endpoints:
                    return HealthStatus(
                        status="healthy",
                        message=f"Service has {endpoint_count} endpoint(s)",
                        details={"endpoint_count": endpoint_count},
                    )

                return HealthStatus(
                    status="degraded",
                    message="Service exists but has no endpoints",
                )

            except ApiError as e:
                if e.status.code == 404:
                    return HealthStatus(
                        status="degraded",
                        message="Service exists but endpoints not found",
                    )
                raise

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
