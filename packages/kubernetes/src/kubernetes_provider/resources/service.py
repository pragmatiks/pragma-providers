"""Kubernetes Service resource."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import ClassVar, Literal

from gcp_provider import GKE
from lightkube import ApiError
from lightkube.models.core_v1 import ServicePort, ServiceSpec
from lightkube.models.meta_v1 import ObjectMeta
from lightkube.resources.core_v1 import Endpoints, Service as K8sService
from pydantic import BaseModel
from pragma_sdk import Config, Dependency, HealthStatus, LogEntry, Outputs, Resource

from kubernetes_provider.client import create_client_from_gke


class PortConfig(BaseModel):
    """Service port configuration.

    Attributes:
        name: Optional name for the port.
        port: Port exposed by the service.
        target_port: Port on the target pods.
        protocol: Protocol (TCP or UDP).
    """

    model_config = {"extra": "forbid"}

    name: str | None = None
    port: int
    target_port: int | None = None
    protocol: Literal["TCP", "UDP"] = "TCP"


class ServiceConfig(Config):
    """Configuration for a Kubernetes Service.

    Attributes:
        cluster: GKE cluster dependency providing Kubernetes credentials.
        namespace: Kubernetes namespace for the service.
        type: Service type (ClusterIP, NodePort, LoadBalancer, Headless).
        selector: Label selector for target pods.
        ports: List of port configurations.
        cluster_ip: Explicit cluster IP (use "None" for headless services).
    """

    cluster: Dependency[GKE]
    namespace: str = "default"
    type: Literal["ClusterIP", "NodePort", "LoadBalancer", "Headless"] = "ClusterIP"
    selector: dict[str, str]
    ports: list[PortConfig]
    cluster_ip: str | None = None


class ServiceOutputs(Outputs):
    """Outputs from Kubernetes Service creation.

    Attributes:
        name: Service name.
        namespace: Kubernetes namespace.
        cluster_ip: Assigned cluster IP (empty for headless).
        type: Service type.
    """

    name: str
    namespace: str
    cluster_ip: str
    type: str


class Service(Resource[ServiceConfig, ServiceOutputs]):
    """Kubernetes Service resource.

    Creates and manages Kubernetes Services using lightkube.
    Services are immediately ready after apply (no polling needed).

    Lifecycle:
        - on_create: Apply service configuration
        - on_update: Apply updated service configuration
        - on_delete: Delete the service
    """

    provider: ClassVar[str] = "kubernetes"
    resource: ClassVar[str] = "service"

    async def _get_client(self):
        """Get lightkube client from GKE cluster credentials."""
        cluster = await self.config.cluster.resolve()
        outputs = cluster.outputs

        if outputs is None:
            msg = "GKE cluster outputs not available"
            raise RuntimeError(msg)

        creds = cluster.config.credentials

        return create_client_from_gke(outputs, creds)

    def _build_service(self) -> K8sService:
        """Build Kubernetes Service object from config."""
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
        """Build outputs from Kubernetes Service object."""
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
        client = await self._get_client()
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

        Raises:
            ValueError: If immutable fields changed.
        """
        if previous_config.cluster.id != self.config.cluster.id:
            msg = "Cannot change cluster; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.namespace != self.config.namespace:
            msg = "Cannot change namespace; delete and recreate resource"
            raise ValueError(msg)

        client = await self._get_client()
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
        """
        client = await self._get_client()

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
        """
        client = await self._get_client()

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

        This method exists for interface compatibility but yields nothing.

        Args:
            since: Ignored for services.
            tail: Ignored for services.

        Yields:
            Nothing - services don't have logs.
        """
        yield LogEntry(
            timestamp=datetime.now(timezone.utc),
            level="info",
            message="Services do not produce logs",
        )
        return
