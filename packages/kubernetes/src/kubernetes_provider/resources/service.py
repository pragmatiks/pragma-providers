"""Kubernetes Service resource."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal, cast

from lightkube import AsyncClient
from lightkube.models.core_v1 import ServicePort, ServiceSpec
from lightkube.resources.core_v1 import Endpoints
from lightkube.resources.core_v1 import Service as K8sService
from pragma_sdk import (
    AlreadyExistsError,
    Config,
    Field,
    HealthStatus,
    ImmutableDependency,
    ImmutableField,
    LogEntry,
    Outputs,
    Resource,
)
from pydantic import BaseModel

from kubernetes_provider.marker import MarkerRefusal
from kubernetes_provider.objects import (
    FIELD_MANAGER,
    build_object_metadata,
    check_create_conflict,
    create_object,
    delete_object,
    fetch_object,
    fetch_own_object,
    wait_for_absence,
)
from kubernetes_provider.resources.config import KubernetesConfig


LOAD_BALANCER_POLL_INTERVAL_SECONDS = 5

logger = logging.getLogger(__name__)


def extract_load_balancer_address(service: K8sService) -> str | None:
    """Read the external address the cluster assigned to a LoadBalancer Service.

    Args:
        service: Service as read from the cluster.

    Returns:
        The IP of the first load-balancer ingress point, else its hostname; ``None`` when the
        Service is not of type LoadBalancer or its status lists no ingress point with either.
    """
    if service.spec is None or service.spec.type != "LoadBalancer":
        return None

    load_balancer = service.status.loadBalancer if service.status else None
    ingress = load_balancer.ingress if load_balancer else None

    if not ingress:
        return None

    return ingress[0].ip or ingress[0].hostname or None


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
        load_balancer_address: External IP of a LoadBalancer Service's first ingress point, else
            its hostname; ``None`` for every other type.
    """

    name: str
    namespace: str
    cluster_ip: str
    type: str
    load_balancer_address: str | None = None


class Service(Resource[ServiceConfig, ServiceOutputs]):
    """Kubernetes Service resource.

    Exposes workloads via ClusterIP, NodePort, LoadBalancer, or Headless
    service types. Headless services (``type: "Headless"``) automatically
    set ``clusterIP: None`` for DNS-based pod discovery.

    A LoadBalancer Service is ready only once the cluster assigns it an
    external address: ``on_create`` and ``on_update`` read the Service until
    its status carries one and report it as ``load_balancer_address``. On a
    cluster that assigns none (kind, bare metal without a load-balancer
    controller) the operation fails at its operation deadline. An address the
    cloud re-assigns to a live Service no longer matches the recorded
    outputs. Every other type is ready once applied. Health checks verify
    both the Service existence and the presence of backend endpoints.

    Creates the service, marked with this resource's ``pragma.io/resource``
    annotation; an existing service carrying it is this resource's earlier
    create and is applied over, any other is refused. Updates use server-side
    apply under ``field_manager="pragma-kubernetes"``, forcing conflicts.

    Lifecycle:
        - on_create: Create the service; apply over this resource's earlier
          create, refuse any other existing service; wait for a LoadBalancer's address
        - on_observe: Look up the service by name and namespace, reporting its address; raise
          AlreadyExistsError for one without this resource's annotation
        - on_update: Apply updated service configuration; wait for a LoadBalancer's address
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
            for p in cast(list[PortConfig], self.config.ports)
        ]

        service_type = cast(str, self.config.type)
        cluster_ip = cast(str | None, self.config.cluster_ip)

        if service_type == "Headless":
            service_type = "ClusterIP"
            cluster_ip = "None"

        spec = ServiceSpec(
            type=service_type,
            selector=cast(dict[str, str], self.config.selector),
            ports=ports,
        )

        if cluster_ip:
            spec.clusterIP = cluster_ip

        return K8sService(
            metadata=build_object_metadata(self.id, self.name, cast(str, self.config.namespace)),
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
            load_balancer_address=extract_load_balancer_address(service),
        )

    async def wait_until_ready(self, client: AsyncClient, service: K8sService) -> K8sService:
        """Read a LoadBalancer Service until the cluster assigns it an address; return any other type as given.

        The reads repeat every ``LOAD_BALANCER_POLL_INTERVAL_SECONDS`` with no bound of their own:
        the operation deadline ends them, by cancelling the handler.

        Args:
            client: Lightkube client for the target cluster.
            service: Service as the create or apply returned it.

        Returns:
            The Service as last read: for a LoadBalancer, once its status carries an ingress IP or
            hostname; for every other type, ``service`` itself.

        Raises:
            ApiError: If reading a LoadBalancer Service fails, including when it no longer exists.
        """
        if self.config.type != "LoadBalancer":
            return service

        namespace = cast(str, self.config.namespace)
        address = extract_load_balancer_address(service)

        if address is None:
            logger.info(
                "kubernetes_service_load_balancer_waiting resource=%s namespace=%s service=%s interval_seconds=%s",
                self.id,
                namespace,
                self.name,
                LOAD_BALANCER_POLL_INTERVAL_SECONDS,
            )

        while address is None:
            await asyncio.sleep(LOAD_BALANCER_POLL_INTERVAL_SECONDS)
            service = await client.get(K8sService, self.name, namespace=namespace)
            address = extract_load_balancer_address(service)

        logger.info(
            "kubernetes_service_load_balancer_assigned resource=%s namespace=%s service=%s address=%s",
            self.id,
            namespace,
            self.name,
            address,
        )

        return service

    async def on_create(self) -> ServiceOutputs:
        """Create the Kubernetes Service.

        Returns:
            ServiceOutputs with service details.

        Raises:
            AlreadyExistsError: If the namespace holds a service of this name that this resource
                did not create.
            RuntimeError: If this resource's earlier service is being deleted.
            ApiError: If reading a LoadBalancer Service for its address fails, including when it
                no longer exists.
        """
        desired = self._build_service()

        async with self._get_client() as client:
            try:
                service = await create_object(client, desired)
            except AlreadyExistsError as error:
                await check_create_conflict(client, desired, self.id, error)
            else:
                ready_service = await self.wait_until_ready(client, service)

                return self.build_outputs(ready_service)

        return await self.on_update(None)

    async def on_observe(self) -> ServiceOutputs | None:
        """Look up the Kubernetes Service named after this resource.

        Returns:
            ServiceOutputs carrying the load-balancer address its status reports, ``None`` in it
            while the cluster has assigned a LoadBalancer Service none yet; ``None`` if the service
            does not exist.

        Raises:
            AlreadyExistsError: If the service carries no ``pragma.io/resource`` annotation naming this
                resource.
        """
        async with self._get_client() as client:
            service = await fetch_own_object(
                client, K8sService, self.id, self.name, cast(str, self.config.namespace), MarkerRefusal.OBSERVE
            )

        if service is None:
            return None

        return self.build_outputs(service)

    async def on_update(self, previous_config: ServiceConfig | None) -> ServiceOutputs:
        """Apply the full desired service configuration, refusing a headless switch against the live service.

        A LoadBalancer Service is returned once its address is assigned.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            ServiceOutputs with updated service details.

        Raises:
            ValueError: If the service in the cluster is headless and the desired one is not, or
                the reverse; the cluster IP of an existing service cannot change.
            ApiError: If reading a LoadBalancer Service for its address fails, including when it
                no longer exists.
        """
        desired_headless = self.config.type == "Headless" or self.config.cluster_ip == "None"

        async with self._get_client() as client:
            live_service = await fetch_object(client, K8sService, self.name, cast(str, self.config.namespace))
            live_headless = live_service is not None and self.build_outputs(live_service).cluster_ip == "None"

            if live_service is not None and live_headless != desired_headless:
                requested_change = "become headless" if desired_headless else "stop being headless"
                message = (
                    f"Service {self.name} cannot {requested_change}: the cluster IP of the service in the "
                    "cluster cannot change"
                )
                raise ValueError(message)

            service = await client.apply(self._build_service(), field_manager=FIELD_MANAGER, force=True)
            ready_service = await self.wait_until_ready(client, service)

        return self.build_outputs(ready_service)

    async def on_delete(self) -> None:
        """Delete Kubernetes Service and wait until it is gone.

        Idempotent: Succeeds if service doesn't exist.

        Raises:
            TimeoutError: If the service is still present after the deletion timeout.
        """
        async with self._get_client() as client:
            await delete_object(client, K8sService, self.name, cast(str, self.config.namespace))
            await wait_for_absence(client, K8sService, self.name, cast(str, self.config.namespace))

    async def health(self) -> HealthStatus:
        """Check Service health by verifying existence and endpoints.

        Returns:
            HealthStatus indicating healthy/degraded/unhealthy.
        """
        async with self._get_client() as client:
            service = await fetch_object(client, K8sService, self.name, cast(str, self.config.namespace))
            endpoints = await fetch_object(client, Endpoints, self.name, cast(str, self.config.namespace))

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
