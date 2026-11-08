"""Qdrant Database resource - deploys Qdrant to Kubernetes using Kubernetes resources."""

from __future__ import annotations

import logging
import secrets
from collections.abc import AsyncIterator
from datetime import datetime
from typing import cast

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
from pragma_sdk import (
    ChildrenPending,
    Config,
    Field,
    HealthStatus,
    ImmutableDependency,
    LifecycleState,
    LogEntry,
    Outputs,
    Resource,
)
from pydantic import BaseModel, model_validator
from pydantic import Field as PydanticField


API_KEY_ENVIRONMENT_VARIABLE = "QDRANT__SERVICE__API_KEY"

logger = logging.getLogger(__name__)


def extract_api_key(statefulset: StatefulSet) -> str | None:
    """Read the API key a StatefulSet's containers pass to Qdrant.

    Args:
        statefulset: The StatefulSet child as the platform records it.

    Returns:
        The first non-empty ``QDRANT__SERVICE__API_KEY`` value of its containers, or ``None``
        when none passes one.
    """
    for container in cast(list[ContainerConfig], statefulset.config.containers):
        for variable in container.env or []:
            if variable.name == API_KEY_ENVIRONMENT_VARIABLE and variable.value:
                return variable.value

    return None


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
        generate_api_key: If True and api_key is None, generates a secure 32-char key once and
            keeps the key its StatefulSet child already carries.
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
    """Qdrant database deployed to Kubernetes through owned child resources.

    Applies three child resources through the platform, which records this database as their owner:
    - Headless Service for pod DNS
    - StatefulSet with persistent storage
    - LoadBalancer client Service for external access

    Lifecycle:
        - on_create, on_update: ``converge``: apply the three children, read them, and return
          ``ChildrenPending()`` until all three are READY, then the outputs
        - on_delete: Do nothing; the owned children are deleted with this database

    With ``generate_api_key``, the key is the one the StatefulSet child already carries; a key is
    generated only when there is no such child or it carries none, so a rerun keeps the key and
    does not roll the pods. The outputs report the key the StatefulSet child carries when all
    three children are READY.
    """

    computed = True

    async def resolve_api_key(self) -> str | None:
        """Resolve the API key the database runs with.

        Returns:
            ``api_key`` when set; with ``generate_api_key``, the key its StatefulSet child already
            carries, or a new 32-character hex key when there is no such child or it carries
            none; ``None`` otherwise.

        Raises:
            CallbackRefusedError: If the platform refuses the read of the StatefulSet child.
        """
        if self.config.api_key is not None:
            return cast(str, self.config.api_key)

        if not self.config.generate_api_key:
            return None

        recorded = await self._build_statefulset(None).read()
        recorded_key = extract_api_key(recorded) if recorded is not None else None

        if recorded_key is not None:
            return recorded_key

        logger.warning(
            "qdrant_api_key_generated resource=%s statefulset=%s reason=%s",
            self.id,
            self._statefulset_name(),
            "absent" if recorded is None else "no_key",
        )

        return secrets.token_hex(16)

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

    def _build_statefulset(self, api_key: str | None) -> StatefulSet:
        """Build StatefulSet for Qdrant pods.

        Args:
            api_key: Key Qdrant requires from clients, or ``None`` for none.

        Returns:
            Configured StatefulSet resource.
        """
        labels = self._labels()

        resources_config = None
        if self.config.resources:
            resources = cast(ResourceConfig, self.config.resources)
            resources_config = ResourcesConfig(
                requests={
                    "memory": resources.memory,
                    "cpu": resources.cpu,
                },
                limits={
                    "memory": resources.memory,
                    "cpu": resources.cpu,
                },
            )

        volume_mounts = [
            VolumeMountConfig(name="qdrant-storage", mount_path="/qdrant/storage"),
        ]

        env_vars = None
        if api_key:
            env_vars = [
                EnvVarConfig(name=API_KEY_ENVIRONMENT_VARIABLE, value=api_key),
            ]

        container = ContainerConfig(
            name="qdrant",
            image=cast(str, self.config.image),
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
            storage = cast(StorageConfig, self.config.storage)
            storage_class = storage.class_
            storage_size = storage.size

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

    def build_outputs(self, recorded_statefulset: StatefulSet, recorded_client_service: Service) -> DatabaseOutputs:
        """Build the outputs from the READY children as the platform records them.

        Args:
            recorded_statefulset: The StatefulSet child as read.
            recorded_client_service: The client Service child as read.

        Returns:
            DatabaseOutputs with the client Service's external URLs and the StatefulSet's key.

        Raises:
            RuntimeError: If the READY client Service reports no load-balancer address.
        """
        service_outputs = recorded_client_service.outputs
        address = service_outputs.load_balancer_address if service_outputs is not None else None

        if address is None:
            logger.error(
                "qdrant_load_balancer_address_missing resource=%s service=%s",
                self.id,
                recorded_client_service.name,
            )
            message = (
                f"Resource {self.id}: Service {recorded_client_service.name} is ready but reports no "
                "load-balancer address"
            )
            raise RuntimeError(message)

        return DatabaseOutputs(
            url=f"http://{address}:6333",
            grpc_url=f"http://{address}:6334",
            api_key=extract_api_key(recorded_statefulset),
        )

    def log_pending_children(self, recorded_children: dict[str, StatefulSet | Service | None]) -> None:
        """Log each child that is absent or not READY, the reason this pass returns ``ChildrenPending()``.

        A child absent right after its apply is logged at warning, one not READY at info.

        Args:
            recorded_children: Each child as read, keyed by its resource ID.
        """
        for child_id, recorded_child in recorded_children.items():
            if recorded_child is None:
                logger.warning("qdrant_children_pending resource=%s child=%s reason=absent", self.id, child_id)
            elif recorded_child.lifecycle_state != LifecycleState.READY:
                logger.info(
                    "qdrant_children_pending resource=%s child=%s reason=not_ready lifecycle_state=%s",
                    self.id,
                    child_id,
                    recorded_child.lifecycle_state,
                )

    async def converge(self) -> DatabaseOutputs | ChildrenPending:
        """Apply the three children, then report the outputs once all of them are READY.

        Returns:
            DatabaseOutputs when every child reads READY, else ``ChildrenPending()``.

        Raises:
            CallbackRefusedError: If the platform refuses an apply or a read of a child.
            RuntimeError: If the READY client Service reports no load-balancer address.
        """
        api_key = await self.resolve_api_key()

        headless_service = self._build_headless_service()
        statefulset = self._build_statefulset(api_key)
        client_service = self._build_client_service()

        for child in (headless_service, statefulset, client_service):
            await child.apply()

        recorded_statefulset = await statefulset.read()
        recorded_client_service = await client_service.read()
        recorded_children = {
            headless_service.id: await headless_service.read(),
            statefulset.id: recorded_statefulset,
            client_service.id: recorded_client_service,
        }

        if any(child is None or child.lifecycle_state != LifecycleState.READY for child in recorded_children.values()):
            self.log_pending_children(recorded_children)

            return ChildrenPending()

        return self.build_outputs(cast(StatefulSet, recorded_statefulset), cast(Service, recorded_client_service))

    async def on_create(self) -> DatabaseOutputs | ChildrenPending:
        """Deploy Qdrant through owned child resources.

        Returns:
            DatabaseOutputs once every child is READY, else ``ChildrenPending()``.

        Raises:
            CallbackRefusedError: If the platform refuses an apply or a read of a child.
            RuntimeError: If the READY client Service reports no load-balancer address.
        """
        return await self.converge()

    async def on_update(self, previous_config: DatabaseConfig | None) -> DatabaseOutputs | ChildrenPending:
        """Reapply every child resource with the desired configuration.

        Args:
            previous_config: The previous configuration, if any; every pass reapplies all
                children, so it is not read.

        Returns:
            DatabaseOutputs once every child is READY, else ``ChildrenPending()``.

        Raises:
            CallbackRefusedError: If the platform refuses an apply or a read of a child.
            RuntimeError: If the READY client Service reports no load-balancer address.
        """
        return await self.converge()

    async def on_delete(self) -> None:
        """Do nothing; the owned children are deleted with this database.

        Note:
            The stored vectors live on the StatefulSet's PersistentVolumeClaims; whether they are
            deleted with the StatefulSet follows the kubernetes provider's StatefulSet.
        """

    async def health(self) -> HealthStatus:
        """Check Qdrant database health via StatefulSet status.

        Returns:
            HealthStatus indicating healthy/degraded/unhealthy.
        """
        statefulset = self._build_statefulset(None)

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
        statefulset = self._build_statefulset(None)

        async for entry in statefulset.logs(since=since, tail=tail):
            yield entry
