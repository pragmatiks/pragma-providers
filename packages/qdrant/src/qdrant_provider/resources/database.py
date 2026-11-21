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
    ProbeConfig,
    ResourcesConfig,
    SecretEnvVarConfig,
    VolumeClaimTemplateConfig,
    VolumeMountConfig,
)
from pragma_provider import Secret, SecretConfig
from pragma_sdk import (
    ChildrenPending,
    Config,
    Field,
    FieldReference,
    ImmutableDependency,
    LifecycleState,
    LogEntry,
    Outputs,
    Resource,
    Vitals,
)
from pydantic import BaseModel, SecretStr, model_validator
from pydantic import Field as PydanticField


API_KEY_ENVIRONMENT_VARIABLE = "QDRANT__SERVICE__API_KEY"
API_KEY_SECRET_ENTRY = "api_key"

logger = logging.getLogger(__name__)


def extract_api_key(secret: Secret) -> SecretStr | None:
    """Read the API key a database's secret child holds.

    Args:
        secret: The secret child as the platform records it.

    Returns:
        The ``api_key`` entry of the secret's data, a ``SecretStr`` the caller reveals with
        ``get_secret_value()``, or ``None`` when it holds none.
    """
    data = cast(dict[str, SecretStr], secret.config.data)

    return data.get(API_KEY_SECRET_ENTRY)


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
        api_key: API key for Qdrant authentication, a reference to a secret output. If
            provided, used directly.
        generate_api_key: If True and api_key is None, generates a secure 32-char key once and
            keeps the key its secret child already holds.
    """

    config: ImmutableDependency[KubernetesConfig]
    replicas: Field[int] = 1
    image: Field[str] = "qdrant/qdrant:latest"
    storage: Field[StorageConfig] | None = None
    resources: Field[ResourceConfig] | None = None
    api_key: Field[SecretStr] | None = None
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
        api_key: The API key for authentication (if configured), copied from the secret child.
    """

    url: str
    grpc_url: str
    api_key: SecretStr | None


class Database(Resource[DatabaseConfig, DatabaseOutputs]):
    """Qdrant database deployed to Kubernetes through owned child resources.

    Applies four child resources through the platform, which records this database as their owner:
    - ``pragma/secret`` holding the API key
    - Headless Service for pod DNS
    - StatefulSet with persistent storage
    - LoadBalancer client Service for external access

    Lifecycle:
        - on_create, on_update: ``converge``: apply the secret child first, then the other three,
          read them, and return ``ChildrenPending()`` until all four are READY, then the outputs
        - on_delete: Do nothing; the owned children are deleted with this database

    The API key is ``api_key`` when set and not empty; with ``generate_api_key``, an existing key
    is kept and a new one is generated only when none exists.
    """

    computed = True

    async def resolve_api_key(self) -> SecretStr | None:
        """Resolve the API key the database runs with.

        Returns:
            ``api_key`` when set and not empty; with ``generate_api_key``, the key its secret child already
            holds, or a new 32-character hex key when there is no such child or it holds
            none; ``None`` otherwise.

        Raises:
            CallbackRefusedError: If the platform refuses the read of the secret child.
        """
        if self.config.api_key is not None:
            api_key = self.resolved(self.config.api_key)

            return api_key if api_key.get_secret_value() else None

        if not self.config.generate_api_key:
            return None

        recorded = await self.build_api_key_secret(None).read()
        recorded_key = extract_api_key(recorded) if recorded is not None else None

        if recorded_key is not None:
            return recorded_key

        logger.warning(
            "qdrant_api_key_generated resource=%s secret=%s reason=%s",
            self.id,
            self.api_key_secret_name(),
            "absent" if recorded is None else "no_key",
        )

        return SecretStr(secrets.token_hex(16))

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

    def api_key_secret_name(self) -> str:
        """Get the name of the secret child holding the API key.

        Returns:
            The secret name.
        """
        return f"qdrant-{self.name}-api-key"

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

    def build_api_key_secret(self, api_key: SecretStr | None) -> Secret:
        """Build the secret child holding the API key.

        Args:
            api_key: Key Qdrant requires from clients, or ``None`` for none.

        Returns:
            Configured Secret resource, with an ``api_key`` entry only when a key is given.
        """
        data = {API_KEY_SECRET_ENTRY: api_key} if api_key is not None else {}

        return Secret(
            project_id=self.project_id,
            name=self.api_key_secret_name(),
            config=SecretConfig(data=data),
        )

    @staticmethod
    def build_api_key_reference(secret: Secret) -> FieldReference:
        """Build the reference the StatefulSet's API key environment variable resolves.

        Args:
            secret: The secret child holding the API key.

        Returns:
            Reference to the secret child's ``outputs.api_key``.
        """
        identity = secret.identity

        return FieldReference(
            project_id=identity.project_id,
            provider=identity.provider,
            resource=identity.resource,
            name=identity.name,
            field=f"outputs.{API_KEY_SECRET_ENTRY}",
        )

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

    def _build_statefulset(self, api_key_reference: FieldReference | None) -> StatefulSet:
        """Build StatefulSet for Qdrant pods.

        Args:
            api_key_reference: Reference to the secret child's ``outputs.api_key``, which Qdrant
                requires from clients, or ``None`` for no key.

        Returns:
            Configured StatefulSet resource.
        """
        labels = self._labels()

        resources_config = None
        if self.config.resources:
            resources = self.resolved(self.config.resources)
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

        secret_env = (
            [SecretEnvVarConfig(name=API_KEY_ENVIRONMENT_VARIABLE, value=api_key_reference)]
            if api_key_reference is not None
            else None
        )

        container = ContainerConfig(
            name="qdrant",
            image=self.resolved(self.config.image),
            ports=[
                ContainerPortConfig(name="rest", container_port=6333),
                ContainerPortConfig(name="grpc", container_port=6334),
            ],
            secret_env=secret_env,
            volume_mounts=volume_mounts,
            resources=resources_config,
            readiness_probe=ProbeConfig(tcp_socket_port=6333),
            liveness_probe=ProbeConfig(tcp_socket_port=6333, initial_delay_seconds=30),
        )

        storage_class = None
        storage_size = "10Gi"

        if self.config.storage:
            storage = self.resolved(self.config.storage)
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

    def build_outputs(self, recorded_secret: Secret, recorded_client_service: Service) -> DatabaseOutputs:
        """Build the outputs from the READY children as the platform records them.

        Args:
            recorded_secret: The secret child as read.
            recorded_client_service: The client Service child as read.

        Returns:
            DatabaseOutputs with the client Service's external URLs and the secret child's key.

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
            api_key=extract_api_key(recorded_secret),
        )

    def log_pending_children(self, recorded_children: dict[str, Secret | StatefulSet | Service | None]) -> None:
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
        """Apply the four children, secret first, then report the outputs once all of them are READY.

        A READY secret child that holds no key although one was resolved is logged as
        ``qdrant_api_key_missing``.

        Returns:
            DatabaseOutputs when every child reads READY, else ``ChildrenPending()``.

        Raises:
            CallbackRefusedError: If the platform refuses an apply or a read of a child.
            RuntimeError: If the READY client Service reports no load-balancer address.
        """
        api_key = await self.resolve_api_key()

        secret = self.build_api_key_secret(api_key)
        headless_service = self._build_headless_service()
        api_key_reference = self.build_api_key_reference(secret) if api_key is not None else None
        statefulset = self._build_statefulset(api_key_reference)
        client_service = self._build_client_service()

        for child in (secret, headless_service, statefulset, client_service):
            await child.apply()

        recorded_secret = await secret.read()
        recorded_headless_service = await headless_service.read()
        recorded_statefulset = await statefulset.read()
        recorded_client_service = await client_service.read()
        recorded_children = {
            secret.id: recorded_secret,
            headless_service.id: recorded_headless_service,
            statefulset.id: recorded_statefulset,
            client_service.id: recorded_client_service,
        }

        if any(child is None or child.lifecycle_state != LifecycleState.READY for child in recorded_children.values()):
            self.log_pending_children(recorded_children)

            return ChildrenPending()

        ready_secret = cast(Secret, recorded_secret)

        if api_key is not None and extract_api_key(ready_secret) is None:
            logger.error("qdrant_api_key_missing resource=%s secret=%s", self.id, ready_secret.name)

        return self.build_outputs(ready_secret, cast(Service, recorded_client_service))

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

    async def vitals(self) -> Vitals:
        """Report the Qdrant database's vitals from its StatefulSet.

        Returns:
            Vitals of the underlying StatefulSet: nominal, impaired or down.
        """
        statefulset = self._build_statefulset(None)

        return await statefulset.vitals()

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
