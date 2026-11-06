"""Kubernetes ConfigMap resource."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import cast

from lightkube.resources.core_v1 import ConfigMap as K8sConfigMap
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


class ConfigMapConfig(Config):
    """Configuration for a Kubernetes ConfigMap.

    Attributes:
        config: Kubernetes config dependency providing cluster access.
        namespace: Kubernetes namespace for the configmap (immutable after creation).
        data: Key-value pairs to store in the configmap.
    """

    config: ImmutableDependency[KubernetesConfig]
    namespace: ImmutableField[str] = "default"
    data: Field[dict[str, str]]


class ConfigMapOutputs(Outputs):
    """Outputs from Kubernetes ConfigMap creation.

    Attributes:
        name: ConfigMap name as created in the cluster.
        namespace: Kubernetes namespace containing the configmap.
        data: Key-value pairs stored in the configmap.
    """

    name: str
    namespace: str
    data: dict[str, str]


class ConfigMap(Resource[ConfigMapConfig, ConfigMapOutputs]):
    """Kubernetes ConfigMap resource.

    Stores non-sensitive configuration data as key-value pairs that can be
    mounted as files or exposed as environment variables in pods.

    Creates the ConfigMap, marked with this resource's ``pragma.io/resource``
    annotation; an existing ConfigMap carrying it is this resource's earlier
    create and is applied over, any other is refused. Updates use server-side
    apply under ``field_manager="pragma-kubernetes"``, forcing conflicts. Health checks
    verify the ConfigMap exists and report the number of stored keys.

    Lifecycle:
        - on_create: Create the configmap; apply over this resource's earlier
          create, refuse any other existing configmap
        - on_observe: Look up the configmap by name and namespace; raise AlreadyExistsError for one
          without this resource's annotation
        - on_update: Apply updated configmap configuration
        - on_delete: Delete the configmap and wait until it is gone (idempotent)
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

    def _build_configmap(self) -> K8sConfigMap:
        """Build Kubernetes ConfigMap object from config.

        Returns:
            Kubernetes ConfigMap object ready to apply.
        """
        return K8sConfigMap(
            metadata=build_object_metadata(self.id, self.name, cast(str, self.config.namespace)),
            data=cast(dict[str, str], self.config.data),
        )

    def build_outputs(self, configmap: K8sConfigMap) -> ConfigMapOutputs:
        """Build outputs from the ConfigMap as stored in the cluster.

        Args:
            configmap: ConfigMap returned by the cluster.

        Returns:
            ConfigMapOutputs with configmap details.
        """
        return ConfigMapOutputs(
            name=self.name,
            namespace=cast(str, self.config.namespace),
            data=configmap.data or {},
        )

    async def on_create(self) -> ConfigMapOutputs:
        """Create the Kubernetes ConfigMap.

        Returns:
            ConfigMapOutputs with configmap details.

        Raises:
            AlreadyExistsError: If the namespace holds a configmap of this name that this resource
                did not create.
            RuntimeError: If this resource's earlier configmap is being deleted.
        """
        desired = self._build_configmap()

        async with self._get_client() as client:
            try:
                configmap = await create_object(client, desired)
            except AlreadyExistsError as error:
                await check_create_conflict(client, desired, self.id, error)
            else:
                return self.build_outputs(configmap)

        return await self.on_update(None)

    async def on_observe(self) -> ConfigMapOutputs | None:
        """Look up the Kubernetes ConfigMap named after this resource.

        Returns:
            ConfigMapOutputs, or ``None`` if the configmap does not exist.

        Raises:
            AlreadyExistsError: If the configmap carries no ``pragma.io/resource`` annotation naming this
                resource.
        """
        async with self._get_client() as client:
            configmap = await fetch_own_object(
                client, K8sConfigMap, self.id, self.name, cast(str, self.config.namespace), MarkerRefusal.OBSERVE
            )

        if configmap is None:
            return None

        return self.build_outputs(configmap)

    async def on_update(self, previous_config: ConfigMapConfig | None) -> ConfigMapOutputs:
        """Apply the full desired configmap configuration.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            ConfigMapOutputs with updated configmap details.
        """
        async with self._get_client() as client:
            configmap = await client.apply(self._build_configmap(), field_manager=FIELD_MANAGER, force=True)

        return self.build_outputs(configmap)

    async def on_delete(self) -> None:
        """Delete Kubernetes ConfigMap and wait until it is gone.

        Idempotent: Succeeds if configmap doesn't exist.

        Raises:
            TimeoutError: If the configmap is still present after the deletion timeout.
        """
        async with self._get_client() as client:
            await delete_object(client, K8sConfigMap, self.name, cast(str, self.config.namespace))
            await wait_for_absence(client, K8sConfigMap, self.name, cast(str, self.config.namespace))

    async def health(self) -> HealthStatus:
        """Check ConfigMap health by verifying it exists.

        Returns:
            HealthStatus indicating healthy/unhealthy.
        """
        async with self._get_client() as client:
            configmap = await fetch_object(client, K8sConfigMap, self.name, cast(str, self.config.namespace))

        if configmap is None:
            return HealthStatus(
                status="unhealthy",
                message=f"ConfigMap {self.name} not found",
            )

        key_count = len(configmap.data) if configmap.data else 0

        return HealthStatus(
            status="healthy",
            message=f"ConfigMap exists with {key_count} key(s)",
            details={"key_count": key_count},
        )

    async def logs(
        self,
        since: datetime | None = None,
        tail: int = 100,
    ) -> AsyncIterator[LogEntry]:
        """ConfigMaps do not produce logs.

        This method exists for interface compatibility but yields nothing.

        Args:
            since: Ignored for configmaps.
            tail: Ignored for configmaps.

        Yields:
            Nothing - configmaps don't have logs.
        """
        yield LogEntry(
            timestamp=datetime.now(UTC),
            level="info",
            message="ConfigMaps do not produce logs",
        )
        return
