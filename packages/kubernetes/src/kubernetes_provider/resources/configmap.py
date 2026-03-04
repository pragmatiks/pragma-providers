"""Kubernetes ConfigMap resource."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import ClassVar

from gcp_provider import GKE
from lightkube import ApiError
from lightkube.models.meta_v1 import ObjectMeta
from lightkube.resources.core_v1 import ConfigMap as K8sConfigMap
from pragma_sdk import Config, Field, HealthStatus, ImmutableDependency, ImmutableField, LogEntry, Outputs, Resource

from kubernetes_provider.client import create_client_from_gke


class ConfigMapConfig(Config):
    """Configuration for a Kubernetes ConfigMap.

    Attributes:
        cluster: GKE cluster dependency providing Kubernetes credentials.
        namespace: Kubernetes namespace for the configmap (immutable after creation).
        data: Key-value pairs to store in the configmap.
    """

    cluster: ImmutableDependency[GKE]
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

    Uses server-side apply with ``field_manager="pragma-kubernetes"`` for
    idempotent operations. Health checks verify the ConfigMap exists and
    report the number of stored keys.

    Lifecycle:
        - on_create: Apply configmap configuration
        - on_update: Apply updated configmap configuration
        - on_delete: Delete the configmap (idempotent)
    """

    provider: ClassVar[str] = "kubernetes"
    resource: ClassVar[str] = "configmap"
    description = "Manages Kubernetes config maps for application configuration."

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

    def _build_configmap(self) -> K8sConfigMap:
        """Build Kubernetes ConfigMap object from config.

        Returns:
            Kubernetes ConfigMap object ready to apply.
        """
        return K8sConfigMap(
            metadata=ObjectMeta(
                name=self.name,
                namespace=self.config.namespace,
            ),
            data=self.config.data,
        )

    def _build_outputs(self) -> ConfigMapOutputs:
        """Build outputs.

        Returns:
            ConfigMapOutputs with configmap details.
        """
        return ConfigMapOutputs(
            name=self.name,
            namespace=self.config.namespace,
            data=self.config.data,
        )

    async def on_create(self) -> ConfigMapOutputs:
        """Create or update Kubernetes ConfigMap.

        Idempotent: Uses apply() which handles both create and update.

        Returns:
            ConfigMapOutputs with configmap details.
        """
        async with self._get_client() as client:
            configmap = self._build_configmap()

            await client.apply(configmap, field_manager="pragma-kubernetes")

            return self._build_outputs()

    async def on_update(self, previous_config: ConfigMapConfig) -> ConfigMapOutputs:
        """Update Kubernetes ConfigMap.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            ConfigMapOutputs with updated configmap details.
        """
        async with self._get_client() as client:
            configmap = self._build_configmap()

            await client.apply(configmap, field_manager="pragma-kubernetes")

            return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete Kubernetes ConfigMap.

        Idempotent: Succeeds if configmap doesn't exist.

        Raises:
            ApiError: If deletion fails for reasons other than not found.
        """
        async with self._get_client() as client:
            try:
                await client.delete(
                    K8sConfigMap,
                    name=self.name,
                    namespace=self.config.namespace,
                )
            except ApiError as e:
                if e.status.code != 404:
                    raise

    async def health(self) -> HealthStatus:
        """Check ConfigMap health by verifying it exists.

        Returns:
            HealthStatus indicating healthy/unhealthy.

        Raises:
            ApiError: If health check fails for reasons other than not found.
        """
        async with self._get_client() as client:
            try:
                configmap = await client.get(
                    K8sConfigMap,
                    name=self.name,
                    namespace=self.config.namespace,
                )

                key_count = len(configmap.data) if configmap.data else 0

                return HealthStatus(
                    status="healthy",
                    message=f"ConfigMap exists with {key_count} key(s)",
                    details={"key_count": key_count},
                )

            except ApiError as e:
                if e.status.code == 404:
                    return HealthStatus(
                        status="unhealthy",
                        message="ConfigMap not found",
                    )
                raise

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
