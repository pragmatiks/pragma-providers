"""Kubernetes ConfigMap resource."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import ClassVar

from gcp_provider import GKE
from lightkube import ApiError
from lightkube.models.meta_v1 import ObjectMeta
from lightkube.resources.core_v1 import ConfigMap as K8sConfigMap
from pragma_sdk import Config, Dependency, HealthStatus, LogEntry, Outputs, Resource

from kubernetes_provider.client import create_client_from_gke


class ConfigMapConfig(Config):
    """Configuration for a Kubernetes ConfigMap.

    Attributes:
        cluster: GKE cluster dependency providing Kubernetes credentials.
        namespace: Kubernetes namespace for the configmap.
        data: Key-value pairs to store in the configmap.
    """

    cluster: Dependency[GKE]
    namespace: str = "default"
    data: dict[str, str]


class ConfigMapOutputs(Outputs):
    """Outputs from Kubernetes ConfigMap creation.

    Attributes:
        name: ConfigMap name.
        namespace: Kubernetes namespace.
        data: Key-value pairs stored in the configmap.
    """

    name: str
    namespace: str
    data: dict[str, str]


class ConfigMap(Resource[ConfigMapConfig, ConfigMapOutputs]):
    """Kubernetes ConfigMap resource.

    Creates and manages Kubernetes ConfigMaps using lightkube.

    Lifecycle:
        - on_create: Apply configmap configuration
        - on_update: Apply updated configmap configuration
        - on_delete: Delete the configmap
    """

    provider: ClassVar[str] = "kubernetes"
    resource: ClassVar[str] = "configmap"

    async def _get_client(self):
        """Get lightkube client from GKE cluster credentials."""
        cluster = await self.config.cluster.resolve()
        outputs = cluster.outputs

        if outputs is None:
            msg = "GKE cluster outputs not available"
            raise RuntimeError(msg)

        creds = cluster.config.credentials

        return create_client_from_gke(outputs, creds)

    def _build_configmap(self) -> K8sConfigMap:
        """Build Kubernetes ConfigMap object from config."""
        return K8sConfigMap(
            metadata=ObjectMeta(
                name=self.name,
                namespace=self.config.namespace,
            ),
            data=self.config.data,
        )

    def _build_outputs(self) -> ConfigMapOutputs:
        """Build outputs."""
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
        client = await self._get_client()
        configmap = self._build_configmap()

        await client.apply(configmap, field_manager="pragma-kubernetes")

        return self._build_outputs()

    async def on_update(self, previous_config: ConfigMapConfig) -> ConfigMapOutputs:
        """Update Kubernetes ConfigMap.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            ConfigMapOutputs with updated configmap details.

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
        configmap = self._build_configmap()

        await client.apply(configmap, field_manager="pragma-kubernetes")

        return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete Kubernetes ConfigMap.

        Idempotent: Succeeds if configmap doesn't exist.
        """
        client = await self._get_client()

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
        """
        client = await self._get_client()

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
            timestamp=datetime.now(timezone.utc),
            level="info",
            message="ConfigMaps do not produce logs",
        )
        return
