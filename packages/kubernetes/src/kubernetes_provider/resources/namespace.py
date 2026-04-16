"""Kubernetes Namespace resource."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from lightkube import ApiError
from lightkube.models.meta_v1 import ObjectMeta
from lightkube.resources.core_v1 import Namespace as K8sNamespace
from pragma_sdk import Config, Field, HealthStatus, ImmutableDependency, LogEntry, Outputs, Resource

from kubernetes_provider.resources.config import KubernetesConfig


class NamespaceConfig(Config):
    """Configuration for a Kubernetes Namespace.

    Namespaces are cluster-scoped resources (no namespace field).

    Attributes:
        config: Kubernetes config dependency providing cluster access.
        labels: Optional labels to apply to the namespace.
    """

    config: ImmutableDependency[KubernetesConfig]
    labels: Field[dict[str, str]] | None = None


class NamespaceOutputs(Outputs):
    """Outputs from Kubernetes Namespace creation.

    Attributes:
        name: Namespace name.
    """

    name: str


class Namespace(Resource[NamespaceConfig, NamespaceOutputs]):
    """Kubernetes Namespace resource.

    Manages cluster-scoped Namespace objects for workload isolation. Namespaces
    do not belong to another namespace and have no ``namespace`` config field.

    Uses server-side apply with ``field_manager="pragma-kubernetes"`` for
    idempotent operations. Health checks verify the namespace exists and
    reports its phase (Active or Terminating).

    Lifecycle:
        - on_create: Apply namespace configuration
        - on_update: Apply updated namespace configuration (labels)
        - on_delete: Delete the namespace (idempotent)
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

    def _build_namespace(self) -> K8sNamespace:
        """Build Kubernetes Namespace object from config.

        Returns:
            Kubernetes Namespace object ready to apply.
        """
        return K8sNamespace(
            metadata=ObjectMeta(
                name=self.name,
                labels=self.config.labels,
            ),
        )

    def _build_outputs(self) -> NamespaceOutputs:
        """Build outputs.

        Returns:
            NamespaceOutputs with namespace name.
        """
        return NamespaceOutputs(name=self.name)

    async def on_create(self) -> NamespaceOutputs:
        """Create or update Kubernetes Namespace.

        Idempotent: Uses apply() which handles both create and update.

        Returns:
            NamespaceOutputs with namespace name.
        """
        async with self._get_client() as client:
            namespace = self._build_namespace()

            await client.apply(namespace, field_manager="pragma-kubernetes")

            return self._build_outputs()

    async def on_update(self, previous_config: NamespaceConfig) -> NamespaceOutputs:
        """Update Kubernetes Namespace.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            NamespaceOutputs with namespace name.
        """
        async with self._get_client() as client:
            namespace = self._build_namespace()

            await client.apply(namespace, field_manager="pragma-kubernetes")

            return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete Kubernetes Namespace.

        Idempotent: Succeeds if namespace doesn't exist.

        Raises:
            ApiError: If deletion fails for reasons other than not found.
        """
        async with self._get_client() as client:
            try:
                await client.delete(K8sNamespace, name=self.name)
            except ApiError as e:
                if e.status.code != 404:
                    raise

    @classmethod
    def upgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs

    @classmethod
    def downgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs

    async def health(self) -> HealthStatus:
        """Check Namespace health by verifying it exists and is active.

        Returns:
            HealthStatus indicating healthy/degraded/unhealthy.

        Raises:
            ApiError: If health check fails for reasons other than not found.
        """
        async with self._get_client() as client:
            try:
                ns = await client.get(K8sNamespace, name=self.name)

                phase = None

                if ns.status and ns.status.phase:
                    phase = ns.status.phase

                if phase == "Active":
                    return HealthStatus(
                        status="healthy",
                        message=f"Namespace {self.name} is active",
                        details={"phase": phase},
                    )

                return HealthStatus(
                    status="degraded",
                    message=f"Namespace {self.name} phase: {phase}",
                    details={"phase": phase},
                )

            except ApiError as e:
                if e.status.code == 404:
                    return HealthStatus(
                        status="unhealthy",
                        message="Namespace not found",
                    )
                raise

    async def logs(
        self,
        since: datetime | None = None,
        tail: int = 100,
    ) -> AsyncIterator[LogEntry]:
        """Namespaces do not produce logs.

        This method exists for interface compatibility and yields a single informational entry.

        Args:
            since: Ignored for namespaces.
            tail: Ignored for namespaces.

        Yields:
            A single informational LogEntry indicating that namespaces do not produce logs.
        """
        yield LogEntry(
            timestamp=datetime.now(UTC),
            level="info",
            message="Namespaces do not produce logs",
        )
        return
