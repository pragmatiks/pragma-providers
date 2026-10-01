"""Kubernetes Namespace resource."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from lightkube.models.meta_v1 import ObjectMeta
from lightkube.resources.core_v1 import Namespace as K8sNamespace
from pragma_sdk import Config, Field, HealthStatus, ImmutableDependency, LogEntry, Outputs, Resource

from kubernetes_provider.objects import delete_object, fetch_object, wait_for_absence
from kubernetes_provider.resources.config import KubernetesConfig


NAMESPACE_DELETION_TIMEOUT_SECONDS = 900


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
        - on_observe: Look up the namespace by name
        - on_update: Apply updated namespace configuration (labels)
        - on_delete: Delete the namespace and wait until it is gone (idempotent)
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

    def build_outputs(self, namespace: K8sNamespace) -> NamespaceOutputs:
        """Build outputs from the Namespace as read from the cluster.

        Args:
            namespace: Namespace returned by the cluster.

        Returns:
            NamespaceOutputs with namespace name.
        """
        assert namespace.metadata is not None
        assert namespace.metadata.name is not None

        return NamespaceOutputs(name=namespace.metadata.name)

    async def on_create(self) -> NamespaceOutputs:
        """Create or update Kubernetes Namespace.

        Idempotent: Uses apply() which handles both create and update.

        Returns:
            NamespaceOutputs with namespace name.
        """
        async with self._get_client() as client:
            namespace = await client.apply(self._build_namespace(), field_manager="pragma-kubernetes")

        return self.build_outputs(namespace)

    async def on_observe(self) -> NamespaceOutputs | None:
        """Look up the Kubernetes Namespace named after this resource.

        Returns:
            NamespaceOutputs, or ``None`` if the namespace does not exist.
        """
        async with self._get_client() as client:
            namespace = await fetch_object(client, K8sNamespace, self.name)

        if namespace is None:
            return None

        return self.build_outputs(namespace)

    async def on_update(self, previous_config: NamespaceConfig | None) -> NamespaceOutputs:
        """Apply the full desired namespace configuration.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            NamespaceOutputs with namespace name.
        """
        return await self.on_create()

    async def on_delete(self) -> None:
        """Delete Kubernetes Namespace and wait until it and everything in it is gone.

        Idempotent: Succeeds if namespace doesn't exist. Waits up to 900 s, since the cluster
        finalizes every object in the namespace first.

        Raises:
            TimeoutError: If the namespace is still present after the deletion timeout.
        """
        async with self._get_client() as client:
            await delete_object(client, K8sNamespace, self.name)
            await wait_for_absence(client, K8sNamespace, self.name, timeout_seconds=NAMESPACE_DELETION_TIMEOUT_SECONDS)

    async def health(self) -> HealthStatus:
        """Check Namespace health by verifying it exists and is active.

        Returns:
            HealthStatus indicating healthy/degraded/unhealthy.
        """
        async with self._get_client() as client:
            namespace = await fetch_object(client, K8sNamespace, self.name)

        if namespace is None:
            return HealthStatus(
                status="unhealthy",
                message=f"Namespace {self.name} not found",
            )

        phase = namespace.status.phase if namespace.status else None

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
