"""Kubernetes Secret resource."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import ClassVar

from gcp_provider import GKE
from lightkube import ApiError
from lightkube.models.meta_v1 import ObjectMeta
from lightkube.resources.core_v1 import Secret as K8sSecret
from pragma_sdk import Config, Dependency, HealthStatus, LogEntry, Outputs, Resource

from kubernetes_provider.client import create_client_from_gke


class SecretConfig(Config):
    """Configuration for a Kubernetes Secret.

    Attributes:
        cluster: GKE cluster dependency providing Kubernetes credentials.
        namespace: Kubernetes namespace for the secret.
        type: Secret type (Opaque, kubernetes.io/tls, etc.).
        data: Base64-encoded key-value pairs (will be encoded if not already).
        string_data: Plain text key-value pairs (Kubernetes will base64 encode).
    """

    cluster: Dependency[GKE]
    namespace: str = "default"
    type: str = "Opaque"
    data: dict[str, str] | None = None
    string_data: dict[str, str] | None = None


class SecretOutputs(Outputs):
    """Outputs from Kubernetes Secret creation.

    Attributes:
        name: Secret name.
        namespace: Kubernetes namespace.
        type: Secret type.
        data: Key-value pairs stored in the secret (decoded from base64).
    """

    name: str
    namespace: str
    type: str
    data: dict[str, str]


class Secret(Resource[SecretConfig, SecretOutputs]):
    """Kubernetes Secret resource.

    Creates and manages Kubernetes Secrets using lightkube.
    Data values are base64 encoded automatically.

    Lifecycle:
        - on_create: Apply secret configuration
        - on_update: Apply updated secret configuration
        - on_delete: Delete the secret
    """

    provider: ClassVar[str] = "kubernetes"
    resource: ClassVar[str] = "secret"

    async def _get_client(self):
        """Get lightkube client from GKE cluster credentials."""
        cluster = await self.config.cluster.resolve()
        outputs = cluster.outputs

        if outputs is None:
            msg = "GKE cluster outputs not available"
            raise RuntimeError(msg)

        creds = cluster.config.credentials

        return create_client_from_gke(outputs, creds)

    def _encode_data(self, data: dict[str, str]) -> dict[str, str]:
        """Base64 encode data values.

        Args:
            data: Plain text key-value pairs.

        Returns:
            Base64-encoded key-value pairs.
        """
        return {k: base64.b64encode(v.encode()).decode() for k, v in data.items()}

    def _build_secret(self) -> K8sSecret:
        """Build Kubernetes Secret object from config."""
        secret = K8sSecret(
            metadata=ObjectMeta(
                name=self.name,
                namespace=self.config.namespace,
            ),
            type=self.config.type,
        )

        if self.config.data:
            secret.data = self._encode_data(self.config.data)

        if self.config.string_data:
            secret.stringData = self.config.string_data

        return secret

    def _build_outputs(self) -> SecretOutputs:
        """Build outputs with decoded data."""
        merged_data: dict[str, str] = {}

        if self.config.data:
            merged_data.update(self.config.data)

        if self.config.string_data:
            merged_data.update(self.config.string_data)

        return SecretOutputs(
            name=self.name,
            namespace=self.config.namespace,
            type=self.config.type,
            data=merged_data,
        )

    async def on_create(self) -> SecretOutputs:
        """Create or update Kubernetes Secret.

        Idempotent: Uses apply() which handles both create and update.

        Returns:
            SecretOutputs with secret details.
        """
        client = await self._get_client()
        secret = self._build_secret()

        await client.apply(secret, field_manager="pragma-kubernetes")

        return self._build_outputs()

    async def on_update(self, previous_config: SecretConfig) -> SecretOutputs:
        """Update Kubernetes Secret.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            SecretOutputs with updated secret details.

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
        secret = self._build_secret()

        await client.apply(secret, field_manager="pragma-kubernetes")

        return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete Kubernetes Secret.

        Idempotent: Succeeds if secret doesn't exist.
        """
        client = await self._get_client()

        try:
            await client.delete(
                K8sSecret,
                name=self.name,
                namespace=self.config.namespace,
            )
        except ApiError as e:
            if e.status.code != 404:
                raise

    async def health(self) -> HealthStatus:
        """Check Secret health by verifying it exists.

        Returns:
            HealthStatus indicating healthy/unhealthy.
        """
        client = await self._get_client()

        try:
            secret = await client.get(
                K8sSecret,
                name=self.name,
                namespace=self.config.namespace,
            )

            key_count = len(secret.data) if secret.data else 0

            return HealthStatus(
                status="healthy",
                message=f"Secret exists with {key_count} key(s)",
                details={"key_count": key_count, "type": secret.type},
            )

        except ApiError as e:
            if e.status.code == 404:
                return HealthStatus(
                    status="unhealthy",
                    message="Secret not found",
                )
            raise

    async def logs(
        self,
        since: datetime | None = None,
        tail: int = 100,
    ) -> AsyncIterator[LogEntry]:
        """Secrets do not produce logs.

        This method exists for interface compatibility but yields nothing.

        Args:
            since: Ignored for secrets.
            tail: Ignored for secrets.

        Yields:
            Nothing - secrets don't have logs.
        """
        yield LogEntry(
            timestamp=datetime.now(timezone.utc),
            level="info",
            message="Secrets do not produce logs",
        )
        return
