"""Kubernetes Secret resource."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import ClassVar

from gcp_provider import GKE
from lightkube import ApiError
from lightkube.models.meta_v1 import ObjectMeta
from lightkube.resources.core_v1 import Secret as K8sSecret
from pragma_sdk import Config, Field, HealthStatus, ImmutableDependency, ImmutableField, LogEntry, Outputs, Resource

from kubernetes_provider.client import create_client_from_gke


class SecretConfig(Config):
    """Configuration for a Kubernetes Secret.

    Attributes:
        cluster: GKE cluster dependency providing Kubernetes credentials.
        namespace: Kubernetes namespace for the secret (immutable after creation).
        type: Secret type (e.g., ``Opaque``, ``kubernetes.io/tls``).
        data: Key-value pairs that will be base64-encoded before storage.
        string_data: Plain-text key-value pairs (Kubernetes base64-encodes them on the server).
    """

    cluster: ImmutableDependency[GKE]
    namespace: ImmutableField[str] = "default"
    type: Field[str] = "Opaque"
    data: Field[dict[str, str]] | None = None
    string_data: Field[dict[str, str]] | None = None


class SecretOutputs(Outputs):
    """Outputs from Kubernetes Secret creation.

    Attributes:
        name: Secret name as created in the cluster.
        namespace: Kubernetes namespace containing the secret.
        type: Kubernetes secret type (e.g., ``Opaque``).
        data: Merged key-value pairs from both ``data`` and ``string_data`` inputs (plain text).
    """

    name: str
    namespace: str
    type: str
    data: dict[str, str]


class Secret(Resource[SecretConfig, SecretOutputs]):
    """Kubernetes Secret resource.

    Stores sensitive data (credentials, tokens, TLS certificates) as
    base64-encoded key-value pairs. Supports both ``data`` (auto-encoded)
    and ``string_data`` (server-encoded) inputs.

    Uses server-side apply with ``field_manager="pragma-kubernetes"`` for
    idempotent operations. Health checks verify the secret exists and
    report the number of stored keys.

    Lifecycle:
        - on_create: Apply secret configuration
        - on_update: Apply updated secret configuration
        - on_delete: Delete the secret (idempotent)
    """

    provider: ClassVar[str] = "kubernetes"
    resource: ClassVar[str] = "secret"
    description = "Manages Kubernetes secrets for sensitive data."

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

    def _encode_data(self, data: dict[str, str]) -> dict[str, str]:
        """Base64 encode data values.

        Args:
            data: Plain text key-value pairs.

        Returns:
            Base64-encoded key-value pairs.
        """
        return {k: base64.b64encode(v.encode()).decode() for k, v in data.items()}

    def _build_secret(self) -> K8sSecret:
        """Build Kubernetes Secret object from config.

        Returns:
            Kubernetes Secret object ready to apply.
        """
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
        """Build outputs with decoded data.

        Returns:
            SecretOutputs with secret details.
        """
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
        async with self._get_client() as client:
            secret = self._build_secret()

            await client.apply(secret, field_manager="pragma-kubernetes")

            return self._build_outputs()

    async def on_update(self, previous_config: SecretConfig) -> SecretOutputs:
        """Update Kubernetes Secret.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            SecretOutputs with updated secret details.
        """
        async with self._get_client() as client:
            secret = self._build_secret()

            await client.apply(secret, field_manager="pragma-kubernetes")

            return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete Kubernetes Secret.

        Idempotent: Succeeds if secret doesn't exist.

        Raises:
            ApiError: If deletion fails for reasons other than not found.
        """
        async with self._get_client() as client:
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

        Raises:
            ApiError: If health check fails for reasons other than not found.
        """
        async with self._get_client() as client:
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
            timestamp=datetime.now(UTC),
            level="info",
            message="Secrets do not produce logs",
        )
        return
