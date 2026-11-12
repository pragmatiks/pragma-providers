"""Kubernetes Secret resource."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from lightkube.resources.core_v1 import Secret as K8sSecret
from pragma_sdk import (
    AlreadyExistsError,
    Config,
    Field,
    ImmutableDependency,
    ImmutableField,
    LogEntry,
    Outputs,
    Resource,
    Vitals,
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


class SecretConfig(Config):
    """Configuration for a Kubernetes Secret.

    Attributes:
        config: Kubernetes config dependency providing cluster access.
        namespace: Kubernetes namespace for the secret (immutable after creation).
        type: Secret type (e.g., ``Opaque``, ``kubernetes.io/tls``).
        data: Key-value pairs that will be base64-encoded before storage.
        string_data: Plain-text key-value pairs (Kubernetes base64-encodes them on the server).
    """

    config: ImmutableDependency[KubernetesConfig]
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

    Creates the secret, marked with this resource's ``pragma.io/resource``
    annotation; an existing secret carrying it is this resource's earlier
    create and is applied over, any other is refused. Updates use server-side
    apply under ``field_manager="pragma-kubernetes"``, forcing conflicts. Vitals
    verify the secret exists and report the number of stored keys.

    Lifecycle:
        - on_create: Create the secret; apply over this resource's earlier
          create, refuse any other existing secret
        - on_observe: Look up the secret by name and namespace; raise AlreadyExistsError for one
          without this resource's annotation
        - on_update: Apply updated secret configuration
        - on_delete: Delete the secret and wait until it is gone (idempotent)
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
            metadata=build_object_metadata(self.id, self.name, self.resolved(self.config.namespace)),
            type=self.resolved(self.config.type),
        )

        if self.config.data:
            secret.data = self._encode_data(self.resolved(self.config.data))

        if self.config.string_data:
            secret.stringData = self.resolved(self.config.string_data)

        return secret

    def build_outputs(self, secret: K8sSecret) -> SecretOutputs:
        """Build outputs from the Secret as stored in the cluster, reporting only declared keys.

        Args:
            secret: Secret returned by the cluster.

        Returns:
            SecretOutputs with secret details and plain-text data.

        Raises:
            UnicodeDecodeError: If a declared key holds a value that is not UTF-8 text.
        """
        declared_keys = {*(self.config.data or {}), *(self.config.string_data or {})}

        return SecretOutputs(
            name=self.name,
            namespace=self.resolved(self.config.namespace),
            type=secret.type or "Opaque",
            data={
                key: base64.b64decode(value).decode()
                for key, value in (secret.data or {}).items()
                if key in declared_keys
            },
        )

    async def on_create(self) -> SecretOutputs:
        """Create the Kubernetes Secret.

        Returns:
            SecretOutputs with secret details.

        Raises:
            AlreadyExistsError: If the namespace holds a secret of this name that this resource
                did not create.
            RuntimeError: If this resource's earlier secret is being deleted.
        """
        desired = self._build_secret()

        async with self._get_client() as client:
            try:
                secret = await create_object(client, desired)
            except AlreadyExistsError as error:
                await check_create_conflict(client, desired, self.id, error)
            else:
                return self.build_outputs(secret)

        return await self.on_update(None)

    async def on_observe(self) -> SecretOutputs | None:
        """Look up the Kubernetes Secret named after this resource.

        Returns:
            SecretOutputs, or ``None`` if the secret does not exist.

        Raises:
            AlreadyExistsError: If the secret carries no ``pragma.io/resource`` annotation naming this
                resource.
        """
        async with self._get_client() as client:
            secret = await fetch_own_object(
                client, K8sSecret, self.id, self.name, self.resolved(self.config.namespace), MarkerRefusal.OBSERVE
            )

        if secret is None:
            return None

        return self.build_outputs(secret)

    async def on_update(self, previous_config: SecretConfig | None) -> SecretOutputs:
        """Apply the full desired secret configuration.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            SecretOutputs with updated secret details.
        """
        async with self._get_client() as client:
            secret = await client.apply(self._build_secret(), field_manager=FIELD_MANAGER, force=True)

        return self.build_outputs(secret)

    async def on_delete(self) -> None:
        """Delete Kubernetes Secret and wait until it is gone.

        Idempotent: Succeeds if secret doesn't exist.

        Raises:
            TimeoutError: If the secret is still present after the deletion timeout.
        """
        async with self._get_client() as client:
            await delete_object(client, K8sSecret, self.name, self.resolved(self.config.namespace))
            await wait_for_absence(client, K8sSecret, self.name, self.resolved(self.config.namespace))

    async def vitals(self) -> Vitals:
        """Report the Secret's vitals from its existence and key count.

        Returns:
            Vitals: nominal when the Secret exists, down when it is absent.
        """
        async with self._get_client() as client:
            secret = await fetch_object(client, K8sSecret, self.name, self.resolved(self.config.namespace))

        if secret is None:
            return Vitals(
                status="down",
                message=f"Secret {self.name} not found",
            )

        key_count = len(secret.data) if secret.data else 0

        return Vitals(
            status="nominal",
            message=f"Secret exists with {key_count} key(s)",
            details={"key_count": key_count, "type": secret.type},
        )

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
