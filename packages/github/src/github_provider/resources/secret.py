"""GitHub Secret resource."""

from __future__ import annotations

import base64
from typing import Any

from nacl.encoding import Base64Encoder
from nacl.public import PublicKey, SealedBox
from pragma_sdk import Config, Field, ImmutableField, Outputs, Resource, SensitiveField

from github_provider.client import create_github_client, raise_for_status


class SecretConfig(Config):
    """Configuration for a GitHub secret.

    Secrets can be scoped to a repository or to a specific deployment
    environment within a repository. When ``environment_name`` is set,
    the secret is created as an environment secret.

    Attributes:
        access_token: GitHub personal access token for authentication.
        owner: GitHub user or organization that owns the repository.
        repository: Repository name.
        secret_name: Name of the secret. Must match the pattern
            ``[A-Z_][A-Z0-9_]*``.
        secret_value: Plaintext secret value. Encrypted automatically
            before upload using the repository or environment public key.
        environment_name: Optional deployment environment name. When set,
            the secret is scoped to this environment instead of the repository.
    """

    access_token: SensitiveField[str]
    owner: ImmutableField[str]
    repository: ImmutableField[str]
    secret_name: ImmutableField[str]
    secret_value: SensitiveField[str]
    environment_name: Field[str] | None = None


class SecretOutputs(Outputs):
    """Outputs from GitHub secret creation.

    Attributes:
        secret_name: Name of the secret.
        scope: Scope of the secret (``repository`` or ``environment``).
        environment_name: Environment name if this is an environment secret,
            empty string otherwise.
        created_at: ISO 8601 timestamp of when the secret was created.
        updated_at: ISO 8601 timestamp of when the secret was last updated.
    """

    secret_name: str
    scope: str
    environment_name: str
    created_at: str
    updated_at: str


def _encrypt_secret(public_key_b64: str, secret_value: str) -> str:
    """Encrypt a secret value using the repository's NaCl public key.

    Args:
        public_key_b64: Base64-encoded public key from the GitHub API.
        secret_value: Plaintext secret value to encrypt.

    Returns:
        Base64-encoded encrypted secret value.
    """
    public_key_bytes = base64.b64decode(public_key_b64)
    public_key = PublicKey(public_key_bytes)
    sealed_box = SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode(), encoder=Base64Encoder)

    return encrypted.decode("utf-8")


class Secret(Resource[SecretConfig, SecretOutputs]):
    """GitHub secret resource.

    Creates and manages secrets on GitHub repositories or deployment
    environments via the REST API. Secrets are encrypted using the
    repository or environment public key (NaCl sealed box) before upload.

    GitHub secrets cannot be read back after creation -- only the
    metadata (name, creation time, update time) is available.

    Lifecycle:
        - on_create: Fetches the public key, encrypts the secret value,
          and creates or replaces the secret. Idempotent -- creating an
          existing secret replaces its value.
        - on_update: Re-encrypts and replaces the secret value using PUT.
          Same operation as create.
        - on_delete: Deletes the secret. Idempotent -- succeeds if the
          secret does not exist.

    Example::

        resources:
          - name: deploy-key
            provider: github
            type: secret
            config:
              access_token:
                provider: pragma
                resource: secret
                name: github-token
                field: outputs.value
              owner: my-org
              repository: my-repo
              secret_name: DEPLOY_KEY
              secret_value:
                provider: pragma
                resource: secret
                name: deploy-key
                field: outputs.value

          - name: env-secret
            provider: github
            type: secret
            config:
              access_token:
                provider: pragma
                resource: secret
                name: github-token
                field: outputs.value
              owner: my-org
              repository: my-repo
              secret_name: API_KEY
              secret_value:
                provider: pragma
                resource: secret
                name: api-key
                field: outputs.value
              environment_name: production
    """

    def _is_environment_secret(self) -> bool:
        """Check whether this secret is scoped to an environment.

        Returns:
            True if environment_name is set.
        """
        return self.config.environment_name is not None

    async def _fetch_public_key(self, client: Any) -> tuple[str, str]:
        """Fetch the public key for secret encryption.

        Uses the environment public key endpoint when the secret is
        environment-scoped, otherwise uses the repository endpoint.

        Args:
            client: Authenticated GitHub API client.

        Returns:
            Tuple of (key_b64, key_id).
        """
        if self._is_environment_secret():
            response = await client.get(
                f"/repos/{self.config.owner}/{self.config.repository}"
                f"/environments/{self.config.environment_name}/secrets/public-key",
            )
        else:
            response = await client.get(
                f"/repos/{self.config.owner}/{self.config.repository}/actions/secrets/public-key",
            )

        await raise_for_status(response)
        data = response.json()

        return data["key"], data["key_id"]

    async def _put_secret(self, client: Any, encrypted_value: str, key_id: str) -> None:
        """Create or replace the secret with the encrypted value.

        Args:
            client: Authenticated GitHub API client.
            encrypted_value: Base64-encoded encrypted secret value.
            key_id: Public key ID used for encryption.
        """
        body = {
            "encrypted_value": encrypted_value,
            "key_id": key_id,
        }

        if self._is_environment_secret():
            response = await client.put(
                f"/repos/{self.config.owner}/{self.config.repository}"
                f"/environments/{self.config.environment_name}/secrets/{self.config.secret_name}",
                json=body,
            )
        else:
            response = await client.put(
                f"/repos/{self.config.owner}/{self.config.repository}/actions/secrets/{self.config.secret_name}",
                json=body,
            )

        await raise_for_status(response)

    async def _fetch_secret_metadata(self, client: Any) -> dict[str, Any]:
        """Fetch secret metadata (name, timestamps).

        Args:
            client: Authenticated GitHub API client.

        Returns:
            Secret metadata dictionary from the API.
        """
        if self._is_environment_secret():
            response = await client.get(
                f"/repos/{self.config.owner}/{self.config.repository}"
                f"/environments/{self.config.environment_name}/secrets/{self.config.secret_name}",
            )
        else:
            response = await client.get(
                f"/repos/{self.config.owner}/{self.config.repository}/actions/secrets/{self.config.secret_name}",
            )

        await raise_for_status(response)

        return response.json()

    def _build_outputs(self, metadata: dict[str, Any]) -> SecretOutputs:
        """Build outputs from secret metadata.

        Args:
            metadata: Secret metadata from the GitHub API.

        Returns:
            SecretOutputs with secret metadata.
        """
        scope = "environment" if self._is_environment_secret() else "repository"

        return SecretOutputs(
            secret_name=metadata["name"],
            scope=scope,
            environment_name=self.config.environment_name or "",
            created_at=metadata.get("created_at", ""),
            updated_at=metadata.get("updated_at", ""),
        )

    async def _apply_secret(self) -> SecretOutputs:
        """Encrypt and create or replace the secret.

        Returns:
            SecretOutputs with secret metadata.
        """
        client = create_github_client(self.config.access_token)

        try:
            key_b64, key_id = await self._fetch_public_key(client)
            encrypted_value = _encrypt_secret(key_b64, self.config.secret_value)
            await self._put_secret(client, encrypted_value, key_id)
            metadata = await self._fetch_secret_metadata(client)

            return self._build_outputs(metadata)
        finally:
            await client.aclose()

    async def on_create(self) -> SecretOutputs:
        """Create or replace a GitHub secret.

        Returns:
            SecretOutputs with secret metadata.
        """
        return await self._apply_secret()

    async def on_update(self, previous_config: SecretConfig) -> SecretOutputs:
        """Update the secret value.

        Secrets are replaced entirely on every update since the value
        cannot be read back for comparison.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            SecretOutputs with updated secret metadata.
        """
        return await self._apply_secret()

    async def on_delete(self) -> None:
        """Delete the GitHub secret.

        Idempotent: Succeeds if the secret does not exist.
        """
        if self.outputs is None:
            return

        client = create_github_client(self.config.access_token)

        try:
            if self._is_environment_secret():
                response = await client.delete(
                    f"/repos/{self.config.owner}/{self.config.repository}"
                    f"/environments/{self.config.environment_name}/secrets/{self.config.secret_name}",
                )
            else:
                response = await client.delete(
                    f"/repos/{self.config.owner}/{self.config.repository}/actions/secrets/{self.config.secret_name}",
                )

            if response.status_code == 404:
                return

            await raise_for_status(response)
        finally:
            await client.aclose()

    @classmethod
    def upgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs

    @classmethod
    def downgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs
