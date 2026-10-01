"""GCP Secret Manager resource."""

from __future__ import annotations

import json
from typing import Any, cast

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud.secretmanager_v1 import SecretManagerServiceAsyncClient, SecretVersion
from google.oauth2 import service_account
from pragma_sdk import Config, Field, ImmutableField, Outputs, Resource


class SecretConfig(Config):
    """Configuration for a GCP Secret Manager secret.

    Attributes:
        project_id: GCP project ID where the secret will be created.
        secret_id: Identifier for the secret within GCP (must be unique per project).
        data: Secret payload data to store.
        credentials: GCP service account credentials JSON object or string.
            Required for multi-tenant SaaS - no ADC fallback.
            Use a pragma/secret resource with a FieldReference to provide credentials.
    """

    project_id: ImmutableField[str]
    secret_id: ImmutableField[str]
    data: Field[str]
    credentials: Field[dict[str, Any] | str]


class SecretOutputs(Outputs):
    """Outputs from GCP Secret Manager secret creation.

    Attributes:
        resource_name: Full GCP resource name (projects/{project}/secrets/{id}).
        version_name: Full resource name of the latest enabled version, if any.
        version_id: The latest enabled version number as a string, if any.
    """

    resource_name: str
    version_name: str | None
    version_id: str | None


class Secret(Resource[SecretConfig, SecretOutputs]):
    """GCP Secret Manager secret resource.

    Creates and manages secrets in GCP Secret Manager using user-provided
    service account credentials (multi-tenant SaaS pattern). Secrets use
    automatic replication across GCP regions.

    Lifecycle:
        - on_create: Creates the secret and an initial version. Idempotent --
          if the secret already exists, adds a new version instead.
        - on_observe: Reads the secret and its latest enabled version.
        - on_update: Adds a new secret version when ``data`` differs from the
          latest enabled version. Previous versions are retained.
        - on_delete: Deletes the secret and all its versions. Idempotent --
          succeeds silently if the secret does not exist.

    Required IAM role: ``roles/secretmanager.admin``

    Required API: ``secretmanager.googleapis.com``

    Example::

        resources:
          - name: api-key
            provider: gcp
            type: secret
            config:
              project_id: my-project
              secret_id: api-key
              data: "sk-secret-value"
              credentials:
                provider: pragma
                resource: secret
                name: gcp-credentials
                field: outputs.credentials_json
    """

    def _get_client(self) -> SecretManagerServiceAsyncClient:
        """Get Secret Manager async client with user-provided credentials.

        Creates a client authenticated with the user's GCP service account
        credentials rather than using ADC/Workload Identity. This is required
        for multi-tenant SaaS where each user operates in their own GCP project.

        Returns:
            Configured Secret Manager async client using user's credentials.
        """
        creds_data = self.config.credentials

        if isinstance(creds_data, str):
            creds_data = json.loads(creds_data)

        credentials = service_account.Credentials.from_service_account_info(creds_data)
        return SecretManagerServiceAsyncClient(credentials=credentials)

    def _secret_path(self) -> str:
        """Build secret resource path.

        Returns:
            Full GCP resource path for this secret.
        """
        return f"projects/{self.config.project_id}/secrets/{self.config.secret_id}"

    def encode_payload(self) -> bytes:
        """Encode ``data`` as the payload bytes Secret Manager stores.

        Returns:
            The UTF-8 encoding of ``data``.
        """
        return cast(str, self.config.data).encode("utf-8")

    async def add_version(self, client: SecretManagerServiceAsyncClient, secret_name: str) -> SecretVersion:
        """Add a version holding ``data`` to the secret.

        Args:
            client: Secret Manager client.
            secret_name: Full resource name of the secret receiving the version.

        Returns:
            The new secret version.
        """
        return await client.add_secret_version(
            request={
                "parent": secret_name,
                "payload": {"data": self.encode_payload()},
            }
        )

    async def fetch_latest_enabled_version(self, client: SecretManagerServiceAsyncClient) -> SecretVersion | None:
        """Fetch the newest enabled version of the secret.

        Args:
            client: Secret Manager client.

        Returns:
            The newest enabled version, or None if the secret has none.
        """
        versions = await client.list_secret_versions(
            request={"parent": self._secret_path(), "filter": "state:ENABLED", "page_size": 1}
        )

        async for version in versions:
            return version

        return None

    @staticmethod
    async def fetch_payload(client: SecretManagerServiceAsyncClient, version: SecretVersion) -> bytes:
        """Fetch the payload a secret version holds.

        Args:
            client: Secret Manager client.
            version: Version to read.

        Returns:
            The version's payload bytes.
        """
        response = await client.access_secret_version(name=version.name)
        return response.payload.data

    @staticmethod
    def build_outputs(secret_name: str, version: SecretVersion | None) -> SecretOutputs:
        """Build outputs from the secret name and its current version.

        Args:
            secret_name: Full resource name of the secret.
            version: Latest enabled version of the secret, or None if it has none.

        Returns:
            SecretOutputs with resource name and version info.
        """
        if version is None:
            return SecretOutputs(resource_name=secret_name, version_name=None, version_id=None)

        return SecretOutputs(
            resource_name=secret_name,
            version_name=version.name,
            version_id=version.name.split("/")[-1],
        )

    async def on_create(self) -> SecretOutputs:
        """Create GCP secret with initial version.

        Idempotent: If secret already exists, adds a new version.

        Returns:
            SecretOutputs with resource name and version info.
        """
        client = self._get_client()
        parent = f"projects/{self.config.project_id}"

        try:
            secret = await client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": self.config.secret_id,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
        except AlreadyExists:
            secret = await client.get_secret(name=self._secret_path())

        version = await self.add_version(client, secret.name)

        return self.build_outputs(secret.name, version)

    async def on_observe(self) -> SecretOutputs | None:
        """Read the secret and its latest enabled version.

        Returns:
            SecretOutputs for the secret, or None if it does not exist.
        """
        client = self._get_client()

        try:
            secret = await client.get_secret(name=self._secret_path())
        except NotFound:
            return None

        version = await self.fetch_latest_enabled_version(client)

        return self.build_outputs(secret.name, version)

    async def on_update(self, previous_config: SecretConfig | None) -> SecretOutputs:
        """Add a secret version when the latest enabled payload differs from ``data``.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            SecretOutputs with the current version info.
        """
        client = self._get_client()
        secret = await client.get_secret(name=self._secret_path())
        version = await self.fetch_latest_enabled_version(client)

        if version is None or await self.fetch_payload(client, version) != self.encode_payload():
            version = await self.add_version(client, secret.name)

        return self.build_outputs(secret.name, version)

    async def on_delete(self) -> None:
        """Delete secret and all versions.

        Idempotent: Succeeds if secret doesn't exist.
        """
        client = self._get_client()

        try:
            await client.delete_secret(name=self._secret_path())
        except NotFound:
            pass
