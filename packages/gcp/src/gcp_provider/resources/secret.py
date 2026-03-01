"""GCP Secret Manager resource."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud.secretmanager_v1 import SecretManagerServiceAsyncClient
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
        version_name: Full version resource name including version number.
        version_id: The version number as a string.
    """

    resource_name: str
    version_name: str
    version_id: str


class Secret(Resource[SecretConfig, SecretOutputs]):
    """GCP Secret Manager secret resource.

    Creates and manages secrets in GCP Secret Manager using user-provided
    service account credentials (multi-tenant SaaS pattern).

    Lifecycle:
        - on_create: Creates secret and initial version
        - on_update: Creates new version if data changed
        - on_delete: Deletes secret and all versions
    """

    provider: ClassVar[str] = "gcp"
    resource: ClassVar[str] = "secret"

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

        version = await client.add_secret_version(
            request={
                "parent": secret.name,
                "payload": {"data": self.config.data.encode("utf-8")},
            }
        )

        return SecretOutputs(
            resource_name=secret.name,
            version_name=version.name,
            version_id=version.name.split("/")[-1],
        )

    async def on_update(self, previous_config: SecretConfig) -> SecretOutputs:
        """Update secret by creating new version if data changed.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            SecretOutputs with updated version info.
        """
        if previous_config.data == self.config.data and self.outputs is not None:
            return self.outputs

        client = self._get_client()
        version = await client.add_secret_version(
            request={
                "parent": self._secret_path(),
                "payload": {"data": self.config.data.encode("utf-8")},
            }
        )

        return SecretOutputs(
            resource_name=self._secret_path(),
            version_name=version.name,
            version_id=version.name.split("/")[-1],
        )

    async def on_delete(self) -> None:
        """Delete secret and all versions.

        Idempotent: Succeeds if secret doesn't exist.
        """
        client = self._get_client()

        try:
            await client.delete_secret(name=self._secret_path())
        except NotFound:
            pass
