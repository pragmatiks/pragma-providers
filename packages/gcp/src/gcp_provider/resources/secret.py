"""GCP Secret Manager resource."""

from __future__ import annotations

from typing import ClassVar

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import secretmanager
from pragma_sdk import Config, Outputs, Resource


class SecretConfig(Config):
    """Configuration for a GCP Secret Manager secret.

    Attributes:
        project_id: GCP project ID where the secret will be created.
        secret_id: Identifier for the secret within GCP (must be unique per project).
        data: Secret payload data to store.
    """

    project_id: str
    secret_id: str
    data: str


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

    Creates and manages secrets in GCP Secret Manager using Workload Identity
    for authentication (no service account keys required).

    Lifecycle:
        - on_create: Creates secret and initial version
        - on_update: Creates new version if data changed
        - on_delete: Deletes secret and all versions
    """

    provider: ClassVar[str] = "gcp"
    resource: ClassVar[str] = "secret"

    def _get_client(self) -> secretmanager.SecretManagerServiceClient:
        """Get Secret Manager client (uses ADC/Workload Identity).

        Returns:
            Configured Secret Manager client.
        """
        return secretmanager.SecretManagerServiceClient()

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

        # Try to create secret (idempotent via AlreadyExists handling)
        try:
            secret = client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": self.config.secret_id,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
        except AlreadyExists:
            # Secret exists - this is idempotent retry, get existing
            secret = client.get_secret(name=self._secret_path())

        # Add version with payload
        version = client.add_secret_version(
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

        Raises:
            ValueError: If project_id or secret_id changed (requires delete + create).
        """
        if previous_config.project_id != self.config.project_id:
            msg = "Cannot change project_id; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.secret_id != self.config.secret_id:
            msg = "Cannot change secret_id; delete and recreate resource"
            raise ValueError(msg)

        # If data unchanged, return existing outputs (idempotent)
        if previous_config.data == self.config.data and self.outputs is not None:
            return self.outputs

        # Add new version
        client = self._get_client()
        version = client.add_secret_version(
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
            client.delete_secret(name=self._secret_path())
        except NotFound:
            # Already deleted - idempotent success
            pass
