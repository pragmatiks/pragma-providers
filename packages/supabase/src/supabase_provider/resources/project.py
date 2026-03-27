"""Supabase Project resource."""

from __future__ import annotations

import asyncio

import httpx
from pragma_sdk import Config, Field, ImmutableField, Outputs, Resource, SensitiveField

from supabase_provider.client import create_management_client, raise_for_status


class ProjectConfig(Config):
    """Configuration for a Supabase project.

    Attributes:
        access_token: Supabase personal access token for the Management API.
            Use a pragma/secret resource with a FieldReference to provide this.
        organization_id: Supabase organization ID that owns the project.
        name: Display name for the project.
        region: Deployment region (e.g., ``us-east-1``, ``eu-west-1``).
            Use ``GET /v1/projects/available-regions`` to list valid regions.
        database_password: Password for the project's PostgreSQL database.
    """

    access_token: SensitiveField[str]
    organization_id: ImmutableField[str]
    name: Field[str]
    region: ImmutableField[str]
    database_password: SensitiveField[str]


class ProjectOutputs(Outputs):
    """Outputs from Supabase project creation.

    Attributes:
        project_ref: Unique project reference ID used in all API calls.
        name: Project display name.
        organization_id: Owning organization ID.
        region: Deployment region.
        status: Current project status (e.g., ACTIVE_HEALTHY, COMING_UP).
        endpoint: Project API endpoint URL.
        anon_key: Public anonymous API key for client-side access.
        service_role_key: Service role API key for server-side access.
    """

    project_ref: str
    name: str
    organization_id: str
    region: str
    status: str
    endpoint: str
    anon_key: str
    service_role_key: str


_POLL_INTERVAL_SECONDS = 10
_MAX_POLL_ATTEMPTS = 60  # 60 * 10s = 10 minutes max wait


class Project(Resource[ProjectConfig, ProjectOutputs]):
    """Supabase project resource.

    Creates and manages Supabase projects via the Management API. Each project
    includes a PostgreSQL database, authentication, and API endpoints.

    Lifecycle:
        - on_create: Creates a new project and polls until ACTIVE_HEALTHY.
          Not idempotent -- duplicate calls create duplicate projects.
        - on_update: Updates the project name if changed. Region and
          organization are immutable.
        - on_delete: Deletes the project. Idempotent -- succeeds if the
          project does not exist.

    Example::

        resources:
          - name: my-app
            provider: supabase
            type: project
            config:
              access_token:
                provider: pragma
                resource: secret
                name: supabase-token
                field: outputs.value
              organization_id: org_abc123
              name: my-app
              region: eu-west-1
              database_password:
                provider: pragma
                resource: secret
                name: supabase-db-password
                field: outputs.value
    """

    async def _wait_for_healthy(self, client: httpx.AsyncClient, project_ref: str) -> dict:
        """Poll project health until services are ready.

        Args:
            client: Authenticated Management API client.
            project_ref: Project reference ID.

        Returns:
            Project data dictionary from the API.

        Raises:
            TimeoutError: If the project does not become healthy in time.
        """
        for _ in range(_MAX_POLL_ATTEMPTS):
            response = await client.get(f"/projects/{project_ref}/health")

            if response.status_code in {401, 403, 404}:
                await raise_for_status(response)

            if response.is_success:
                health_data = response.json()
                services = {item["name"]: item["status"] for item in health_data}

                if services and all(status == "ACTIVE_HEALTHY" for status in services.values()):
                    project_response = await client.get(f"/projects/{project_ref}")
                    await raise_for_status(project_response)
                    return project_response.json()

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        msg = (
            f"Project {project_ref} did not become healthy within {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS} seconds"
        )
        raise TimeoutError(msg)

    async def _fetch_api_keys(self, client: httpx.AsyncClient, project_ref: str) -> tuple[str, str]:
        """Fetch the project's API keys.

        Args:
            client: Authenticated Management API client.
            project_ref: Project reference ID.

        Returns:
            Tuple of (anon_key, service_role_key).
        """
        response = await client.get(f"/projects/{project_ref}/api-keys")
        await raise_for_status(response)
        keys = response.json()

        anon_key = ""
        service_role_key = ""
        for key in keys:
            if key.get("name") == "anon":
                anon_key = key["api_key"]
            elif key.get("name") == "service_role":
                service_role_key = key["api_key"]

        return anon_key, service_role_key

    def _build_outputs(self, project_data: dict, anon_key: str, service_role_key: str) -> ProjectOutputs:
        """Build outputs from project API response data.

        Args:
            project_data: Raw project data from the Management API.
            anon_key: Public anonymous API key.
            service_role_key: Service role API key.

        Returns:
            ProjectOutputs with project metadata and API keys.
        """
        project_ref = project_data["id"]
        return ProjectOutputs(
            project_ref=project_ref,
            name=project_data["name"],
            organization_id=project_data["organization_id"],
            region=project_data["region"],
            status=project_data.get("status", "UNKNOWN"),
            endpoint=f"https://{project_ref}.supabase.co",
            anon_key=anon_key,
            service_role_key=service_role_key,
        )

    async def on_create(self) -> ProjectOutputs:
        """Create a Supabase project and wait until healthy.

        Returns:
            ProjectOutputs with project details and API keys.
        """
        client = create_management_client(self.config.access_token)

        try:
            response = await client.post(
                "/projects",
                json={
                    "organization_id": self.config.organization_id,
                    "name": self.config.name,
                    "region": self.config.region,
                    "db_pass": self.config.database_password,
                },
            )
            await raise_for_status(response)
            project_data = response.json()
            project_ref = project_data["id"]

            project_data = await self._wait_for_healthy(client, project_ref)
            anon_key, service_role_key = await self._fetch_api_keys(client, project_ref)

            return self._build_outputs(project_data, anon_key, service_role_key)
        finally:
            await client.aclose()

    async def on_update(self, previous_config: ProjectConfig) -> ProjectOutputs:
        """Update the project name if changed.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            ProjectOutputs with current project state.

        Raises:
            RuntimeError: If no existing outputs are available for the project.
        """
        if self.outputs is not None and previous_config.name == self.config.name:
            return self.outputs

        if self.outputs is None:
            msg = "Cannot update project without existing outputs"
            raise RuntimeError(msg)

        client = create_management_client(self.config.access_token)

        try:
            project_ref = self.outputs.project_ref

            if previous_config.name != self.config.name:
                response = await client.patch(
                    f"/projects/{project_ref}",
                    json={"name": self.config.name},
                )
                await raise_for_status(response)

            project_response = await client.get(f"/projects/{project_ref}")
            await raise_for_status(project_response)
            project_data = project_response.json()

            anon_key, service_role_key = await self._fetch_api_keys(client, project_ref)

            return self._build_outputs(project_data, anon_key, service_role_key)
        finally:
            await client.aclose()

    async def on_delete(self) -> None:
        """Delete the Supabase project.

        Idempotent: Succeeds if the project does not exist.
        """
        if self.outputs is None:
            return

        client = create_management_client(self.config.access_token)

        try:
            response = await client.delete(f"/projects/{self.outputs.project_ref}")

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
