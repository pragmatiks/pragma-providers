"""Supabase Project resource."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from pragma_sdk import Config, Field, HealthStatus, ImmutableField, Outputs, Resource, SensitiveField
from pydantic import BaseModel
from pydantic import Field as PydanticField

from supabase_provider.client import create_management_client, raise_for_status


POLL_INTERVAL_SECONDS = 10
HEALTHY_TIMEOUT_SECONDS = 900
READINESS_SERVICES = "auth,db,rest"
EXTERNAL_PROVIDERS = ("google", "github", "apple")
TRANSITIONAL_STATUSES = frozenset({"COMING_UP", "RESTORING", "UPGRADING", "RESTARTING", "RESIZING"})
DELETION_STATUSES = frozenset({"GOING_DOWN", "REMOVED"})


class ExternalProviderConfig(BaseModel):
    """Configuration for an external OAuth provider.

    Attributes:
        enabled: Whether this provider is enabled.
        client_id: OAuth client ID for the provider.
        secret: OAuth client secret for the provider.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = True
    client_id: str = ""
    secret: str = ""


class ProjectConfig(Config):
    """Configuration for a Supabase project and its authentication settings.

    Attributes:
        access_token: Supabase personal access token for the Management API.
            Use a pragma/secret resource with a FieldReference to provide this.
        organization_slug: Slug of the Supabase organization that owns the project.
        name: Project name, unique within the organization.
        region: Deployment region (e.g., ``us-east-1``, ``eu-west-1``).
            Use ``GET /v1/projects/available-regions`` to list valid regions.
        database_password: Password for the project's PostgreSQL database,
            applied at creation and whenever it changes.
        site_url: The base URL of the site where users are redirected after
            authentication. Left unchanged when unset.
        additional_redirect_urls: Allowed redirect URLs beyond the site URL.
        disable_signup: Whether to disable new user signups.
        jwt_expiry: JWT token expiry time in seconds. Defaults to 3600.
        external_email_enabled: Whether email/password authentication is enabled.
        external_phone_enabled: Whether phone/OTP authentication is enabled.
        mailer_autoconfirm: Whether to auto-confirm email addresses on signup.
        external_google: Google OAuth provider configuration; disabled when unset.
        external_github: GitHub OAuth provider configuration; disabled when unset.
        external_apple: Apple OAuth provider configuration; disabled when unset.
    """

    access_token: SensitiveField[str]
    organization_slug: ImmutableField[str]
    name: ImmutableField[str]
    region: ImmutableField[str]
    database_password: SensitiveField[str]
    site_url: Field[str] | None = None
    additional_redirect_urls: Field[list[str]] = PydanticField(default_factory=list)
    disable_signup: Field[bool] = False
    jwt_expiry: Field[int] = 3600
    external_email_enabled: Field[bool] = True
    external_phone_enabled: Field[bool] = False
    mailer_autoconfirm: Field[bool] = False
    external_google: Field[ExternalProviderConfig] | None = None
    external_github: Field[ExternalProviderConfig] | None = None
    external_apple: Field[ExternalProviderConfig] | None = None


class ProjectOutputs(Outputs):
    """Outputs of a Supabase project and its authentication settings.

    Attributes:
        project_ref: Unique project reference ID used in all API calls.
        name: Project name.
        organization_slug: Slug of the owning organization.
        region: Deployment region.
        endpoint: Project API endpoint URL.
        anon_key: Public anonymous API key for client-side access.
        service_role_key: Service role API key for server-side access.
        site_url: Configured site URL.
        disable_signup: Whether signups are disabled.
        jwt_expiry: JWT token expiry in seconds.
        external_email_enabled: Whether email auth is enabled.
        external_phone_enabled: Whether phone auth is enabled.
        mailer_autoconfirm: Whether email auto-confirm is on.
        external_providers_enabled: Enabled external OAuth providers.

    While the project is not ``ACTIVE_HEALTHY`` its API keys and auth config
    are not read: the keys are empty, the auth settings are ``None`` and no
    external provider is listed.
    """

    project_ref: str
    name: str
    organization_slug: str
    region: str
    endpoint: str
    anon_key: str
    service_role_key: str
    site_url: str | None
    disable_signup: bool | None
    jwt_expiry: int | None
    external_email_enabled: bool | None
    external_phone_enabled: bool | None
    mailer_autoconfirm: bool | None
    external_providers_enabled: list[str]


def build_auth_settings(config: ProjectConfig) -> dict[str, Any]:
    """Build the auth config request body from ``config``.

    Args:
        config: Project configuration holding the authentication settings.

    Returns:
        Request body for ``PATCH /projects/{ref}/config/auth``.
    """
    settings: dict[str, Any] = {
        "disable_signup": config.disable_signup,
        "jwt_exp": config.jwt_expiry,
        "external_email_enabled": config.external_email_enabled,
        "external_phone_enabled": config.external_phone_enabled,
        "mailer_autoconfirm": config.mailer_autoconfirm,
        "uri_allow_list": ",".join(config.additional_redirect_urls),
    }

    if config.site_url is not None:
        settings["site_url"] = config.site_url

    for provider_name in EXTERNAL_PROVIDERS:
        provider: ExternalProviderConfig | None = getattr(config, f"external_{provider_name}")
        settings[f"external_{provider_name}_enabled"] = provider is not None and provider.enabled

        if provider is not None:
            settings[f"external_{provider_name}_client_id"] = provider.client_id
            settings[f"external_{provider_name}_secret"] = provider.secret

    return settings


def build_outputs(
    project: dict[str, Any],
    api_keys: list[dict[str, Any]],
    auth_settings: dict[str, Any],
) -> ProjectOutputs:
    """Build outputs from Management API responses.

    Args:
        project: Project from ``GET /projects`` or ``GET /projects/{ref}``.
        api_keys: Keys from ``GET /projects/{ref}/api-keys``; empty when not read.
        auth_settings: Auth config from ``GET /projects/{ref}/config/auth``; empty when not read.

    Returns:
        ProjectOutputs with project metadata, API keys and auth settings.
    """
    project_ref = project["ref"]
    keys_by_name = {key["name"]: key.get("api_key") or "" for key in api_keys}

    return ProjectOutputs(
        project_ref=project_ref,
        name=project["name"],
        organization_slug=project["organization_slug"],
        region=project["region"],
        endpoint=f"https://{project_ref}.supabase.co",
        anon_key=keys_by_name.get("anon", ""),
        service_role_key=keys_by_name.get("service_role", ""),
        site_url=auth_settings.get("site_url"),
        disable_signup=auth_settings.get("disable_signup"),
        jwt_expiry=auth_settings.get("jwt_exp"),
        external_email_enabled=auth_settings.get("external_email_enabled"),
        external_phone_enabled=auth_settings.get("external_phone_enabled"),
        mailer_autoconfirm=auth_settings.get("mailer_autoconfirm"),
        external_providers_enabled=[
            provider_name
            for provider_name in EXTERNAL_PROVIDERS
            if auth_settings.get(f"external_{provider_name}_enabled")
        ],
    )


class Project(Resource[ProjectConfig, ProjectOutputs]):
    """Supabase project resource, including its authentication settings.

    Creates and manages a Supabase project via the Management API. Each
    project includes a PostgreSQL database, authentication, and API
    endpoints. The project is located by ``organization_slug`` and ``name``.

    Lifecycle:
        - on_create: Creates the project, waits until healthy, then applies the auth settings.
        - on_observe: Finds the project; does not wait for it to become healthy, and reads
          its API keys and auth config only while it is ``ACTIVE_HEALTHY``.
        - on_update: Waits until the existing project is healthy, then applies a changed
          database password and the auth settings; fails if the project does not exist
          or is neither active nor transitioning.
        - on_delete: Deletes the project; succeeds if it does not exist.

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
              organization_slug: my-org-slug
              name: my-app
              region: eu-west-1
              database_password:
                provider: pragma
                resource: secret
                name: supabase-db-password
                field: outputs.value
              site_url: "https://myapp.example.com"
              disable_signup: false
              external_github:
                enabled: true
                client_id: "gh-client-id"
                secret: "gh-client-secret"
    """

    async def find_project(self, client: httpx.AsyncClient) -> dict[str, Any] | None:
        """Find the project with this resource's organization and name.

        Args:
            client: Authenticated Management API client.

        Returns:
            The project, or ``None`` if no live project has that name.

        Raises:
            RuntimeError: If the organization holds more than one project with that name.
        """
        response = await client.get("/projects")
        await raise_for_status(response)
        matches = [
            project
            for project in response.json()
            if project["organization_slug"] == self.config.organization_slug
            and project["name"] == self.config.name
            and project["status"] not in DELETION_STATUSES
        ]

        if len(matches) > 1:
            msg = f"Organization {self.config.organization_slug} has {len(matches)} projects named {self.config.name}"
            raise RuntimeError(msg)

        return matches[0] if matches else None

    async def fetch_project(self, client: httpx.AsyncClient, project_ref: str) -> dict[str, Any]:
        """Fetch the project by reference.

        Args:
            client: Authenticated Management API client.
            project_ref: Project reference ID.

        Returns:
            The project as returned by ``GET /projects/{ref}``.

        Raises:
            httpx.HTTPStatusError: If the Management API rejects the request.
        """
        response = await client.get(f"/projects/{project_ref}")
        await raise_for_status(response)

        return response.json()

    async def wait_for_healthy(self, client: httpx.AsyncClient, project_ref: str) -> dict[str, Any]:
        """Poll the project's auth, database and REST services until all are healthy.

        Args:
            client: Authenticated Management API client.
            project_ref: Project reference ID.

        Returns:
            The project once its services are healthy.

        Raises:
            TimeoutError: If the project is not healthy within ``HEALTHY_TIMEOUT_SECONDS``; the message
                carries the last HTTP status and the last per-service statuses.
            httpx.HTTPStatusError: If the Management API answers 401, 403 or 404 while polling, or rejects
                the fetch of the healthy project.
        """
        last_http_status: int | None = None
        last_services: dict[str, str] = {}
        deadline = time.monotonic() + HEALTHY_TIMEOUT_SECONDS

        while True:
            response = await client.get(f"/projects/{project_ref}/health", params={"services": READINESS_SERVICES})

            if response.status_code in {401, 403, 404}:
                await raise_for_status(response)

            last_http_status = response.status_code
            last_services = {}

            if response.is_success:
                last_services = {service["name"]: service["status"] for service in response.json()}

            if last_services and all(status == "ACTIVE_HEALTHY" for status in last_services.values()):
                return await self.fetch_project(client, project_ref)

            if time.monotonic() >= deadline:
                break

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        services = ",".join(f"{name}:{status}" for name, status in last_services.items()) or "none"
        msg = (
            f"Project {project_ref} did not become healthy within {HEALTHY_TIMEOUT_SECONDS} seconds "
            f"(last_http_status={last_http_status} services={services})"
        )
        raise TimeoutError(msg)

    async def apply_database_password(self, client: httpx.AsyncClient, project_ref: str) -> None:
        """Set the project's database password to this resource's password.

        Args:
            client: Authenticated Management API client.
            project_ref: Project reference ID.
        """
        response = await client.patch(
            f"/projects/{project_ref}/database/password",
            json={"password": self.config.database_password},
        )
        await raise_for_status(response)

    async def apply_auth_settings(self, client: httpx.AsyncClient, project_ref: str) -> None:
        """Converge the project's auth config to this resource's settings.

        Args:
            client: Authenticated Management API client.
            project_ref: Project reference ID.
        """
        response = await client.patch(f"/projects/{project_ref}/config/auth", json=build_auth_settings(self.config))
        await raise_for_status(response)

    async def fetch_outputs(self, client: httpx.AsyncClient, project: dict[str, Any]) -> ProjectOutputs:
        """Read the project's API keys and auth config into outputs.

        Args:
            client: Authenticated Management API client.
            project: The project as returned by the Management API.

        Returns:
            ProjectOutputs for the project.
        """
        project_ref = project["ref"]
        keys_response = await client.get(f"/projects/{project_ref}/api-keys")
        await raise_for_status(keys_response)

        auth_response = await client.get(f"/projects/{project_ref}/config/auth")
        await raise_for_status(auth_response)

        return build_outputs(project, keys_response.json(), auth_response.json())

    async def on_create(self) -> ProjectOutputs:
        """Create the project, wait until healthy, and apply the auth settings.

        Returns:
            ProjectOutputs with project details, API keys and auth settings.
        """
        async with create_management_client(self.config.access_token) as client:
            response = await client.post(
                "/projects",
                json={
                    "organization_slug": self.config.organization_slug,
                    "name": self.config.name,
                    "region": self.config.region,
                    "db_pass": self.config.database_password,
                },
            )
            await raise_for_status(response)
            project = await self.wait_for_healthy(client, response.json()["ref"])
            await self.apply_auth_settings(client, project["ref"])

            return await self.fetch_outputs(client, project)

    async def on_observe(self) -> ProjectOutputs | None:
        """Find the project and read its outputs, whatever its health.

        Returns:
            ProjectOutputs for the project, or ``None`` when it does not exist. The
            API keys and auth settings are left unread unless the project is ``ACTIVE_HEALTHY``.
        """
        async with create_management_client(self.config.access_token) as client:
            project = await self.find_project(client)

            if project is None:
                return None

            if project["status"] != "ACTIVE_HEALTHY":
                return build_outputs(project, [], {})

            return await self.fetch_outputs(client, project)

    async def on_update(self, previous_config: ProjectConfig | None) -> ProjectOutputs:
        """Wait until the existing project is healthy, then apply the database password and auth settings.

        The database password is sent when it differs from ``previous_config`` or when
        ``previous_config`` is unknown.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            ProjectOutputs with project details, API keys and auth settings.

        Raises:
            RuntimeError: If the project does not exist, or is neither ``ACTIVE_HEALTHY``
                nor transitioning (for example paused).
        """
        async with create_management_client(self.config.access_token) as client:
            project = await self.find_project(client)

            if project is None:
                msg = f"Project {self.config.name} not found in organization {self.config.organization_slug}"
                raise RuntimeError(msg)

            status = project["status"]

            if status != "ACTIVE_HEALTHY" and status not in TRANSITIONAL_STATUSES:
                msg = f"Project {self.config.name} is {status}; it must be active to apply configuration"
                raise RuntimeError(msg)

            project = await self.wait_for_healthy(client, project["ref"])

            if previous_config is None or previous_config.database_password != self.config.database_password:
                await self.apply_database_password(client, project["ref"])

            await self.apply_auth_settings(client, project["ref"])

            return await self.fetch_outputs(client, project)

    async def on_delete(self) -> None:
        """Delete the project; succeeds if it does not exist."""
        async with create_management_client(self.config.access_token) as client:
            project = await self.find_project(client)

            if project is None:
                return

            response = await client.delete(f"/projects/{project['ref']}")

            if response.status_code == 404:
                return

            await raise_for_status(response)

    async def health(self) -> HealthStatus:
        """Report the project's status as health.

        Returns:
            Healthy when ``ACTIVE_HEALTHY``, degraded while transitioning, unhealthy otherwise.
        """
        async with create_management_client(self.config.access_token) as client:
            project = await self.find_project(client)

        if project is None:
            return HealthStatus(status="unhealthy", message="Project not found")

        status = project["status"]

        if status == "ACTIVE_HEALTHY":
            return HealthStatus(status="healthy", message="Project is active")

        if status in TRANSITIONAL_STATUSES:
            return HealthStatus(status="degraded", message=f"Project is {status}")

        return HealthStatus(status="unhealthy", message=f"Project is {status}")
