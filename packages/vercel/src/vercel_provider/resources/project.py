"""Vercel Project resource."""

from __future__ import annotations

from typing import Any

import httpx
from pragma_sdk import Config, Field, ImmutableField, Outputs, Resource, SensitiveField
from pydantic import BaseModel
from pydantic import Field as PydanticField

from vercel_provider.client import create_vercel_client, raise_for_status


class EnvironmentVariableConfig(BaseModel):
    """Configuration for a Vercel environment variable.

    Attributes:
        key: Environment variable name.
        value: Environment variable value.
        target: Deployment targets (e.g., ``production``, ``preview``, ``development``).
        variable_type: Type of variable (``plain``, ``encrypted``, ``secret``, ``system``).
    """

    model_config = {"extra": "forbid"}

    key: str
    value: str
    target: list[str] = PydanticField(default_factory=lambda: ["production", "preview", "development"])
    variable_type: str = "encrypted"


class GitRepositoryConfig(BaseModel):
    """Configuration for connecting a Git repository to the project.

    Attributes:
        repo: Repository identifier in ``owner/repo`` format.
        repo_type: Git provider (``github``, ``gitlab``, ``bitbucket``).
    """

    model_config = {"extra": "forbid"}

    repo: str
    repo_type: str = "github"


class ProjectConfig(Config):
    """Configuration for a Vercel project.

    Attributes:
        access_token: Vercel API token for authentication.
            Use a pragma/secret resource with a FieldReference to provide this.
        name: Project name. Must be unique within the team/account.
        framework: Framework preset (e.g., ``nextjs``, ``vite``, ``remix``).
            When None, Vercel auto-detects the framework.
        git_repository: Git repository to connect to the project.
        build_command: Custom build command. When None, auto-detected.
        output_directory: Custom output directory. When None, auto-detected.
        install_command: Custom install command. When None, auto-detected.
        root_directory: Root directory of the project within the repository.
        environment_variables: Environment variables to set on the project.
        team_id: Vercel team ID. Required for team-owned projects.
    """

    access_token: SensitiveField[str]
    name: ImmutableField[str]
    framework: Field[str] | None = None
    git_repository: Field[GitRepositoryConfig] | None = None
    build_command: Field[str] | None = None
    output_directory: Field[str] | None = None
    install_command: Field[str] | None = None
    root_directory: Field[str] | None = None
    environment_variables: Field[list[EnvironmentVariableConfig]] = PydanticField(default_factory=list)
    team_id: Field[str] | None = None


class ProjectOutputs(Outputs):
    """Outputs from Vercel project creation.

    Attributes:
        project_id: Unique project identifier assigned by Vercel.
        name: Project name.
        account_id: Account or team ID that owns the project.
        framework: Detected or configured framework.
        url: Default project URL on Vercel.
    """

    project_id: str
    name: str
    account_id: str
    framework: str
    url: str


def _build_create_body(config: ProjectConfig) -> dict[str, Any]:
    """Build the API request body for project creation.

    Args:
        config: Project configuration.

    Returns:
        Dictionary suitable for POST /v10/projects request body.
    """
    body: dict[str, Any] = {"name": config.name}

    if config.framework is not None:
        body["framework"] = config.framework

    if config.git_repository is not None:
        body["gitRepository"] = {
            "repo": config.git_repository.repo,
            "type": config.git_repository.repo_type,
        }

    if config.build_command is not None:
        body["buildCommand"] = config.build_command

    if config.output_directory is not None:
        body["outputDirectory"] = config.output_directory

    if config.install_command is not None:
        body["installCommand"] = config.install_command

    if config.root_directory is not None:
        body["rootDirectory"] = config.root_directory

    return body


def _build_update_body(config: ProjectConfig) -> dict[str, Any]:
    """Build the API request body for project update.

    Only includes mutable fields. Name is immutable and excluded.

    Args:
        config: Current project configuration.

    Returns:
        Dictionary suitable for PATCH /v9/projects/{idOrName} request body.
    """
    body: dict[str, Any] = {}

    if config.framework is not None:
        body["framework"] = config.framework

    if config.build_command is not None:
        body["buildCommand"] = config.build_command

    if config.output_directory is not None:
        body["outputDirectory"] = config.output_directory

    if config.install_command is not None:
        body["installCommand"] = config.install_command

    if config.root_directory is not None:
        body["rootDirectory"] = config.root_directory

    return body


def _build_outputs(data: dict[str, Any]) -> ProjectOutputs:
    """Build outputs from Vercel project API response data.

    Args:
        data: Raw project data from the Vercel API.

    Returns:
        ProjectOutputs with project metadata.
    """
    project_id = data["id"]
    name = data["name"]

    return ProjectOutputs(
        project_id=project_id,
        name=name,
        account_id=data.get("accountId", ""),
        framework=data.get("framework") or "",
        url=f"https://{name}.vercel.app",
    )


async def _sync_environment_variables(
    client: httpx.AsyncClient,
    project_id: str,
    variables: list[EnvironmentVariableConfig],
    team_id: str | None,
) -> None:
    """Synchronize environment variables on the project.

    Removes all existing environment variables and creates the declared ones.
    This ensures the project environment matches the declared configuration.

    Args:
        client: Authenticated Vercel API client.
        project_id: Vercel project ID.
        variables: Desired environment variable configurations.
        team_id: Optional team ID for team-scoped requests.
    """
    params: dict[str, str] = {}

    if team_id:
        params["teamId"] = team_id

    existing_response = await client.get(f"/v9/projects/{project_id}/env", params=params)
    await raise_for_status(existing_response)
    existing_data = existing_response.json()
    existing_envs: list[dict[str, Any]] = existing_data.get("envs", [])

    for env in existing_envs:
        env_id = env["id"]
        delete_response = await client.delete(f"/v9/projects/{project_id}/env/{env_id}", params=params)

        if delete_response.status_code == 404:
            continue

        await raise_for_status(delete_response)

    if not variables:
        return

    env_body = [
        {
            "key": var.key,
            "value": var.value,
            "target": var.target,
            "type": var.variable_type,
        }
        for var in variables
    ]

    create_response = await client.post(f"/v10/projects/{project_id}/env", params=params, json=env_body)
    await raise_for_status(create_response)


class Project(Resource[ProjectConfig, ProjectOutputs]):
    """Vercel project resource.

    Creates and manages Vercel projects via the REST API. Each project
    can be connected to a Git repository for automatic deployments
    and configured with framework presets, build settings, and
    environment variables.

    Lifecycle:
        - on_create: Creates a new project with the provided configuration,
          then syncs environment variables. Not idempotent -- duplicate calls
          create duplicate projects.
        - on_update: Updates mutable project settings (framework, build commands,
          environment variables). Name is immutable.
        - on_delete: Deletes the project. Idempotent -- succeeds if the
          project does not exist.

    Example::

        resources:
          - name: my-app
            provider: vercel
            type: project
            config:
              access_token:
                provider: pragma
                resource: secret
                name: vercel-token
                field: outputs.value
              name: my-app
              framework: nextjs
              git_repository:
                repo: my-org/my-app
                repo_type: github
              environment_variables:
                - key: DATABASE_URL
                  value:
                    provider: supabase
                    resource: project
                    name: my-db
                    field: outputs.endpoint
                  target:
                    - production
                    - preview
    """

    def _query_params(self) -> dict[str, str]:
        """Build common query parameters for API requests.

        Returns:
            Dictionary of query parameters including team_id if configured.
        """
        params: dict[str, str] = {}

        if self.config.team_id:
            params["teamId"] = self.config.team_id

        return params

    async def on_create(self) -> ProjectOutputs:
        """Create a Vercel project and sync environment variables.

        Returns:
            ProjectOutputs with project details.
        """
        client = create_vercel_client(self.config.access_token)

        try:
            body = _build_create_body(self.config)
            response = await client.post("/v10/projects", params=self._query_params(), json=body)
            await raise_for_status(response)
            project_data = response.json()

            if self.config.environment_variables:
                await _sync_environment_variables(
                    client,
                    project_data["id"],
                    self.config.environment_variables,
                    self.config.team_id,
                )

            return _build_outputs(project_data)
        finally:
            await client.aclose()

    async def on_update(self, previous_config: ProjectConfig) -> ProjectOutputs:
        """Update the project settings if changed.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            ProjectOutputs with current project state.

        Raises:
            RuntimeError: If no existing outputs are available for the project.
        """
        if self.outputs is None:
            msg = "Cannot update project without existing outputs"
            raise RuntimeError(msg)

        client = create_vercel_client(self.config.access_token)

        try:
            project_id = self.outputs.project_id
            params = self._query_params()

            update_body = _build_update_body(self.config)

            if update_body:
                response = await client.patch(f"/v9/projects/{project_id}", params=params, json=update_body)
                await raise_for_status(response)

            await _sync_environment_variables(
                client,
                project_id,
                self.config.environment_variables,
                self.config.team_id,
            )

            get_response = await client.get(f"/v9/projects/{project_id}", params=params)
            await raise_for_status(get_response)

            return _build_outputs(get_response.json())
        finally:
            await client.aclose()

    async def on_delete(self) -> None:
        """Delete the Vercel project.

        Idempotent: Succeeds if the project does not exist.
        """
        if self.outputs is None:
            return

        client = create_vercel_client(self.config.access_token)

        try:
            response = await client.delete(
                f"/v9/projects/{self.outputs.project_id}",
                params=self._query_params(),
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
