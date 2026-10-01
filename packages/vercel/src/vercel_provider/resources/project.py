"""Vercel Project resource."""

from __future__ import annotations

from typing import Any, cast

import httpx
from pragma_sdk import Config, Field, ImmutableField, Outputs, Resource, SensitiveField
from pydantic import BaseModel
from pydantic import Field as PydanticField

from vercel_provider.client import create_vercel_client, fetch_optional_json, raise_for_status


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
        git_repository: Git repository to connect to the project. Immutable, because Vercel
            cannot change a project's Git link in place.
        build_command: Custom build command. When None, auto-detected.
        output_directory: Custom output directory. When None, auto-detected.
        install_command: Custom install command. When None, auto-detected.
        root_directory: Root directory of the project within the repository.
        environment_variables: Environment variables to set on the project.
        team_id: Vercel team ID. Required for team-owned projects. Immutable.
    """

    access_token: SensitiveField[str]
    name: ImmutableField[str]
    framework: Field[str] | None = None
    git_repository: ImmutableField[GitRepositoryConfig] | None = None
    build_command: Field[str] | None = None
    output_directory: Field[str] | None = None
    install_command: Field[str] | None = None
    root_directory: Field[str] | None = None
    environment_variables: Field[list[EnvironmentVariableConfig]] = PydanticField(default_factory=list)
    team_id: ImmutableField[str] | None = None


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

    Only includes mutable fields. Unset build settings are sent as ``null`` to restore
    auto-detection; ``framework`` is omitted when unset because Vercel reads ``null`` as none.

    Args:
        config: Current project configuration.

    Returns:
        Dictionary suitable for PATCH /v9/projects/{idOrName} request body.
    """
    body: dict[str, Any] = {
        "buildCommand": config.build_command,
        "outputDirectory": config.output_directory,
        "installCommand": config.install_command,
        "rootDirectory": config.root_directory,
    }

    if config.framework is not None:
        body["framework"] = config.framework

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


def build_variable_identities(variables: list[EnvironmentVariableConfig]) -> set[tuple[str, tuple[str, ...], None]]:
    """Build the identities of declared environment variables.

    Args:
        variables: Declared environment variable configurations.

    Returns:
        One ``(key, sorted targets, Git branch)`` identity per variable; declared
        variables carry no Git branch.
    """
    return {(variable.key, tuple(sorted(variable.target)), None) for variable in variables}


async def delete_dropped_environment_variables(
    client: httpx.AsyncClient,
    project: str,
    previous_variables: list[EnvironmentVariableConfig],
    variables: list[EnvironmentVariableConfig],
    params: dict[str, str],
) -> None:
    """Delete the environment variables that ``previous_variables`` declared and ``variables`` no longer declares.

    An environment variable is identified by its key, its targets and its Git branch.
    Variables that were never declared, such as ones an integration injected or ones
    scoped to a Git branch, are left alone.

    Args:
        client: Authenticated Vercel API client.
        project: Vercel project ID or name.
        previous_variables: Environment variable configurations declared before.
        variables: Desired environment variable configurations.
        params: Query parameters, such as the team ID.

    Raises:
        httpx.HTTPStatusError: If Vercel rejects the listing, or rejects a deletion with a status other than 404.
    """
    dropped = build_variable_identities(previous_variables) - build_variable_identities(variables)

    if not dropped:
        return

    response = await client.get(f"/v10/projects/{project}/env", params=params)
    await raise_for_status(response)

    for environment_variable in response.json().get("envs", []):
        target = environment_variable.get("target") or []
        targets = [target] if isinstance(target, str) else target
        identity = (environment_variable["key"], tuple(sorted(targets)), environment_variable.get("gitBranch"))

        if identity not in dropped:
            continue

        variable_id = environment_variable["id"]
        delete_response = await client.delete(f"/v9/projects/{project}/env/{variable_id}", params=params)

        if delete_response.status_code == 404:
            continue

        await raise_for_status(delete_response)


def format_rejected_variable(error: dict[str, Any]) -> str:
    """Format one entry of Vercel's rejected environment variables.

    Args:
        error: The ``error`` object Vercel reported for the variable.

    Returns:
        The variable key followed by Vercel's message.
    """
    key = error.get("envVarKey") or error.get("key", "unknown key")

    return f"{key}: {error.get('message')}"


async def upsert_environment_variables(
    client: httpx.AsyncClient,
    project: str,
    variables: list[EnvironmentVariableConfig],
    params: dict[str, str],
) -> None:
    """Create or overwrite the declared environment variables on the project.

    Args:
        client: Authenticated Vercel API client.
        project: Vercel project ID or name.
        variables: Desired environment variable configurations.
        params: Query parameters, such as the team ID.

    Raises:
        RuntimeError: If Vercel rejects any declared variable.
    """
    if not variables:
        return

    upsert_body = [
        {
            "key": variable.key,
            "value": variable.value,
            "target": variable.target,
            "type": variable.variable_type,
        }
        for variable in variables
    ]
    upsert_params = {**params, "upsert": "true"}

    response = await client.post(
        f"/v10/projects/{project}/env",
        params=upsert_params,
        json=upsert_body,
    )
    await raise_for_status(response)
    failed = response.json().get("failed") or []

    if failed:
        messages = "; ".join(format_rejected_variable(entry.get("error") or {}) for entry in failed)
        msg = f"Vercel rejected environment variables: {messages}"
        raise RuntimeError(msg)


class Project(Resource[ProjectConfig, ProjectOutputs]):
    """Vercel project resource.

    Creates and manages Vercel projects via the REST API. Each project
    can be connected to a Git repository for automatic deployments
    and configured with framework presets, build settings, and
    environment variables.

    The project is located by ``name`` and ``team_id``.

    Lifecycle:
        - on_create: Creates a new project with the provided configuration,
          then syncs environment variables.
        - on_observe: Reads the project by name.
        - on_update: Applies the mutable project settings (framework, build
          commands, environment variables) and deletes the environment variables
          it no longer declares.
        - on_delete: Deletes the project. Succeeds if the project does not exist.

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

    def project_path(self) -> str:
        """Build the API path of the project named by ``config.name``.

        Returns:
            The ``/v9/projects/{name}`` path.
        """
        return f"/v9/projects/{self.config.name}"

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
        async with create_vercel_client(self.config.access_token) as client:
            body = _build_create_body(self.config)
            response = await client.post("/v10/projects", params=self._query_params(), json=body)
            await raise_for_status(response)
            project_data = response.json()

            await upsert_environment_variables(
                client,
                project_data["id"],
                self.config.environment_variables,
                self._query_params(),
            )

            return _build_outputs(project_data)

    async def on_observe(self) -> ProjectOutputs | None:
        """Read the project named by ``config.name``.

        Returns:
            ProjectOutputs, or ``None`` if the project does not exist.
        """
        async with create_vercel_client(self.config.access_token) as client:
            project = await fetch_optional_json(client, self.project_path(), self._query_params())

        if project is None:
            return None

        return _build_outputs(project)

    async def on_update(self, previous_config: ProjectConfig | None) -> ProjectOutputs:
        """Apply the mutable project settings and environment variables.

        Deletes only the environment variables ``previous_config`` declared and the
        current configuration no longer declares; every other variable on the project
        is left alone, and nothing is deleted when ``previous_config`` is unknown.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            ProjectOutputs with current project state.
        """
        async with create_vercel_client(self.config.access_token) as client:
            params = self._query_params()

            response = await client.patch(self.project_path(), params=params, json=_build_update_body(self.config))
            await raise_for_status(response)

            previous_variables = (
                cast(list[EnvironmentVariableConfig], previous_config.environment_variables) if previous_config else []
            )

            await delete_dropped_environment_variables(
                client,
                self.config.name,
                previous_variables,
                self.config.environment_variables,
                params,
            )
            await upsert_environment_variables(
                client,
                self.config.name,
                self.config.environment_variables,
                params,
            )

            get_response = await client.get(self.project_path(), params=params)
            await raise_for_status(get_response)

            return _build_outputs(get_response.json())

    async def on_delete(self) -> None:
        """Delete the Vercel project named by ``config.name``.

        Succeeds if the project does not exist.
        """
        async with create_vercel_client(self.config.access_token) as client:
            response = await client.delete(self.project_path(), params=self._query_params())

            if response.status_code == 404:
                return

            await raise_for_status(response)
