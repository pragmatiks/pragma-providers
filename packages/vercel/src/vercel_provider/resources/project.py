"""Vercel Project resource."""

from __future__ import annotations

from typing import Any

import httpx
from pragma_sdk import RECOVERY_GUIDANCE, Config, Field, ImmutableField, Outputs, Resource
from pydantic import BaseModel, SecretStr, model_validator
from pydantic import Field as PydanticField

from vercel_provider.client import (
    create_vercel_client,
    fetch_optional_json,
    raise_for_create_status,
    raise_for_status,
)
from vercel_provider.resources.budget import compute_handler_deadline, compute_request_timeout
from vercel_provider.resources.deployment_policy import build_deployment_policy, is_deployment_policy_current
from vercel_provider.resources.project_deployment import (
    DeploymentConfig,
    DeploymentRequest,
    build_deployment_body,
    build_deployment_digest,
    build_git_source,
    converge_deployment,
    extract_deployment_id,
    fetch_stamped_deployments,
    resolve_ref,
)


PROJECT_LINK_PATH = "/v9/projects/{project}/link"
"""Path of Vercel's project link endpoint, which takes ``{"type", "repo"}`` and answers the linked project, its
``link`` included. Linking starts no build. Absent from https://vercel.com/openapi.json; read from the Vercel CLI
(``packages/cli/src/util/git/connect-git-provider.ts``) and probed 2026-11-23."""
DEPLOYMENT_NEEDS_REPOSITORY = "deployment needs git_repository: Pragmatiks deploys from the project's Git link."
PLAIN_VARIABLE_TYPE = "plain"


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
        repo: Repository in ``owner/repo`` format, or a reference to a repository output such as a
            ``github/repository`` resource's ``outputs.full_name``, which makes the project wait for it.
        repo_type: Git provider (``github``, ``gitlab``, ``bitbucket``). Pragmatiks deploys only ``github``.
    """

    model_config = {"extra": "forbid"}

    repo: Field[str]
    repo_type: str = "github"


class ProjectConfig(Config):
    """Configuration for a Vercel project.

    Attributes:
        access_token: Vercel API token for authentication, a reference to a
            secret output such as a pragma/secret entry.
        name: Project name. Must be unique within the team/account.
        framework: Framework preset (e.g., ``nextjs``, ``vite``, ``remix``).
            When None, Vercel auto-detects the framework.
        git_repository: Git repository to connect to the project. Immutable, because Vercel
            cannot change a project's Git link in place.
        git_push_deploys: Whether Vercel builds every push to the linked repository. ``False`` writes a
            project deployment policy allowing every deployment source but Git, which overrides the
            team's deployment-source rules for this project.
        deployment: The Git ref Pragmatiks deploys from the project's Git link. When None, Pragmatiks
            deploys nothing. Needs a GitHub ``git_repository``.
        build_command: Custom build command. When None, auto-detected.
        output_directory: Custom output directory. When None, auto-detected.
        install_command: Custom install command. When None, auto-detected.
        root_directory: Root directory of the project within the repository.
        environment_variables: Environment variables to set on the project.
        team_id: Vercel team ID. Required for team-owned projects. Immutable.
    """

    access_token: Field[SecretStr]
    name: ImmutableField[str]
    framework: Field[str] | None = None
    git_repository: ImmutableField[GitRepositoryConfig] | None = None
    git_push_deploys: Field[bool] = True
    deployment: Field[DeploymentConfig] | None = None
    build_command: Field[str] | None = None
    output_directory: Field[str] | None = None
    install_command: Field[str] | None = None
    root_directory: Field[str] | None = None
    environment_variables: Field[list[EnvironmentVariableConfig]] = PydanticField(default_factory=list)
    team_id: ImmutableField[str] | None = None

    @model_validator(mode="after")
    def validate_deployment_source(self) -> ProjectConfig:
        """Refuse a ``deployment`` without a GitHub ``git_repository`` to deploy from.

        A ``git_repository`` given as one reference is checked when the deployment is built instead.

        Returns:
            Self after validation.

        Raises:
            ValueError: If ``deployment`` is set without ``git_repository``, or with a non-GitHub one.
        """
        if self.deployment is None:
            return self

        if self.git_repository is None:
            raise ValueError(DEPLOYMENT_NEEDS_REPOSITORY)

        if isinstance(self.git_repository, GitRepositoryConfig) and self.git_repository.repo_type != "github":
            message = (
                f"deployment needs a github git_repository; {self.git_repository.repo_type} projects deploy on push."
            )
            raise ValueError(message)

        return self


class ProjectOutputs(Outputs):
    """Outputs of a Vercel project and the deployment Pragmatiks made of it.

    Attributes:
        project_id: Unique project identifier assigned by Vercel.
        name: Project name.
        account_id: Account or team ID that owns the project.
        framework: Detected or configured framework.
        url: Default project URL on Vercel.
        deployment_id: The newest deployment Pragmatiks made that did not end ``ERROR``, ``CANCELED`` or
            ``BLOCKED`` (refused by a deployment policy or check), or None before the first.
            A Git push can replace it on Vercel without changing it here.
        deployment_url: That deployment's URL, or None.
        deployment_target: Where Vercel put that deployment, ``production`` or ``preview``, or None.
    """

    project_id: str
    name: str
    account_id: str
    framework: str
    url: str
    deployment_id: str | None = None
    deployment_url: str | None = None
    deployment_target: str | None = None


def build_create_body(config: ProjectConfig, repository: str | None) -> dict[str, Any]:
    """Build the API request body for project creation.

    The create takes no ``deploymentPolicy``, so a project whose Git pushes must not deploy is created without
    its repository and linked once the policy is in place.

    Args:
        config: Project configuration.
        repository: The resolved ``owner/repo`` to link in the create, or None to create it unlinked.

    Returns:
        Dictionary suitable for POST /v11/projects request body.
    """
    body: dict[str, Any] = {"name": config.name}

    if config.framework is not None:
        body["framework"] = config.framework

    if repository is not None and isinstance(config.git_repository, GitRepositoryConfig):
        body["gitRepository"] = {"repo": repository, "type": config.git_repository.repo_type}

    if config.build_command is not None:
        body["buildCommand"] = config.build_command

    if config.output_directory is not None:
        body["outputDirectory"] = config.output_directory

    if config.install_command is not None:
        body["installCommand"] = config.install_command

    if config.root_directory is not None:
        body["rootDirectory"] = config.root_directory

    return body


def build_update_body(
    build_settings: dict[str, str | None], project: dict[str, Any], git_push_deploys: bool
) -> dict[str, Any]:
    """Build the API request body for project update.

    Unset build settings are sent as ``null`` to restore auto-detection; ``framework`` is omitted when
    unset because Vercel reads ``null`` as none. The deployment policy is sent only when the project's
    differs from the one ``git_push_deploys`` asks for.

    Args:
        build_settings: Framework, build, output and install commands, and root directory, by Vercel name.
        project: Current project data from the Vercel API.
        git_push_deploys: Whether Vercel builds the pushes to the linked repository.

    Returns:
        Dictionary suitable for PATCH /v9/projects/{idOrName} request body.
    """
    body: dict[str, Any] = {key: value for key, value in build_settings.items() if key != "framework" or value}

    if not is_deployment_policy_current(project, git_push_deploys):
        body["deploymentPolicy"] = build_deployment_policy(git_push_deploys)

    return body


def build_outputs(project: dict[str, Any], deployment: dict[str, Any] | None) -> ProjectOutputs:
    """Build outputs from Vercel project data and the project's newest served stamped deployment.

    Args:
        project: Raw project data from the Vercel API.
        deployment: The newest stamped deployment that did not fail, or None.

    Returns:
        ProjectOutputs with project and deployment metadata.
    """
    name = project["name"]
    outputs = ProjectOutputs(
        project_id=project["id"],
        name=name,
        account_id=project.get("accountId", ""),
        framework=project.get("framework") or "",
        url=f"https://{name}.vercel.app",
    )

    if deployment is None:
        return outputs

    return outputs.model_copy(
        update={
            "deployment_id": extract_deployment_id(deployment),
            "deployment_url": f"https://{deployment['url']}",
            "deployment_target": deployment.get("target") or "preview",
        }
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


def build_listed_identity(environment_variable: dict[str, Any]) -> tuple[str, tuple[str, ...], str | None]:
    """Build the identity of an environment variable Vercel lists.

    Args:
        environment_variable: One entry of Vercel's environment variable listing.

    Returns:
        Its ``(key, sorted targets, Git branch)`` identity, comparable with ``build_variable_identities``.
    """
    target = environment_variable.get("target") or []
    targets = [target] if isinstance(target, str) else target

    return environment_variable["key"], tuple(sorted(targets)), environment_variable.get("gitBranch")


def build_variable_versions(variables: list[EnvironmentVariableConfig], target: str) -> list[tuple[str, str, str]]:
    """Build the version of each declared environment variable a deployment target sees.

    A variable's version is its declared value, of every type: Vercel bumps a variable's ``updatedAt`` on every
    upsert, an unchanged one included, so its listing cannot tell a changed value from a reapplied one.

    Args:
        variables: Declared environment variable configurations.
        target: The deployment target; only variables whose targets include it count.

    Returns:
        ``(key, variable type, value)`` per counted variable, for ``build_deployment_digest`` only.
    """
    return [
        (variable.key, variable.variable_type, variable.value) for variable in variables if target in variable.target
    ]


async def fetch_environment_variables(
    client: httpx.AsyncClient,
    project: str,
    params: dict[str, str],
    deadline: float,
) -> list[dict[str, Any]]:
    """Fetch the environment variables of a project, without their values decrypted.

    Args:
        client: Authenticated Vercel API client.
        project: Vercel project ID or name.
        params: Query parameters, such as the team ID.
        deadline: Handler deadline on the ``time.monotonic`` clock.

    Returns:
        Vercel's environment variable listing.

    Raises:
        httpx.HTTPStatusError: If Vercel rejects the listing.
    """
    timeout = compute_request_timeout(deadline, f"the environment variables of Vercel project {project}")
    response = await client.get(f"/v10/projects/{project}/env", params=params, timeout=timeout)
    await raise_for_status(response)

    return response.json().get("envs", [])


async def delete_dropped_environment_variables(
    client: httpx.AsyncClient,
    project: str,
    previous_variables: list[EnvironmentVariableConfig],
    variables: list[EnvironmentVariableConfig],
    params: dict[str, str],
    deadline: float,
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
        deadline: Handler deadline on the ``time.monotonic`` clock.

    Raises:
        httpx.HTTPStatusError: If Vercel rejects the listing, or rejects a deletion with a status other than 404.
    """
    dropped = build_variable_identities(previous_variables) - build_variable_identities(variables)

    if not dropped:
        return

    for environment_variable in await fetch_environment_variables(client, project, params, deadline):
        if build_listed_identity(environment_variable) not in dropped:
            continue

        timeout = compute_request_timeout(deadline, f"the environment variables of Vercel project {project}")
        delete_response = await client.delete(
            f"/v9/projects/{project}/env/{environment_variable['id']}", params=params, timeout=timeout
        )

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
    deadline: float,
) -> None:
    """Create or overwrite the declared environment variables on the project.

    Args:
        client: Authenticated Vercel API client.
        project: Vercel project ID or name.
        variables: Desired environment variable configurations.
        params: Query parameters, such as the team ID.
        deadline: Handler deadline on the ``time.monotonic`` clock.

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
    timeout = compute_request_timeout(deadline, f"the environment variables of Vercel project {project}")
    response = await client.post(
        f"/v10/projects/{project}/env", params=upsert_params, json=upsert_body, timeout=timeout
    )
    await raise_for_status(response)
    failed = response.json().get("failed") or []

    if failed:
        messages = "; ".join(format_rejected_variable(entry.get("error") or {}) for entry in failed)
        message = f"Vercel rejected environment variables: {messages}"
        raise RuntimeError(message)


class Project(Resource[ProjectConfig, ProjectOutputs]):
    """Vercel project resource, with the deployment Pragmatiks makes of its Git link.

    Creates and manages a Vercel project via the REST API: framework preset, build
    settings, environment variables and an optional Git repository link. With a
    ``deployment`` block, it deploys a Git ref of the linked repository and reports
    that deployment; a branch builds its head, a tag or commit SHA pins a release.
    It redeploys when the deployment block, the environment variables the deployment's
    target sees (a ``plain`` variable's value, Vercel's version of any other), or the
    build settings change, and when the last deployment it made failed. Vercel keeps building every push to the linked
    repository unless ``git_push_deploys`` is ``false``.

    The project is located by ``name`` and ``team_id``; its deployments by the
    ``meta.pragma_resource`` stamp, set to the resource ID. Deployments without the
    stamp, such as Git push builds, are never read.

    Lifecycle:
        - on_create: Creates the project, applies its settings and deployment
          policy, links the repository when the create did not, syncs environment
          variables, then deploys as on_update does.
        - on_observe: Reads the project by name and its newest stamped deployment.
        - on_update: Applies the mutable project settings and deployment policy,
          links the repository when the project has no Git link, syncs environment
          variables, and, with a ``deployment`` block, adopts a
          current stamped deployment or triggers one and waits until it is READY.
        - on_delete: Deletes the project, and Vercel deletes its deployments.
          Succeeds if the project does not exist.

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
                repo:
                  provider: github
                  resource: repository
                  name: my-app
                  field: outputs.full_name
              git_push_deploys: false
              deployment:
                ref: v1.4.0
                target: production
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

    ``repo`` also takes a plain ``owner/repo`` string, such as ``my-org/my-app``.
    """

    def reveal_access_token(self) -> str:
        """Reveal the resolved access token for an API client.

        Returns:
            The plaintext access token.
        """
        return self.resolved(self.config.access_token).get_secret_value()

    def project_path(self) -> str:
        """Build the API path of the project named by ``config.name``.

        Returns:
            The ``/v9/projects/{name}`` path.
        """
        return f"/v9/projects/{self.config.name}"

    def build_query_params(self) -> dict[str, str]:
        """Build common query parameters for API requests.

        Returns:
            Dictionary of query parameters including team_id if configured.
        """
        params: dict[str, str] = {}

        if self.config.team_id:
            params["teamId"] = self.resolved(self.config.team_id)

        return params

    def build_build_settings(self) -> dict[str, str | None]:
        """Build the project's build settings by their Vercel names.

        Returns:
            Framework, build, output and install commands, and root directory; None where unset.
        """
        return {
            "framework": self.resolved(self.config.framework),
            "buildCommand": self.resolved(self.config.build_command),
            "outputDirectory": self.resolved(self.config.output_directory),
            "installCommand": self.resolved(self.config.install_command),
            "rootDirectory": self.resolved(self.config.root_directory),
        }

    def resolve_repository(self) -> str | None:
        """Resolve the configured ``owner/repo``.

        Returns:
            The repository, or None when the project has no ``git_repository``.
        """
        repository = self.resolved(self.config.git_repository)

        if repository is None:
            return None

        return self.resolved(repository.repo)

    async def fetch_project(self, client: httpx.AsyncClient, deadline: float) -> dict[str, Any] | None:
        """Fetch the project named by ``config.name``.

        Args:
            client: Authenticated Vercel API client.
            deadline: Handler deadline on the ``time.monotonic`` clock.

        Returns:
            Project data from the Vercel API, or None when it does not exist.
        """
        timeout = compute_request_timeout(deadline, f"Vercel project {self.config.name}")

        return await fetch_optional_json(client, self.project_path(), self.build_query_params(), timeout)

    async def patch_settings(self, client: httpx.AsyncClient, project: dict[str, Any], deadline: float) -> None:
        """Apply the mutable project settings and the deployment policy.

        Args:
            client: Authenticated Vercel API client.
            project: Current project data from the Vercel API.
            deadline: Handler deadline on the ``time.monotonic`` clock.
        """
        body = build_update_body(self.build_build_settings(), project, self.resolved(self.config.git_push_deploys))
        timeout = compute_request_timeout(deadline, f"Vercel project {self.config.name}")
        response = await client.patch(self.project_path(), params=self.build_query_params(), json=body, timeout=timeout)
        await raise_for_status(response)

    async def link_repository(
        self, client: httpx.AsyncClient, project: dict[str, Any], deadline: float
    ) -> dict[str, Any] | None:
        """Link the configured repository to a project that has no Git link.

        Runs after ``patch_settings``, so a project whose Git pushes must not deploy has its deployment policy
        before the link can build. Safe to rerun: a linked project is left alone.

        Args:
            client: Authenticated Vercel API client.
            project: Current project data from the Vercel API.
            deadline: Handler deadline on the ``time.monotonic`` clock.

        Returns:
            The project's Git link: the one it has, the one the link response carries, or None when no
            repository is configured or the response carries none.

        Raises:
            httpx.HTTPStatusError: If Vercel refuses the link, such as for a missing GitHub Login Connection.
        """
        repository = self.resolve_repository()
        git_repository = self.resolved(self.config.git_repository)

        if project.get("link") or repository is None or git_repository is None:
            return project.get("link")

        timeout = compute_request_timeout(deadline, f"the Git link of Vercel project {self.config.name}")
        response = await client.post(
            PROJECT_LINK_PATH.format(project=project["id"]),
            params=self.build_query_params(),
            json={"type": git_repository.repo_type, "repo": repository},
            timeout=timeout,
        )
        await raise_for_status(response)

        return (response.json() or {}).get("link")

    def build_deployment_request(
        self,
        deployment: DeploymentConfig,
        project: dict[str, Any],
        is_prefix_retry: bool,
        deadline: float,
    ) -> DeploymentRequest:
        """Build the request that keeps the project's stamped deployment current.

        Args:
            deployment: The resolved ``deployment`` block.
            project: Current project data, its ``link`` the freshest this handler saw.
            is_prefix_retry: True when the handler may be finishing a create a stopped host left part done.
            deadline: Handler deadline on the ``time.monotonic`` clock.

        Returns:
            The request for the configured ``deployment`` block.

        Raises:
            ValueError: If the project has no ``git_repository``, or its Git link cannot deploy the repository.
        """
        repository = self.resolve_repository()

        if repository is None:
            raise ValueError(DEPLOYMENT_NEEDS_REPOSITORY)

        target = self.resolved(deployment.target)
        link = project.get("link")
        ref = resolve_ref(self.resolved(deployment.ref), link)
        variables = build_variable_versions(self.resolved(self.config.environment_variables), target)
        digest = build_deployment_digest(self.id, ref, target, self.build_build_settings(), variables)
        body = build_deployment_body(
            project["id"], project["name"], build_git_source(link, repository, ref), target, self.id, digest
        )

        return DeploymentRequest(
            project_id=project["id"],
            resource_id=self.id,
            body=body,
            digest=digest,
            is_prefix_retry=is_prefix_retry,
            params=self.build_query_params(),
            deadline=deadline,
        )

    async def read_outputs(
        self,
        client: httpx.AsyncClient,
        fallback_project: dict[str, Any] | None,
        deadline: float,
    ) -> ProjectOutputs | None:
        """Read the project and its newest served stamped deployment.

        Args:
            client: Authenticated Vercel API client.
            fallback_project: Project data to use when the read does not find the project yet, or None.
            deadline: Handler deadline on the ``time.monotonic`` clock.

        Returns:
            The outputs, or None when the project does not exist.
        """
        project = await self.fetch_project(client, deadline) or fallback_project

        if project is None:
            return None

        lookup = await fetch_stamped_deployments(client, project["id"], self.id, self.build_query_params(), deadline)

        return build_outputs(project, lookup.newest_served)

    async def finish_deployment(
        self,
        client: httpx.AsyncClient,
        project: dict[str, Any],
        fallback_project: dict[str, Any] | None,
        is_prefix_retry: bool,
        deadline: float,
    ) -> ProjectOutputs:
        """Converge the stamped deployment, then read the resulting outputs.

        Args:
            client: Authenticated Vercel API client.
            project: Current project data, its ``link`` the freshest this handler saw.
            fallback_project: Project data for the final read when it does not find the project yet, or None.
            is_prefix_retry: True when the handler may be finishing a create a stopped host left part done.
            deadline: Handler deadline on the ``time.monotonic`` clock.

        Returns:
            ProjectOutputs with the project and its stamped deployment.

        Raises:
            RuntimeError: If the build failed, was cancelled or was blocked, or the project is gone by the final
                read.
            TimeoutError: If the deployment is not READY, or not listed, by the deadline.
            ValueError: If the project has no ``git_repository``, or its Git link cannot deploy the repository.
            httpx.HTTPStatusError: If Vercel refuses a read or the trigger.
        """
        deployment = self.resolved(self.config.deployment)

        if deployment is not None:
            request = self.build_deployment_request(deployment, project, is_prefix_retry, deadline)
            await converge_deployment(client, request)

        outputs = await self.read_outputs(client, fallback_project, deadline)

        if outputs is None:
            message = (
                f"Vercel project {self.config.name} was deleted outside Pragmatiks during the operation. "
                f"{RECOVERY_GUIDANCE}"
            )
            raise RuntimeError(message)

        return outputs

    async def on_create(self) -> ProjectOutputs:
        """Create a Vercel project, apply its settings, link its repository, set its variables, and deploy.

        With ``git_push_deploys`` true the create carries the repository; otherwise the project is created
        unlinked, its push-blocking deployment policy applied, and only then linked, so no push builds.

        Returns:
            ProjectOutputs with project and deployment details.

        Raises:
            AlreadyExistsError: If Vercel already has a project with this name.
        """
        deadline = compute_handler_deadline()
        params = self.build_query_params()
        linked_in_create = self.resolve_repository() if self.resolved(self.config.git_push_deploys) else None

        async with create_vercel_client(self.reveal_access_token()) as client:
            timeout = compute_request_timeout(deadline, f"Vercel project {self.config.name}")
            response = await client.post(
                "/v11/projects", params=params, json=build_create_body(self.config, linked_in_create), timeout=timeout
            )
            await raise_for_create_status(response, f"Vercel project {self.config.name}")
            create_response = response.json()

            await self.patch_settings(client, create_response, deadline)
            link = await self.link_repository(client, create_response, deadline)
            await upsert_environment_variables(
                client, create_response["id"], self.resolved(self.config.environment_variables), params, deadline
            )
            project = await self.fetch_project(client, deadline) or create_response

            return await self.finish_deployment(
                client, {**project, "link": project.get("link") or link}, create_response, False, deadline
            )

    async def on_observe(self) -> ProjectOutputs | None:
        """Read the project named by ``config.name`` and its newest served stamped deployment.

        Returns:
            ProjectOutputs, or ``None`` if the project does not exist.
        """
        async with create_vercel_client(self.reveal_access_token()) as client:
            return await self.read_outputs(client, None, compute_handler_deadline())

    async def on_update(self, previous_config: ProjectConfig | None) -> ProjectOutputs:
        """Apply the mutable project settings, link the repository if unlinked, sync variables, and deploy.

        Deletes only the environment variables ``previous_config`` declared and the
        current configuration no longer declares; every other variable on the project
        is left alone, and nothing is deleted when ``previous_config`` is unknown.
        A call without ``previous_config`` may be finishing a create a stopped host left
        part done, so with nothing listed it first waits for that host's deployment.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            ProjectOutputs with current project and deployment state.

        Raises:
            RuntimeError: If the project does not exist, or its build failed, was cancelled or was blocked.
            ValueError: If the project has no ``git_repository``, or its Git link cannot deploy the repository.
        """
        deadline = compute_handler_deadline()
        params = self.build_query_params()

        async with create_vercel_client(self.reveal_access_token()) as client:
            project = await self.fetch_project(client, deadline)

            if project is None:
                message = f"Vercel project {self.config.name} does not exist. {RECOVERY_GUIDANCE}"
                raise RuntimeError(message)

            await self.patch_settings(client, project, deadline)
            link = await self.link_repository(client, project, deadline)

            name = self.resolved(self.config.name)
            previous_variables = self.resolved(previous_config.environment_variables) if previous_config else []
            variables = self.resolved(self.config.environment_variables)

            await delete_dropped_environment_variables(client, name, previous_variables, variables, params, deadline)
            await upsert_environment_variables(client, name, variables, params, deadline)

            if not project.get("link"):
                project = await self.fetch_project(client, deadline) or project

            return await self.finish_deployment(
                client, {**project, "link": project.get("link") or link}, None, previous_config is None, deadline
            )

    async def on_delete(self) -> None:
        """Delete the Vercel project named by ``config.name``; Vercel deletes its deployments with it.

        Succeeds if the project does not exist.
        """
        async with create_vercel_client(self.reveal_access_token()) as client:
            response = await client.delete(self.project_path(), params=self.build_query_params())

            if response.status_code == 404:
                return

            await raise_for_status(response)
