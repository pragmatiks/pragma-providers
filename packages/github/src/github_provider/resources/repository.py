"""GitHub Repository resource."""

from __future__ import annotations

from typing import Any

from pragma_sdk import Config, Field, ImmutableField, Outputs, Resource, SensitiveField

from github_provider.client import create_github_client, raise_for_status


class RepositoryConfig(Config):
    """Configuration for a GitHub repository.

    Attributes:
        access_token: GitHub personal access token for authentication.
            Use a pragma/secret resource with a FieldReference to provide this.
        owner: GitHub user or organization that owns the repository.
        name: Repository name. Must be unique within the owner's account.
        description: Short description of the repository.
        visibility: Repository visibility (``public`` or ``private``).
        default_branch: Default branch name. Only applied on creation
            when ``auto_init`` is True.
        has_issues: Whether to enable the Issues feature.
        has_wiki: Whether to enable the Wiki feature.
        has_projects: Whether to enable the Projects feature.
        auto_init: Whether to initialize the repository with a README.
            Only used on creation.
        delete_branch_on_merge: Whether to automatically delete head
            branches after pull requests are merged.
        allow_squash_merge: Whether to allow squash-merging pull requests.
        allow_merge_commit: Whether to allow merging pull requests with a merge commit.
        allow_rebase_merge: Whether to allow rebase-merging pull requests.
    """

    access_token: SensitiveField[str]
    owner: ImmutableField[str]
    name: ImmutableField[str]
    description: Field[str] = ""
    visibility: Field[str] = "private"
    default_branch: Field[str] = "main"
    has_issues: Field[bool] = True
    has_wiki: Field[bool] = False
    has_projects: Field[bool] = False
    auto_init: Field[bool] = True
    delete_branch_on_merge: Field[bool] = True
    allow_squash_merge: Field[bool] = True
    allow_merge_commit: Field[bool] = True
    allow_rebase_merge: Field[bool] = True


class RepositoryOutputs(Outputs):
    """Outputs from GitHub repository creation.

    Attributes:
        repository_id: Numeric repository ID assigned by GitHub.
        full_name: Full repository name in ``owner/repo`` format.
        html_url: URL to the repository on GitHub.
        clone_url: HTTPS clone URL.
        ssh_url: SSH clone URL.
        default_branch: Default branch name.
        visibility: Repository visibility.
    """

    repository_id: int
    full_name: str
    html_url: str
    clone_url: str
    ssh_url: str
    default_branch: str
    visibility: str


def _build_create_body(config: RepositoryConfig) -> dict[str, Any]:
    """Build the API request body for repository creation.

    Args:
        config: Repository configuration.

    Returns:
        Dictionary suitable for POST /orgs/{org}/repos or POST /user/repos.
    """
    return {
        "name": config.name,
        "description": config.description,
        "private": config.visibility == "private",
        "auto_init": config.auto_init,
        "has_issues": config.has_issues,
        "has_wiki": config.has_wiki,
        "has_projects": config.has_projects,
        "delete_branch_on_merge": config.delete_branch_on_merge,
        "allow_squash_merge": config.allow_squash_merge,
        "allow_merge_commit": config.allow_merge_commit,
        "allow_rebase_merge": config.allow_rebase_merge,
    }


def _build_update_body(config: RepositoryConfig) -> dict[str, Any]:
    """Build the API request body for repository update.

    Only includes mutable fields. Owner and name are immutable.

    Args:
        config: Current repository configuration.

    Returns:
        Dictionary suitable for PATCH /repos/{owner}/{repo}.
    """
    return {
        "description": config.description,
        "private": config.visibility == "private",
        "has_issues": config.has_issues,
        "has_wiki": config.has_wiki,
        "has_projects": config.has_projects,
        "delete_branch_on_merge": config.delete_branch_on_merge,
        "allow_squash_merge": config.allow_squash_merge,
        "allow_merge_commit": config.allow_merge_commit,
        "allow_rebase_merge": config.allow_rebase_merge,
    }


def _build_outputs(data: dict[str, Any]) -> RepositoryOutputs:
    """Build outputs from GitHub repository API response data.

    Args:
        data: Raw repository data from the GitHub API.

    Returns:
        RepositoryOutputs with repository metadata.
    """
    return RepositoryOutputs(
        repository_id=data["id"],
        full_name=data["full_name"],
        html_url=data["html_url"],
        clone_url=data["clone_url"],
        ssh_url=data["ssh_url"],
        default_branch=data.get("default_branch", "main"),
        visibility=data.get("visibility", "private"),
    )


class Repository(Resource[RepositoryConfig, RepositoryOutputs]):
    """GitHub repository resource.

    Creates and manages GitHub repositories via the REST API. Repositories
    are created under the specified owner (user or organization) and can
    be configured with visibility, features, and merge settings.

    Lifecycle:
        - on_create: Creates a new repository. Uses the organization endpoint
          when the owner is an organization, or the user endpoint otherwise.
          Not idempotent -- duplicate calls create duplicate repositories.
        - on_update: Updates mutable repository settings (description,
          visibility, features, merge options). Owner and name are immutable.
        - on_delete: Deletes the repository. Idempotent -- succeeds if the
          repository does not exist.

    Example::

        resources:
          - name: my-repo
            provider: github
            type: repository
            config:
              access_token:
                provider: pragma
                resource: secret
                name: github-token
                field: outputs.value
              owner: my-org
              name: my-repo
              description: My application repository
              visibility: private
              auto_init: true
              has_issues: true
              has_wiki: false
    """

    async def on_create(self) -> RepositoryOutputs:
        """Create a GitHub repository.

        Returns:
            RepositoryOutputs with repository details.
        """
        client = create_github_client(self.config.access_token)

        try:
            body = _build_create_body(self.config)

            response = await client.post(f"/orgs/{self.config.owner}/repos", json=body)

            if response.status_code == 404:
                response = await client.post("/user/repos", json=body)

            await raise_for_status(response)

            return _build_outputs(response.json())
        finally:
            await client.aclose()

    async def on_update(self, previous_config: RepositoryConfig) -> RepositoryOutputs:
        """Update the repository settings if changed.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            RepositoryOutputs with current repository state.

        Raises:
            RuntimeError: If no existing outputs are available for the repository.
        """
        if self.outputs is None:
            msg = "Cannot update repository without existing outputs"
            raise RuntimeError(msg)

        client = create_github_client(self.config.access_token)

        try:
            update_body = _build_update_body(self.config)
            response = await client.patch(
                f"/repos/{self.config.owner}/{self.config.name}",
                json=update_body,
            )
            await raise_for_status(response)

            return _build_outputs(response.json())
        finally:
            await client.aclose()

    async def on_delete(self) -> None:
        """Delete the GitHub repository.

        Idempotent: Succeeds if the repository does not exist.
        """
        if self.outputs is None:
            return

        client = create_github_client(self.config.access_token)

        try:
            response = await client.delete(f"/repos/{self.config.owner}/{self.config.name}")

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
