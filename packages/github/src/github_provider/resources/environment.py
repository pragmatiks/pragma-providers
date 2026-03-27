"""GitHub Environment resource."""

from __future__ import annotations

from typing import Any

from pragma_sdk import Config, Field, ImmutableField, Outputs, Resource, SensitiveField
from pydantic import BaseModel
from pydantic import Field as PydanticField

from github_provider.client import create_github_client, raise_for_status


class ReviewerConfig(BaseModel):
    """Configuration for an environment protection rule reviewer.

    Attributes:
        reviewer_type: Type of reviewer (``User`` or ``Team``).
        reviewer_id: GitHub ID of the user or team.
    """

    model_config = {"extra": "forbid"}

    reviewer_type: str = "User"
    reviewer_id: int = 0


class ProtectionRulesConfig(BaseModel):
    """Configuration for environment protection rules.

    Attributes:
        wait_timer: Number of minutes to wait before allowing deployments
            (0 to 43200). Set to 0 to disable the wait timer.
        reviewers: List of required reviewers for deployments.
    """

    model_config = {"extra": "forbid"}

    wait_timer: int = 0
    reviewers: list[ReviewerConfig] = PydanticField(default_factory=list)


class EnvironmentConfig(Config):
    """Configuration for a GitHub deployment environment.

    Attributes:
        access_token: GitHub personal access token for authentication.
        owner: GitHub user or organization that owns the repository.
        repository: Repository name where the environment is created.
        environment_name: Name of the deployment environment
            (e.g., ``production``, ``staging``).
        protection_rules: Optional protection rules for the environment.
    """

    access_token: SensitiveField[str]
    owner: ImmutableField[str]
    repository: ImmutableField[str]
    environment_name: ImmutableField[str]
    protection_rules: Field[ProtectionRulesConfig] | None = None


class EnvironmentOutputs(Outputs):
    """Outputs from GitHub environment creation.

    Attributes:
        environment_id: Numeric environment ID assigned by GitHub.
        environment_name: Name of the deployment environment.
        html_url: URL to the environment settings page on GitHub.
        wait_timer: Configured wait timer in minutes (0 if not set).
        reviewers_count: Number of required reviewers.
    """

    environment_id: int
    environment_name: str
    html_url: str
    wait_timer: int
    reviewers_count: int


def _build_create_body(config: EnvironmentConfig) -> dict[str, Any]:
    """Build the API request body for environment creation or update.

    Args:
        config: Environment configuration.

    Returns:
        Dictionary suitable for PUT /repos/{owner}/{repo}/environments/{env}.
    """
    body: dict[str, Any] = {}

    if config.protection_rules is not None:
        body["wait_timer"] = config.protection_rules.wait_timer

        if config.protection_rules.reviewers:
            body["reviewers"] = [
                {"type": reviewer.reviewer_type, "id": reviewer.reviewer_id}
                for reviewer in config.protection_rules.reviewers
            ]

    return body


def _build_outputs(data: dict[str, Any], owner: str, repository: str) -> EnvironmentOutputs:
    """Build outputs from GitHub environment API response data.

    Args:
        data: Raw environment data from the GitHub API.
        owner: Repository owner.
        repository: Repository name.

    Returns:
        EnvironmentOutputs with environment metadata.
    """
    protection_rules = data.get("protection_rules", [])
    wait_timer = 0
    reviewers_count = 0

    for rule in protection_rules:
        if rule.get("type") == "wait_timer":
            wait_timer = rule.get("wait_timer", 0)
        elif rule.get("type") == "required_reviewers":
            reviewers = rule.get("reviewers", [])
            reviewers_count = len(reviewers)

    return EnvironmentOutputs(
        environment_id=data["id"],
        environment_name=data["name"],
        html_url=f"https://github.com/{owner}/{repository}/settings/environments/{data['id']}",
        wait_timer=wait_timer,
        reviewers_count=reviewers_count,
    )


class Environment(Resource[EnvironmentConfig, EnvironmentOutputs]):
    """GitHub deployment environment resource.

    Creates and manages deployment environments on GitHub repositories
    via the REST API. Environments can have protection rules such as
    wait timers and required reviewers.

    Lifecycle:
        - on_create: Creates or updates an environment using PUT (the
          GitHub API uses PUT for both create and update). Idempotent.
        - on_update: Updates the environment configuration by re-applying
          the PUT request with the new settings.
        - on_delete: Deletes the environment. Idempotent -- succeeds if
          the environment does not exist.

    Example::

        resources:
          - name: production-env
            provider: github
            type: environment
            config:
              access_token:
                provider: pragma
                resource: secret
                name: github-token
                field: outputs.value
              owner: my-org
              repository: my-repo
              environment_name: production
              protection_rules:
                wait_timer: 30
                reviewers:
                  - reviewer_type: User
                    reviewer_id: 12345
    """

    async def _apply_environment(self) -> EnvironmentOutputs:
        """Create or update the environment and return outputs.

        Returns:
            EnvironmentOutputs with environment details.
        """
        client = create_github_client(self.config.access_token)

        try:
            body = _build_create_body(self.config)
            response = await client.put(
                f"/repos/{self.config.owner}/{self.config.repository}/environments/{self.config.environment_name}",
                json=body,
            )
            await raise_for_status(response)

            return _build_outputs(response.json(), self.config.owner, self.config.repository)
        finally:
            await client.aclose()

    async def on_create(self) -> EnvironmentOutputs:
        """Create a deployment environment on the repository.

        Returns:
            EnvironmentOutputs with environment details.
        """
        return await self._apply_environment()

    async def on_update(self, previous_config: EnvironmentConfig) -> EnvironmentOutputs:
        """Update the environment configuration.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            EnvironmentOutputs with updated environment details.
        """
        return await self._apply_environment()

    async def on_delete(self) -> None:
        """Delete the deployment environment.

        Idempotent: Succeeds if the environment does not exist.
        """
        client = create_github_client(self.config.access_token)

        try:
            response = await client.delete(
                f"/repos/{self.config.owner}/{self.config.repository}/environments/{self.config.environment_name}",
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
