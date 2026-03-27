"""GitHub provider for Pragmatiks.

Provides GitHub resources for managing repositories, deployment environments,
and secrets via the GitHub REST API.
"""

from pragma_sdk import Provider

from github_provider.resources import (
    Environment,
    EnvironmentConfig,
    EnvironmentOutputs,
    ProtectionRulesConfig,
    Repository,
    RepositoryConfig,
    RepositoryOutputs,
    ReviewerConfig,
    Secret,
    SecretConfig,
    SecretOutputs,
)


github = Provider()

github.resource("repository")(Repository)
github.resource("environment")(Environment)
github.resource("secret")(Secret)

__all__ = [
    "github",
    "Environment",
    "EnvironmentConfig",
    "EnvironmentOutputs",
    "ProtectionRulesConfig",
    "Repository",
    "RepositoryConfig",
    "RepositoryOutputs",
    "ReviewerConfig",
    "Secret",
    "SecretConfig",
    "SecretOutputs",
]
