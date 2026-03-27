"""Resource definitions for GitHub provider.

Import and export your Resource classes here for discovery by the runtime.
"""

from github_provider.resources.environment import (
    Environment,
    EnvironmentConfig,
    EnvironmentOutputs,
    ProtectionRulesConfig,
    ReviewerConfig,
)
from github_provider.resources.repository import (
    Repository,
    RepositoryConfig,
    RepositoryOutputs,
)
from github_provider.resources.secret import (
    Secret,
    SecretConfig,
    SecretOutputs,
)


__all__ = [
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
