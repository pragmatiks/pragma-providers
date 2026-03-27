"""Resource definitions for Vercel provider.

Import and export your Resource classes here for discovery by the runtime.
"""

from vercel_provider.resources.deployment import (
    Deployment,
    DeploymentConfig,
    DeploymentOutputs,
)
from vercel_provider.resources.domain import (
    Domain,
    DomainConfig,
    DomainOutputs,
)
from vercel_provider.resources.project import (
    EnvironmentVariableConfig,
    GitRepositoryConfig,
    Project,
    ProjectConfig,
    ProjectOutputs,
)


__all__ = [
    "Deployment",
    "DeploymentConfig",
    "DeploymentOutputs",
    "Domain",
    "DomainConfig",
    "DomainOutputs",
    "EnvironmentVariableConfig",
    "GitRepositoryConfig",
    "Project",
    "ProjectConfig",
    "ProjectOutputs",
]
