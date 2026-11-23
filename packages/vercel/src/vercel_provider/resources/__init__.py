"""Resource definitions for the Vercel provider: projects, with the deployments Pragmatiks makes, and domains."""

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
from vercel_provider.resources.project_deployment import DeploymentConfig


__all__ = [
    "DeploymentConfig",
    "Domain",
    "DomainConfig",
    "DomainOutputs",
    "EnvironmentVariableConfig",
    "GitRepositoryConfig",
    "Project",
    "ProjectConfig",
    "ProjectOutputs",
]
