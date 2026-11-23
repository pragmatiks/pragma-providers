"""Vercel provider for Pragmatiks.

Provides Vercel resources for managing projects, with the deployments
Pragmatiks makes of their Git link, and custom domains via the Vercel
REST API.
"""

from pragma_sdk import Provider

from vercel_provider.resources import (
    DeploymentConfig,
    Domain,
    DomainConfig,
    DomainOutputs,
    EnvironmentVariableConfig,
    GitRepositoryConfig,
    Project,
    ProjectConfig,
    ProjectOutputs,
)


vercel = Provider()

vercel.resource("project")(Project)
vercel.resource("domain")(Domain)

__all__ = [
    "vercel",
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
