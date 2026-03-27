"""Vercel provider for Pragmatiks.

Provides Vercel resources for managing projects, deployments, and custom
domains via the Vercel REST API.
"""

from pragma_sdk import Provider

from vercel_provider.resources import (
    Deployment,
    DeploymentConfig,
    DeploymentOutputs,
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
vercel.resource("deployment")(Deployment)
vercel.resource("domain")(Domain)

__all__ = [
    "vercel",
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
