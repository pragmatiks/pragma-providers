"""Supabase provider for Pragmatiks.

Provides Supabase resources for managing projects and authentication
configuration via the Supabase Management API.
"""

from pragma_sdk import Provider

from supabase_provider.resources import (
    Auth,
    AuthConfig,
    AuthOutputs,
    ExternalProviderConfig,
    Project,
    ProjectConfig,
    ProjectOutputs,
)


supabase = Provider()

supabase.resource("project")(Project)
supabase.resource("auth")(Auth)

__all__ = [
    "supabase",
    "Auth",
    "AuthConfig",
    "AuthOutputs",
    "ExternalProviderConfig",
    "Project",
    "ProjectConfig",
    "ProjectOutputs",
]
