"""Supabase provider for Pragmatiks.

Provides the Supabase project resource, including its authentication
settings, via the Supabase Management API.
"""

from pragma_sdk import Provider

from supabase_provider.resources import (
    ExternalProviderConfig,
    Project,
    ProjectConfig,
    ProjectOutputs,
)


supabase = Provider()

supabase.resource("project")(Project)

__all__ = [
    "supabase",
    "ExternalProviderConfig",
    "Project",
    "ProjectConfig",
    "ProjectOutputs",
]
