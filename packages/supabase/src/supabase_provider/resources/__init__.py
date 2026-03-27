"""Resource definitions for supabase provider.

Import and export your Resource classes here for discovery by the runtime.
"""

from supabase_provider.resources.auth import (
    Auth,
    AuthConfig,
    AuthOutputs,
    ExternalProviderConfig,
)
from supabase_provider.resources.project import (
    Project,
    ProjectConfig,
    ProjectOutputs,
)


__all__ = [
    "Auth",
    "AuthConfig",
    "AuthOutputs",
    "ExternalProviderConfig",
    "Project",
    "ProjectConfig",
    "ProjectOutputs",
]
