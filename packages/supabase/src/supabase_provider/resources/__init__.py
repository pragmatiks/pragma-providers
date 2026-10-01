"""Resource definitions for supabase provider.

Import and export your Resource classes here for discovery by the runtime.
"""

from supabase_provider.resources.project import (
    ExternalProviderConfig,
    Project,
    ProjectConfig,
    ProjectOutputs,
)


__all__ = [
    "ExternalProviderConfig",
    "Project",
    "ProjectConfig",
    "ProjectOutputs",
]
