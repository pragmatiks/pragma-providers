"""Resource definitions for gcp provider.

Import and export your Resource classes here for discovery by the runtime.
"""

from gcp_provider.resources.secret import (
    Secret,
    SecretConfig,
    SecretOutputs,
)

__all__ = [
    "Secret",
    "SecretConfig",
    "SecretOutputs",
]
