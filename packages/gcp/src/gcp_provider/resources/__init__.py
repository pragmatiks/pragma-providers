"""Resource definitions for gcp provider.

Import and export your Resource classes here for discovery by the runtime.
"""

from gcp_provider.resources.cloudsql import (
    CloudSQL,
    CloudSQLConfig,
    CloudSQLOutputs,
)
from gcp_provider.resources.gke import (
    GKE,
    GKEConfig,
    GKEOutputs,
)
from gcp_provider.resources.secret import (
    Secret,
    SecretConfig,
    SecretOutputs,
)

__all__ = [
    "CloudSQL",
    "CloudSQLConfig",
    "CloudSQLOutputs",
    "GKE",
    "GKEConfig",
    "GKEOutputs",
    "Secret",
    "SecretConfig",
    "SecretOutputs",
]
