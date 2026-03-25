"""GCP provider for Pragmatiks.

Provides GCP resources for managing infrastructure in Google Cloud Platform
using user-provided credentials (multi-tenant SaaS pattern).
"""

from pragma_sdk import Provider

from gcp_provider.resources import (
    GKE,
    Database,
    DatabaseConfig,
    DatabaseInstance,
    DatabaseInstanceConfig,
    DatabaseInstanceOutputs,
    DatabaseOutputs,
    GKEConfig,
    GKEOutputs,
    Secret,
    SecretConfig,
    SecretOutputs,
    User,
    UserConfig,
    UserOutputs,
)


gcp = Provider()

gcp.resource("cloudsql/database_instance")(DatabaseInstance)
gcp.resource("cloudsql/database")(Database)
gcp.resource("cloudsql/user")(User)
gcp.resource("gke")(GKE)
gcp.resource("secret")(Secret)

__all__ = [
    "gcp",
    "Database",
    "DatabaseConfig",
    "DatabaseInstance",
    "DatabaseInstanceConfig",
    "DatabaseInstanceOutputs",
    "DatabaseOutputs",
    "GKE",
    "GKEConfig",
    "GKEOutputs",
    "Secret",
    "SecretConfig",
    "SecretOutputs",
    "User",
    "UserConfig",
    "UserOutputs",
]
