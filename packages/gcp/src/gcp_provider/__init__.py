"""GCP provider for Pragmatiks.

Provides GCP resources for managing infrastructure in Google Cloud Platform
using user-provided credentials (multi-tenant SaaS pattern).
"""

from pragma_sdk import Provider

from gcp_provider.resources import GKE, GKEConfig, GKEOutputs, Secret, SecretConfig, SecretOutputs

gcp = Provider(name="gcp")

# Register resources
gcp.resource("gke")(GKE)
gcp.resource("secret")(Secret)

__all__ = [
    "gcp",
    "GKE",
    "GKEConfig",
    "GKEOutputs",
    "Secret",
    "SecretConfig",
    "SecretOutputs",
]
