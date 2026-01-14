"""GCP provider for Pragmatiks.

Provides GCP Secret Manager resources for managing secrets
in Google Cloud Platform using user-provided credentials.
"""

from pragma_sdk import Provider

from gcp_provider.resources import Secret, SecretConfig, SecretOutputs

gcp = Provider(name="gcp")

# Register resources
gcp.resource("secret")(Secret)

__all__ = [
    "gcp",
    "Secret",
    "SecretConfig",
    "SecretOutputs",
]
