"""Pragma platform secret resource.

Pass-through resource that exposes config.data as outputs, enabling other
resources to reference secret values via FieldReferences.
"""

from __future__ import annotations

from pragma_sdk import Config, Field, Outputs, Resource


class SecretConfig(Config):
    """Configuration for a platform-managed secret.

    Attributes:
        data: Key-value pairs of secret data. Values should be strings.
    """

    data: Field[dict[str, str]]


class SecretOutputs(Outputs):
    """Outputs from platform secret creation.

    Each key-value pair from config.data is exposed as an output field,
    enabling other resources to reference secret values via FieldReferences.
    """

    model_config = {"extra": "allow"}


class Secret(Resource[SecretConfig, SecretOutputs]):
    """Platform-managed secret for storing sensitive key-value data."""

    async def on_create(self) -> SecretOutputs:
        """Store secret data.

        Returns:
            Outputs containing the secret key-value pairs.

        Raises:
            TypeError: If config.data is not a dict.
        """
        data = self.config.data

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for secret data, got {type(data).__name__}")

        return SecretOutputs(**data)

    async def on_update(self, previous_config: SecretConfig) -> SecretOutputs:
        """Update secret data.

        Args:
            previous_config: Previous secret configuration.

        Returns:
            Outputs containing the updated secret key-value pairs.

        Raises:
            TypeError: If config.data is not a dict.
        """
        data = self.config.data

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for secret data, got {type(data).__name__}")

        return SecretOutputs(**data)

    async def on_delete(self) -> None:
        """Secret deletion is a no-op (data stored in resource config)."""
