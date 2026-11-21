"""Pragma platform secret resource.

Pass-through resource that exposes config.data as outputs, so other resources
can reference secret values via FieldReferences. Every value is a ``SecretStr``.
"""

from __future__ import annotations

from pragma_sdk import Config, Field, Outputs, Resource
from pydantic import Field as PydanticField
from pydantic import SecretStr


class SecretConfig(Config):
    """Configuration for a platform-managed secret.

    Attributes:
        data: Key-value pairs of secret data. Each value is a secret string.
    """

    data: Field[dict[str, SecretStr]]


class SecretOutputs(Outputs):
    """Outputs from platform secret creation.

    Each key-value pair from config.data is exposed as an output field,
    enabling other resources to reference secret values via FieldReferences.
    Every entry is a ``SecretStr``.
    """

    model_config = {"extra": "allow"}

    __pydantic_extra__: dict[str, SecretStr] = PydanticField(init=False)


class Secret(Resource[SecretConfig, SecretOutputs]):
    """Platform-managed secret for storing sensitive key-value data."""

    computed = True

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

    async def on_update(self, previous_config: SecretConfig | None) -> SecretOutputs:
        """Update secret data.

        Args:
            previous_config: Previous secret configuration, if any.

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
