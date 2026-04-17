"""Pragma platform configuration resource.

Non-sensitive configuration store that enables provider-level defaults
referenced via FieldReferences (e.g., pragma/config/gcp#outputs.project_id).
"""

from __future__ import annotations

from typing import Any

from pragma_sdk import Config, Field, Outputs, Resource


class ConfigResourceConfig(Config):
    """Configuration for a platform-managed config resource.

    Unlike pragma/secret which only accepts string values, pragma/config
    supports arbitrary JSON types for non-sensitive configuration.

    Attributes:
        data: Key-value pairs of configuration data. Supports any JSON-serializable type.
    """

    data: Field[dict[str, Any]]


class ConfigResourceOutputs(Outputs):
    """Outputs from platform config creation.

    Each key-value pair from config.data is exposed as an output field.
    Supports any JSON-serializable type.
    """

    model_config = {"extra": "allow"}


class ConfigResource(Resource[ConfigResourceConfig, ConfigResourceOutputs]):
    """Platform-managed configuration for non-sensitive data.

    Use pragma/config resources to store provider-level defaults that
    other resources reference via FieldReferences.

    Example usage:
        pragma/config/gcp stores { project_id: "my-project", region: "us-central1" }
        GCP resources reference: pragma/config/gcp#outputs.project_id
    """

    async def on_create(self) -> ConfigResourceOutputs:
        """Store configuration data.

        Returns:
            Outputs containing the configuration key-value pairs.

        Raises:
            TypeError: If config.data is not a dict.
        """
        data = self.config.data

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for config data, got {type(data).__name__}")

        return ConfigResourceOutputs(**data)

    async def on_update(self, previous_config: ConfigResourceConfig) -> ConfigResourceOutputs:
        """Update configuration data.

        Args:
            previous_config: Previous configuration.

        Returns:
            Outputs containing the updated configuration key-value pairs.

        Raises:
            TypeError: If config.data is not a dict.
        """
        data = self.config.data

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for config data, got {type(data).__name__}")

        return ConfigResourceOutputs(**data)

    async def on_delete(self) -> None:
        """Config deletion is a no-op (data stored in resource config)."""
