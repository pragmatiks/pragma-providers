"""Agno Anthropic model resource wrapping the Claude class."""

from __future__ import annotations

from typing import ClassVar

from agno.models.anthropic import Claude
from pragma_sdk import Config, Field, Outputs, Resource


class AnthropicModelConfig(Config):
    """Configuration for Agno Anthropic Claude model.

    Maps to Agno's Claude class from agno.models.anthropic.

    Attributes:
        id: Claude model identifier (e.g., "claude-sonnet-4-20250514").
        api_key: Anthropic API key. Use a FieldReference to inject from pragma/secret.
        max_tokens: Maximum tokens in responses. Defaults to 8192 (Agno default).
        temperature: Sampling temperature (0.0-1.0). Optional.
        top_p: Nucleus sampling parameter. Optional.
        top_k: Top-k sampling parameter. Optional.
        stop_sequences: Stop sequences to end generation. Optional.
    """

    id: str
    api_key: Field[str]
    max_tokens: int = 8192
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None


class AnthropicModelOutputs(Outputs):
    """Serializable outputs from Agno Anthropic model resource.

    Attributes:
        model_id: Configured model identifier.
    """

    model_id: str


class AnthropicModel(Resource[AnthropicModelConfig, AnthropicModelOutputs]):
    """Agno Anthropic Claude model resource.

    Creates and returns an Agno Claude instance configured with the provided
    parameters. Dependent resources (like agno/agent) receive the resource
    instance and call the model() method to get the actual Claude object.

    This is a thin wrapper - the Claude instance is created on-demand
    via the model() method. No API validation is performed since the
    Agno SDK handles authentication when the model is actually used.

    Usage by dependent resources:
        model_resource: Dependency[AnthropicModel]

        async def on_create(self):
            claude = self.model_resource.model()  # Get Claude instance
            agent = AgnoAgent(model=claude, ...)

    Lifecycle:
        - on_create: Return serializable metadata
        - on_update: Return serializable metadata
        - on_delete: No-op (stateless)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "models/anthropic"

    def model(self) -> Claude:
        """Return the configured Claude instance.

        Called by dependent resources (e.g., agno/agent) that need
        the actual SDK object to pass to their constructors.

        Returns:
            Configured Claude instance ready for use with Agno agents.
        """
        kwargs = self.config.model_dump(exclude_none=True)
        kwargs["api_key"] = str(self.config.api_key)

        return Claude(**kwargs)

    async def on_create(self) -> AnthropicModelOutputs:
        """Create resource and return serializable outputs."""
        return AnthropicModelOutputs(model_id=self.config.id)

    async def on_update(self, previous_config: AnthropicModelConfig) -> AnthropicModelOutputs:
        """Update resource and return serializable outputs.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            AnthropicModelOutputs with updated model_id.
        """
        return AnthropicModelOutputs(model_id=self.config.id)

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
