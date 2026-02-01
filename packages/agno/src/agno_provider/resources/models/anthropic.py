"""Agno Anthropic model resource wrapping the Claude class."""

from __future__ import annotations

from typing import ClassVar

from agno.models.anthropic import Claude
from pragma_sdk import Field

from agno_provider.resources.models.base import Model, ModelConfig


class AnthropicModelConfig(ModelConfig):
    """Configuration for Agno Anthropic Claude model.

    Maps to Agno's Claude class from agno.models.anthropic.

    Attributes:
        api_key: Anthropic API key. Use a FieldReference to inject from pragma/secret.
        max_tokens: Maximum tokens in responses. Defaults to 8192 (Agno default).
        temperature: Sampling temperature (0.0-1.0). Optional.
        top_p: Nucleus sampling parameter. Optional.
        top_k: Top-k sampling parameter. Optional.
        stop_sequences: Stop sequences to end generation. Optional.
    """

    api_key: Field[str]
    max_tokens: int = 8192
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None


class AnthropicModel(Model[AnthropicModelConfig, Claude]):
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
    """

    resource: ClassVar[str] = "models/anthropic"

    def model(self) -> Claude:
        """Return the configured Claude instance.

        Called by dependent resources (e.g., agno/agent) that need
        the actual SDK object to pass to their constructors.

        Returns:
            Configured Claude instance ready for use with Agno agents.
        """
        return Claude(**self.config.model_dump(exclude_none=True))
