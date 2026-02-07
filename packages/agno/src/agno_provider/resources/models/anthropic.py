"""Agno Anthropic model resource wrapping the Claude class."""

from __future__ import annotations

from typing import ClassVar, Literal

from agno.models.anthropic import Claude
from pragma_sdk import Field

from agno_provider.resources.base import AgnoSpec
from agno_provider.resources.models.base import Model, ModelConfig, ModelOutputs


class AnthropicModelSpec(AgnoSpec):
    """Specification for an Anthropic Claude model.

    Used for serializing model configuration to outputs.
    Use AnthropicModel.from_spec() to reconstruct the Claude instance at runtime.

    Attributes:
        type: Discriminator field, always "anthropic".
        id: Model identifier (e.g., "claude-sonnet-4-20250514").
        api_key: Anthropic API key.
        max_tokens: Maximum tokens in responses.
        temperature: Sampling temperature (0.0-1.0).
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        stop_sequences: Stop sequences to end generation.
    """

    type: Literal["anthropic"] = "anthropic"
    id: str
    api_key: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None


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


class AnthropicModelOutputs(ModelOutputs):
    """Outputs from Anthropic model resource.

    Attributes:
        spec: The model specification for runtime reconstruction.
    """

    spec: AnthropicModelSpec


class AnthropicModel(Model[AnthropicModelConfig, AnthropicModelOutputs, AnthropicModelSpec, Claude]):
    """Agno Anthropic Claude model resource.

    Creates and returns an Agno Claude instance configured with the provided
    parameters. The Claude instance is created via from_spec() at runtime.

    This is a thin wrapper - the Claude instance is created on-demand.
    No API validation is performed since the Agno SDK handles authentication
    when the model is actually used.

    Runtime reconstruction from spec:
        ```python
        spec = outputs.spec
        model = AnthropicModel.from_spec(spec)
        ```
    """

    resource: ClassVar[str] = "models/anthropic"

    @staticmethod
    def from_spec(spec: AnthropicModelSpec) -> Claude:
        """Factory: construct Agno Claude object from spec.

        Args:
            spec: The model specification.

        Returns:
            Configured Claude instance ready for use.
        """
        return Claude(**spec.model_dump(exclude={"type"}, exclude_none=True))

    def _build_spec(self) -> AnthropicModelSpec:
        """Build spec from current config.

        Creates a specification that can be serialized and used to
        reconstruct the model at runtime.

        Returns:
            AnthropicModelSpec with all configuration fields.
        """
        return AnthropicModelSpec(
            id=self.config.id,
            api_key=str(self.config.api_key),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            stop_sequences=self.config.stop_sequences,
        )

    def _build_outputs(self) -> AnthropicModelOutputs:
        """Build outputs from current config.

        Returns:
            AnthropicModelOutputs with spec.
        """
        return AnthropicModelOutputs(spec=self._build_spec())

    async def on_create(self) -> AnthropicModelOutputs:
        """Create returns serializable outputs with spec.

        Returns:
            AnthropicModelOutputs with spec.
        """
        return self._build_outputs()

    async def on_update(self, previous_config: AnthropicModelConfig) -> AnthropicModelOutputs:  # noqa: ARG002
        """Update returns serializable outputs with spec.

        Args:
            previous_config: The previous configuration (unused for stateless resource).

        Returns:
            AnthropicModelOutputs with spec.
        """
        return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
