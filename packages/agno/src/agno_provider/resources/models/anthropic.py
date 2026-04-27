"""Agno Anthropic model resource wrapping the Claude class."""

from __future__ import annotations

from typing import Any, Literal

import anthropic
from agno.models.anthropic import Claude
from pragma_sdk import Field, SensitiveField
from pydantic import model_validator

from agno_provider.resources.base import AgnoSpec
from agno_provider.resources.models.base import Model, ModelConfig, ModelOutputs


ThinkingMode = Literal["off", "extended", "adaptive"]


def _build_thinking_param(
    thinking_mode: ThinkingMode,
    thinking_budget_tokens: int | None,
) -> dict[str, Any] | None:
    """Translate the high-level thinking mode into agno's ``thinking`` dict.

    Agno's ``Claude`` class accepts a ``thinking`` parameter shaped like
    ``{"type": "enabled", "budget_tokens": N}`` for extended thinking and
    ``{"type": "adaptive"}`` for adaptive thinking. ``None`` disables it.

    Args:
        thinking_mode: One of "off", "extended", or "adaptive".
        thinking_budget_tokens: Token budget for extended thinking. Required
            when ``thinking_mode`` is "extended"; ignored otherwise.

    Returns:
        Dict to pass to ``Claude(thinking=...)``, or ``None`` when disabled.
    """
    if thinking_mode == "off":
        return None

    if thinking_mode == "adaptive":
        return {"type": "adaptive"}

    return {"type": "enabled", "budget_tokens": thinking_budget_tokens}


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
        thinking_mode: Extended-thinking mode. ``"off"`` disables thinking,
            ``"extended"`` enables Anthropic's extended thinking with a fixed
            token budget, and ``"adaptive"`` lets the model pick its own
            thinking budget per turn.
        thinking_budget_tokens: Token budget for extended thinking. Required
            (positive int) when ``thinking_mode == "extended"``; must be
            ``None`` for the other modes.
    """

    type: Literal["anthropic"] = "anthropic"
    id: str
    api_key: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    thinking_mode: ThinkingMode = "off"
    thinking_budget_tokens: int | None = None


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
        thinking_mode: Extended-thinking mode. Defaults to ``"off"``. Set to
            ``"extended"`` to enable Anthropic extended thinking with a fixed
            ``thinking_budget_tokens`` budget, or ``"adaptive"`` to let the
            model pick its own budget per turn. Note that some Claude model
            ids (e.g. Claude 3 / 3.5 Haiku families) do not support thinking;
            agno will reject the combination at runtime.
        thinking_budget_tokens: Token budget when ``thinking_mode`` is
            ``"extended"``. Required (positive integer) in that mode and must
            be omitted for ``"off"`` and ``"adaptive"``.
    """

    api_key: SensitiveField[str]
    max_tokens: Field[int] = 8192
    temperature: Field[float] | None = None
    top_p: Field[float] | None = None
    top_k: Field[int] | None = None
    stop_sequences: Field[list[str]] | None = None
    thinking_mode: Field[ThinkingMode] = "off"
    thinking_budget_tokens: Field[int] | None = None

    @model_validator(mode="after")
    def validate_thinking_budget(self) -> AnthropicModelConfig:
        """Enforce that ``thinking_budget_tokens`` matches ``thinking_mode``.

        ``"extended"`` requires a positive ``thinking_budget_tokens``;
        ``"off"`` and ``"adaptive"`` reject any budget so the catalog cannot
        accidentally silently drop a configured budget when switching modes.

        Returns:
            Self after validation.

        Raises:
            ValueError: If the budget is missing/non-positive for
                ``"extended"``, or set for ``"off"`` / ``"adaptive"``.
        """
        if self.thinking_mode == "extended":
            if self.thinking_budget_tokens is None or self.thinking_budget_tokens <= 0:
                msg = "thinking_budget_tokens must be a positive integer when thinking_mode is 'extended'"
                raise ValueError(msg)
        elif self.thinking_budget_tokens is not None:
            msg = f"thinking_budget_tokens must be None when thinking_mode is {self.thinking_mode!r}"
            raise ValueError(msg)

        return self


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
    On create, the API key and model id are validated against the Anthropic
    API via ``client.models.retrieve``. Invalid credentials or a missing
    model id surface as a FAILED resource rather than silently succeeding
    and breaking on first agent invocation.

    Runtime reconstruction from spec:
        ```python
        spec = outputs.spec
        model = AnthropicModel.from_spec(spec)
        ```
    """

    @staticmethod
    def from_spec(spec: AnthropicModelSpec) -> Claude:
        """Factory: construct Agno Claude object from spec.

        ``thinking_mode`` and ``thinking_budget_tokens`` are translated into
        the ``thinking`` dict shape that ``agno.models.anthropic.Claude``
        expects (``{"type": "enabled", "budget_tokens": N}`` for extended
        thinking, ``{"type": "adaptive"}`` for adaptive). The high-level
        fields are not forwarded as kwargs because Claude does not accept
        them directly.

        Args:
            spec: The model specification.

        Returns:
            Configured Claude instance ready for use.
        """
        kwargs = spec.model_dump(
            exclude={"type", "thinking_mode", "thinking_budget_tokens"},
            exclude_none=True,
        )

        thinking = _build_thinking_param(spec.thinking_mode, spec.thinking_budget_tokens)
        if thinking is not None:
            kwargs["thinking"] = thinking

        return Claude(**kwargs)

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
            thinking_mode=self.config.thinking_mode,
            thinking_budget_tokens=self.config.thinking_budget_tokens,
        )

    def _build_outputs(self) -> AnthropicModelOutputs:
        """Build outputs from current config.

        Returns:
            AnthropicModelOutputs with spec.
        """
        return AnthropicModelOutputs(spec=self._build_spec())

    async def on_create(self) -> AnthropicModelOutputs:
        """Validate credentials against Anthropic and return outputs with spec.

        Performs a lightweight ``client.models.retrieve`` round-trip so an
        invalid API key or nonexistent model id fails the resource at
        create time rather than on first agent invocation. Any error from
        the Anthropic SDK propagates so the runtime marks the resource
        FAILED.

        Returns:
            AnthropicModelOutputs with spec.
        """
        await self._validate_credentials()
        return self._build_outputs()

    async def _validate_credentials(self) -> None:
        """Round-trip to Anthropic to verify the API key and model id.

        Any error from the Anthropic SDK propagates so the runtime marks
        the resource FAILED.
        """
        client = anthropic.AsyncAnthropic(api_key=str(self.config.api_key))
        await client.models.retrieve(self.config.id)

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

    @classmethod
    def upgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs

    @classmethod
    def downgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs
