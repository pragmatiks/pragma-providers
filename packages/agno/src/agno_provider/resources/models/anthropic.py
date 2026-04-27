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
EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]


INCOMPATIBLE_MODE_BY_PREFIX: dict[str, ThinkingMode] = {
    "claude-opus-4-7": "extended",
    "claude-haiku-4-5": "adaptive",
}
"""Maps a model id prefix to the thinking_mode it does NOT support.

Per the Anthropic compatibility matrix, ``claude-opus-4-7*`` is adaptive-only
and ``claude-haiku-4-5*`` is extended-only. Older families (sonnet-4-5,
opus-4-5, sonnet-3-7) accept either mode and are not listed.

See https://platform.claude.com/docs/en/build-with-claude/extended-thinking
and https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking.
"""


NO_THINKING_MODEL_PREFIXES: frozenset[str] = frozenset({
    "claude-3-haiku-",
    "claude-3-5-haiku-",
})
"""Model id prefixes that do not support extended thinking at all.

Mirrors the entries in ``agno.models.anthropic.Claude.NON_THINKING_MODELS``
(``claude-3-haiku-20240307``, ``claude-3-5-haiku-20241022``,
``claude-3-5-haiku-latest``). Any non-``"off"`` thinking_mode on a model
matching one of these prefixes is rejected by ``Claude.__post_init__``;
catching it here keeps the failure at apply time rather than runner-pod
startup.
"""


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


def _build_output_config_param(
    thinking_mode: ThinkingMode,
    effort: EffortLevel | None,
) -> dict[str, Any] | None:
    """Translate ``effort`` into agno's ``output_config`` dict.

    Anthropic's adaptive-thinking mode pairs with a top-level ``output_config``
    object that carries the ``effort`` knob (``low`` / ``medium`` / ``high`` /
    ``xhigh`` / ``max``). Agno's ``Claude`` class plumbs ``output_config``
    straight through to the Anthropic API.

    Args:
        thinking_mode: One of "off", "extended", or "adaptive".
        effort: Effort level for adaptive thinking. Only valid when
            ``thinking_mode`` is "adaptive".

    Returns:
        Dict to pass to ``Claude(output_config=...)``, or ``None`` when no
        effort is configured.
    """
    if thinking_mode != "adaptive" or effort is None:
        return None

    return {"effort": effort}


def _validate_thinking_fields(
    thinking_mode: ThinkingMode,
    thinking_budget_tokens: int | None,
    effort: EffortLevel | None,
) -> None:
    """Enforce per-mode constraints on thinking-related fields.

    ``"off"`` rejects both ``thinking_budget_tokens`` and ``effort``.
    ``"extended"`` requires ``thinking_budget_tokens`` to be set, and when
    it is a concrete integer it must be ``>= 1024`` (Anthropic API
    minimum); ``effort`` is rejected. ``"adaptive"`` rejects
    ``thinking_budget_tokens`` and accepts optional ``effort``.

    Numeric checks against ``thinking_budget_tokens`` are skipped when
    the value is a ``FieldReference`` (still unresolved). Reference
    values get re-validated on the resolved Spec where they are
    guaranteed to be concrete integers. The whole helper is skipped
    when ``thinking_mode`` itself is a ``FieldReference``; mode-aware
    validation re-runs against the resolved Spec.

    Args:
        thinking_mode: One of "off", "extended", or "adaptive".
        thinking_budget_tokens: Token budget for extended thinking.
        effort: Effort level for adaptive thinking.

    Raises:
        ValueError: If any field is set in a mode that does not accept
            it, or if ``thinking_budget_tokens`` is missing or below
            1024 (when concrete) in ``"extended"`` mode.
    """
    if not isinstance(thinking_mode, str):
        return

    if thinking_mode == "off":
        if thinking_budget_tokens is not None:
            msg = "thinking_budget_tokens must be None when thinking_mode is 'off'"
            raise ValueError(msg)
        if effort is not None:
            msg = "effort must be None when thinking_mode is 'off'"
            raise ValueError(msg)
        return

    if thinking_mode == "extended":
        if thinking_budget_tokens is None:
            msg = "extended thinking requires thinking_budget_tokens >= 1024 (Anthropic API minimum)"
            raise ValueError(msg)
        if isinstance(thinking_budget_tokens, int) and thinking_budget_tokens < 1024:
            msg = "extended thinking requires thinking_budget_tokens >= 1024 (Anthropic API minimum)"
            raise ValueError(msg)
        if effort is not None:
            msg = "effort must be None when thinking_mode is 'extended' (effort is only valid for 'adaptive')"
            raise ValueError(msg)
        return

    if thinking_budget_tokens is not None:
        msg = "thinking_budget_tokens must be None when thinking_mode is 'adaptive' (adaptive picks its own budget)"
        raise ValueError(msg)


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
            when ``thinking_mode == "extended"``; must be ``>= 1024``
            (Anthropic API minimum) and less than ``max_tokens``. Must be
            ``None`` for the other modes.
        effort: Effort knob for ``"adaptive"`` thinking. Only valid when
            ``thinking_mode == "adaptive"``; must be ``None`` for the other
            modes. Defaults to ``None``, which lets Anthropic pick its
            default effort level (currently ``"high"``).
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
    effort: EffortLevel | None = None

    @model_validator(mode="after")
    def validate_thinking_fields(self) -> AnthropicModelSpec:
        """Enforce per-mode constraints on thinking-related fields.

        Returns:
            Self after validation.

        Raises:
            ValueError: If ``thinking_budget_tokens`` is greater than or
                equal to ``max_tokens``.
        """
        _validate_thinking_fields(self.thinking_mode, self.thinking_budget_tokens, self.effort)

        if (
            self.thinking_budget_tokens is not None
            and self.max_tokens is not None
            and self.thinking_budget_tokens >= self.max_tokens
        ):
            msg = (
                f"thinking_budget_tokens ({self.thinking_budget_tokens}) must be less "
                f"than max_tokens ({self.max_tokens}) (Anthropic API constraint)"
            )
            raise ValueError(msg)

        return self


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
            model pick its own budget per turn. Per-model compatibility is
            checked in ``on_create`` against Anthropic's documented matrix
            (e.g. ``claude-opus-4-7`` is adaptive-only; ``claude-haiku-4-5``
            is extended-only).
        thinking_budget_tokens: Token budget when ``thinking_mode`` is
            ``"extended"``. Required in that mode and must be ``>= 1024``
            (Anthropic API minimum) and less than ``max_tokens``. Must be
            omitted for ``"off"`` and ``"adaptive"``.
        effort: Effort knob when ``thinking_mode`` is ``"adaptive"``. One of
            ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, ``"max"``. Must
            be omitted for ``"off"`` and ``"extended"``. When omitted in
            adaptive mode, Anthropic uses its default effort (currently
            ``"high"``).
    """

    api_key: SensitiveField[str]
    max_tokens: Field[int] = 8192
    temperature: Field[float] | None = None
    top_p: Field[float] | None = None
    top_k: Field[int] | None = None
    stop_sequences: Field[list[str]] | None = None
    thinking_mode: Field[ThinkingMode] = "off"
    thinking_budget_tokens: Field[int] | None = None
    effort: Field[EffortLevel] | None = None

    @model_validator(mode="after")
    def validate_thinking_fields(self) -> AnthropicModelConfig:
        """Enforce per-mode constraints on thinking-related fields.

        The ``budget < max_tokens`` check only fires when both fields are
        concrete integers. If either is a ``FieldReference`` it is left
        for the resolved Spec to validate.

        Returns:
            Self after validation.

        Raises:
            ValueError: If both ``thinking_budget_tokens`` and ``max_tokens``
                are concrete integers and the budget is greater than or
                equal to ``max_tokens``.
        """
        _validate_thinking_fields(self.thinking_mode, self.thinking_budget_tokens, self.effort)

        if (
            isinstance(self.thinking_budget_tokens, int)
            and isinstance(self.max_tokens, int)
            and self.thinking_budget_tokens >= self.max_tokens
        ):
            msg = (
                f"thinking_budget_tokens ({self.thinking_budget_tokens}) must be less "
                f"than max_tokens ({self.max_tokens}) (Anthropic API constraint)"
            )
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
        thinking, ``{"type": "adaptive"}`` for adaptive). When ``effort`` is
        set (adaptive mode only), it is forwarded as
        ``output_config={"effort": <level>}`` which agno passes through to
        the Anthropic API. The high-level fields are not forwarded as
        kwargs because Claude does not accept them directly.

        Args:
            spec: The model specification.

        Returns:
            Configured Claude instance ready for use.
        """
        kwargs = spec.model_dump(
            exclude={"type", "thinking_mode", "thinking_budget_tokens", "effort"},
            exclude_none=True,
        )

        thinking = _build_thinking_param(spec.thinking_mode, spec.thinking_budget_tokens)
        if thinking is not None:
            kwargs["thinking"] = thinking

        output_config = _build_output_config_param(spec.thinking_mode, spec.effort)
        if output_config is not None:
            kwargs["output_config"] = output_config

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
            effort=self.config.effort,
        )

    def _build_outputs(self) -> AnthropicModelOutputs:
        """Build outputs from current config.

        Returns:
            AnthropicModelOutputs with spec.
        """
        return AnthropicModelOutputs(spec=self._build_spec())

    async def on_create(self) -> AnthropicModelOutputs:
        """Validate credentials and model/mode compatibility, then return outputs.

        Performs a lightweight ``client.models.retrieve`` round-trip so an
        invalid API key or nonexistent model id fails the resource at
        create time rather than on first agent invocation. Then checks the
        configured ``thinking_mode`` against Anthropic's per-model matrix so
        bad combinations (e.g. ``claude-opus-4-7`` with extended thinking)
        fail the resource here rather than at runner-pod startup. Any error
        propagates so the runtime marks the resource FAILED.

        Returns:
            AnthropicModelOutputs with spec.
        """
        await self._validate_credentials()
        self._validate_model_thinking_compatibility()
        return self._build_outputs()

    def _validate_model_thinking_compatibility(self) -> None:
        """Reject model/thinking-mode combinations Anthropic does not accept.

        Some Claude families only support one thinking mode — for example,
        ``claude-opus-4-7`` is adaptive-only and ``claude-haiku-4-5`` is
        extended-only. Legacy haiku families (``claude-3-haiku-*``,
        ``claude-3-5-haiku-*``) do not support extended thinking at all
        and reject any non-``"off"`` mode. Older sonnet/opus families
        (sonnet-4-5, opus-4-5, sonnet-3-7) accept manual thinking and
        are not flagged here. ``"off"`` always passes since both modes
        can be disabled.

        Skipped when ``self.config.id`` or ``self.config.thinking_mode``
        is a ``FieldReference``; the resolved Spec re-runs equivalent
        per-mode validation before reaching the runner.

        See https://platform.claude.com/docs/en/build-with-claude/extended-thinking
        and https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
        for the authoritative compatibility matrix.

        Raises:
            ValueError: If the model id and thinking_mode combination is
                not supported by Anthropic.
        """
        if not isinstance(self.config.id, str) or not isinstance(self.config.thinking_mode, str):
            return

        if self.config.thinking_mode == "off":
            return

        for prefix in NO_THINKING_MODEL_PREFIXES:
            if self.config.id.startswith(prefix):
                msg = (
                    f"Model {self.config.id!r} does not support extended thinking; "
                    f"thinking_mode must be 'off'. "
                    f"See https://platform.claude.com/docs/en/build-with-claude/extended-thinking "
                    f"for the per-model compatibility matrix."
                )
                raise ValueError(msg)

        for prefix, incompatible_mode in INCOMPATIBLE_MODE_BY_PREFIX.items():
            if self.config.id.startswith(prefix) and self.config.thinking_mode == incompatible_mode:
                msg = (
                    f"Model {self.config.id!r} does not support thinking_mode={incompatible_mode!r}. "
                    f"See https://platform.claude.com/docs/en/build-with-claude/extended-thinking "
                    f"for the per-model compatibility matrix."
                )
                raise ValueError(msg)

    async def _validate_credentials(self) -> None:
        """Round-trip to Anthropic to verify the API key and model id.

        Any error from the Anthropic SDK propagates so the runtime marks
        the resource FAILED.
        """
        client = anthropic.AsyncAnthropic(api_key=str(self.config.api_key))
        await client.models.retrieve(self.config.id)

    async def on_update(self, previous_config: AnthropicModelConfig) -> AnthropicModelOutputs:  # noqa: ARG002
        """Validate model/mode compatibility and return serializable outputs with spec.

        Re-runs the model/thinking-mode compatibility check so updating an
        existing resource to an unsupported combination (e.g. switching
        ``thinking_mode`` to ``"extended"`` on ``claude-opus-4-7``) fails
        the update rather than silently propagating to the runner pod.

        Args:
            previous_config: The previous configuration (unused for stateless resource).

        Returns:
            AnthropicModelOutputs with spec.
        """
        self._validate_model_thinking_compatibility()
        return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""

    @classmethod
    def upgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs

    @classmethod
    def downgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs
