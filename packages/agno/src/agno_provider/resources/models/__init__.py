"""Model resources for Agno provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agno_provider.resources.models.anthropic import (
    AnthropicModel,
    AnthropicModelConfig,
    AnthropicModelSpec,
)
from agno_provider.resources.models.base import Model, ModelConfig, ModelOutputs
from agno_provider.resources.models.openai import (
    OpenAIModel,
    OpenAIModelConfig,
    OpenAIModelSpec,
)


if TYPE_CHECKING:
    from agno.models.base import Model as AgnoModel


__all__ = [
    "AnthropicModel",
    "AnthropicModelConfig",
    "Model",
    "ModelConfig",
    "ModelOutputs",
    "OpenAIModel",
    "OpenAIModelConfig",
    "model_from_spec",
]


def model_from_spec(
    spec: OpenAIModelSpec | AnthropicModelSpec,
) -> AgnoModel:
    """Factory: create model from spec based on type.

    Args:
        spec: The model specification (OpenAI or Anthropic).

    Returns:
        Configured Agno model instance.

    Raises:
        TypeError: If spec type is unknown.
    """
    if isinstance(spec, OpenAIModelSpec):
        return OpenAIModel.from_spec(spec)

    if isinstance(spec, AnthropicModelSpec):
        return AnthropicModel.from_spec(spec)

    msg = f"Unknown model spec type: {type(spec)}"
    raise TypeError(msg)
