"""Abstract base class for Agno model resources.

Provides a common interface for all model resources (OpenAI, Anthropic, etc.)
so dependent resources can accept any model type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from pragma_sdk import Config, Outputs, Resource


if TYPE_CHECKING:
    from agno.models.base import Model as AgnoModel


class ModelConfig(Config):
    """Base configuration for all model resources.

    Attributes:
        id: The model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
    """

    id: str


class ModelOutputs(Outputs):
    """Shared outputs for all model resources.

    Attributes:
        model_id: The configured model identifier.
    """

    model_id: str


class Model[ModelConfigT: ModelConfig, ModelT: "AgnoModel"](Resource[ModelConfigT, ModelOutputs], ABC):
    """Abstract base for Agno model resources.

    All model resources must implement `model()` to return the
    configured Agno model instance.

    Type Parameters:
        ModelConfigT: The config type for this model resource.
        ModelT: The Agno model type returned by model().
    """

    provider: ClassVar[str] = "agno"

    @abstractmethod
    def model(self) -> ModelT:
        """Return the configured Agno model instance."""
        ...

    async def on_create(self) -> ModelOutputs:
        """Create returns serializable metadata only.

        Returns:
            ModelOutputs containing model_id.
        """
        return ModelOutputs(model_id=self.config.id)

    async def on_update(self, previous_config: ModelConfigT) -> ModelOutputs:  # noqa: ARG002
        """Update returns serializable metadata.

        Returns:
            ModelOutputs containing model_id.
        """
        return ModelOutputs(model_id=self.config.id)

    async def on_delete(self) -> None:
        """Delete is a no-op since model resources are stateless."""
