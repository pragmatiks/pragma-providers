"""OpenAI embedder resource wrapping Agno's OpenAIEmbedder.

Provides an embedder() method that returns an OpenAIEmbedder instance for use
by dependent resources (e.g., agno/vectordb/qdrant).
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from agno.knowledge.embedder.openai import OpenAIEmbedder
from pragma_sdk import Config, Field, Outputs, Resource


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class EmbedderOpenAIConfig(Config):
    """Configuration for OpenAI embedder.

    Maps directly to Agno's OpenAIEmbedder constructor parameters.

    Attributes:
        id: OpenAI embedding model identifier.
        api_key: OpenAI API key. Use a FieldReference to inject from pragma/secret.
        dimensions: Override embedding dimensions (only for text-embedding-3-* models).
        organization: Optional OpenAI organization ID.
        base_url: Optional custom base URL for OpenAI-compatible APIs.
    """

    id: str = "text-embedding-3-small"
    api_key: Field[str]
    dimensions: int | None = None
    encoding_format: Literal["float", "base64"] = "float"
    organization: str | None = None
    base_url: str | None = None


class EmbedderOpenAIOutputs(Outputs):
    """Serializable outputs from OpenAI embedder resource.

    Attributes:
        model: The configured embedding model identifier.
        dimensions: Embedding dimensions.
        pip_dependencies: Python packages required for this configuration.
        ready: Whether the embedder is ready for use.
    """

    model: str
    dimensions: int
    pip_dependencies: list[str]
    ready: bool


class EmbedderOpenAI(Resource[EmbedderOpenAIConfig, EmbedderOpenAIOutputs]):
    """OpenAI embedder resource wrapping Agno's OpenAIEmbedder.

    This resource is stateless - it just wraps configuration. The actual
    OpenAIEmbedder instance is created on-demand via the embedder() method.

    Lifecycle:
        - on_create: Return serializable metadata
        - on_update: Return updated metadata
        - on_delete: No-op (stateless)

    Example usage with Qdrant:
        ```python
        embedder_resource = await embedder_dependency.resolve()
        vectordb = Qdrant(
            collection="embeddings",
            url="http://localhost:6333",
            embedder=embedder_resource.embedder(),
        )
        ```
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "knowledge/embedder/openai"

    def embedder(self) -> OpenAIEmbedder:
        """Returns the configured OpenAIEmbedder instance.

        Called by dependent resources (e.g., agno/vectordb/qdrant) that need
        the actual SDK object.
        """
        kwargs: dict[str, Any] = {
            "id": self.config.id,
            "api_key": str(self.config.api_key),
            "encoding_format": self.config.encoding_format,
        }

        if self.config.dimensions is not None:
            kwargs["dimensions"] = self.config.dimensions

        if self.config.organization is not None:
            kwargs["organization"] = self.config.organization

        if self.config.base_url is not None:
            kwargs["base_url"] = self.config.base_url

        return OpenAIEmbedder(**kwargs)

    def _get_dimensions(self) -> int:
        """Get embedding dimensions for the configured model.

        Returns:
            Embedding dimensions for the model.
        """
        if self.config.dimensions is not None:
            return self.config.dimensions

        return EMBEDDING_DIMENSIONS.get(self.config.id, 1536)

    async def on_create(self) -> EmbedderOpenAIOutputs:
        """Create returns serializable metadata only.

        Returns:
            EmbedderOpenAIOutputs containing model and dimensions.
        """
        return EmbedderOpenAIOutputs(
            model=self.config.id,
            dimensions=self._get_dimensions(),
            pip_dependencies=[],
            ready=True,
        )

    async def on_update(self, previous_config: EmbedderOpenAIConfig) -> EmbedderOpenAIOutputs:  # noqa: ARG002
        """Update returns serializable metadata.

        Returns:
            EmbedderOpenAIOutputs containing model and dimensions.
        """
        return EmbedderOpenAIOutputs(
            model=self.config.id,
            dimensions=self._get_dimensions(),
            pip_dependencies=[],
            ready=True,
        )

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
