"""OpenAI embedder resource wrapping Agno's OpenAIEmbedder.

Provides a from_spec() method that returns an OpenAIEmbedder instance for use
by dependent resources at runtime.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from agno.knowledge.embedder.openai import OpenAIEmbedder
from pragma_sdk import Config, Field, Outputs

from agno_provider.resources.base import AgnoResource, AgnoSpec


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class EmbedderOpenAISpec(AgnoSpec):
    """Specification for reconstructing OpenAI embedder at runtime.

    Contains all necessary information to create an OpenAIEmbedder instance.

    Attributes:
        id: OpenAI embedding model identifier (e.g., "text-embedding-3-small").
        api_key: OpenAI API key.
        dimensions: Override embedding dimensions (only for text-embedding-3-* models).
        encoding_format: Response encoding format.
        organization: OpenAI organization ID.
        base_url: Custom base URL for OpenAI-compatible APIs.
    """

    id: str
    api_key: str
    dimensions: int | None = None
    encoding_format: Literal["float", "base64"] = "float"
    organization: str | None = None
    base_url: str | None = None


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

    id: Field[str] = "text-embedding-3-small"
    api_key: Field[str]
    dimensions: Field[int] | None = None
    encoding_format: Field[Literal["float", "base64"]] = "float"
    organization: Field[str] | None = None
    base_url: Field[str] | None = None


class EmbedderOpenAIOutputs(Outputs):
    """Serializable outputs from OpenAI embedder resource.

    Attributes:
        pip_dependencies: Python packages required for this configuration.
        spec: Specification for reconstructing the embedder at runtime.
    """

    pip_dependencies: list[str]
    spec: EmbedderOpenAISpec


class EmbedderOpenAI(AgnoResource[EmbedderOpenAIConfig, EmbedderOpenAIOutputs, EmbedderOpenAISpec]):
    """OpenAI embedder resource wrapping Agno's OpenAIEmbedder.

    This resource is stateless - it just wraps configuration. The actual
    OpenAIEmbedder instance is created via from_spec() at runtime.

    Lifecycle:
        - on_create: Return serializable metadata
        - on_update: Return updated metadata
        - on_delete: No-op (stateless)

    Runtime reconstruction via spec:
        embedder = EmbedderOpenAI.from_spec(spec)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "knowledge/embedder/openai"

    @staticmethod
    def from_spec(spec: EmbedderOpenAISpec) -> OpenAIEmbedder:
        """Factory: construct Agno embedder from spec.

        Args:
            spec: The embedder specification.

        Returns:
            Configured OpenAIEmbedder instance.
        """
        return OpenAIEmbedder(**spec.model_dump(exclude_none=True))

    def _build_spec(self) -> EmbedderOpenAISpec:
        """Build spec from current config.

        Creates a specification that can be serialized and used to
        reconstruct the embedder at runtime.

        Returns:
            EmbedderOpenAISpec with all configuration fields.
        """
        return EmbedderOpenAISpec(
            id=self.config.id,
            api_key=str(self.config.api_key),
            dimensions=self.config.dimensions,
            encoding_format=self.config.encoding_format,
            organization=self.config.organization,
            base_url=self.config.base_url,
        )

    def _build_outputs(self) -> EmbedderOpenAIOutputs:
        """Build outputs from current config.

        Returns:
            EmbedderOpenAIOutputs with pip dependencies and spec.
        """
        return EmbedderOpenAIOutputs(
            pip_dependencies=[],
            spec=self._build_spec(),
        )

    async def on_create(self) -> EmbedderOpenAIOutputs:
        """Create returns serializable metadata with spec.

        Returns:
            EmbedderOpenAIOutputs with pip dependencies and spec.
        """
        return self._build_outputs()

    async def on_update(self, previous_config: EmbedderOpenAIConfig) -> EmbedderOpenAIOutputs:  # noqa: ARG002
        """Update returns serializable metadata with spec.

        Args:
            previous_config: The previous configuration (unused for stateless resource).

        Returns:
            EmbedderOpenAIOutputs with pip dependencies and spec.
        """
        return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
