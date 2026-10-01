"""Agno Qdrant VectorDB resource."""

from __future__ import annotations

from typing import Any, Literal

from agno.vectordb.qdrant import Qdrant, SearchType
from pragma_sdk import Config, Dependency, Field, ImmutableField, Outputs
from pydantic import computed_field

from agno_provider.resources.base import AgnoResource, AgnoSpec
from agno_provider.resources.knowledge.embedder.openai import (
    EmbedderOpenAI,
    EmbedderOpenAISpec,
)


class VectordbQdrantSpec(AgnoSpec):
    """Specification for reconstructing Qdrant VectorDB at runtime.

    Contains all necessary information to create a Qdrant instance.

    Attributes:
        url: Qdrant server URL.
        collection: Collection name.
        api_key: Optional API key for authentication.
        search_type: Search type. Only vector search is supported.
        embedder_spec: Nested spec for embedder configuration.
    """

    url: str
    collection: str
    api_key: str | None = None
    search_type: Literal["vector"] = "vector"
    embedder_spec: EmbedderOpenAISpec | None = None

    @computed_field
    @property
    def search_type_enum(self) -> SearchType:
        """Convert string search_type to Agno SearchType enum."""
        return SearchType(self.search_type)


class VectordbQdrantConfig(Config):
    """Configuration for Agno Qdrant VectorDB.

    Provides Agno-compatible vector store configuration using Field references
    to connect to an existing Qdrant instance.

    Attributes:
        url: Qdrant server URL. Can reference qdrant/database outputs.
        collection: Collection name. Can reference qdrant/collection outputs.
        api_key: Optional API key for authentication.
        search_type: Search type. Only vector search is supported, because a
            qdrant/collection holds one unnamed dense vector.
        embedder: Optional embedder resource for automatic vector generation.
    """

    url: ImmutableField[str]
    collection: ImmutableField[str]
    api_key: Field[str] | None = None
    search_type: Field[Literal["vector"]] = "vector"
    embedder: Dependency[EmbedderOpenAI] | None = None


class VectordbQdrantOutputs(Outputs):
    """Outputs from Agno Qdrant VectorDB resource.

    Attributes:
        spec: Specification for reconstructing the vectordb at runtime.
        pip_dependencies: Python packages required for this configuration.
    """

    spec: VectordbQdrantSpec
    pip_dependencies: list[str]


class VectordbQdrant(AgnoResource[VectordbQdrantConfig, VectordbQdrantOutputs, VectordbQdrantSpec]):
    """Agno Qdrant VectorDB resource.

    Wraps Agno's Qdrant class to provide vector storage for knowledge bases.
    Uses Field references to connect to an existing Qdrant instance.

    Example YAML:
        provider: agno
        resource: vectordb/qdrant
        name: embeddings
        config:
          url:
            provider: qdrant
            resource: database
            name: main
            field: url
          collection:
            provider: qdrant
            resource: collection
            name: embeddings
            field: name
          api_key:
            provider: qdrant
            resource: database
            name: main
            field: api_key
          search_type: vector

    Runtime reconstruction via spec:
        qdrant = VectordbQdrant.from_spec(spec)

    Lifecycle:
        - on_create: Return configuration metadata
        - on_update: Return updated metadata
        - on_delete: No-op (stateless wrapper)
    """

    computed = True

    @staticmethod
    def from_spec(spec: VectordbQdrantSpec) -> Qdrant:
        """Factory: construct Agno Qdrant from spec.

        Args:
            spec: The vectordb specification.

        Returns:
            Configured Qdrant instance.
        """
        kwargs: dict[str, Any] = {
            "url": spec.url,
            "collection": spec.collection,
            "search_type": spec.search_type_enum,
        }

        if spec.api_key:
            kwargs["api_key"] = spec.api_key

        if spec.embedder_spec:
            kwargs["embedder"] = EmbedderOpenAI.from_spec(spec.embedder_spec)

        return Qdrant(**kwargs)

    def _build_spec(self) -> VectordbQdrantSpec:
        """Build spec from current config.

        Creates a specification that can be serialized and used to
        reconstruct the vectordb at runtime.

        Returns:
            VectordbQdrantSpec with all configuration fields.
        """
        api_key = str(self.config.api_key) if self.config.api_key is not None else None

        embedder_spec = None
        if (
            self.config.embedder is not None
            and self.config.embedder._resolved is not None
            and self.config.embedder._resolved.outputs is not None
        ):
            embedder_spec = self.config.embedder._resolved.outputs.spec

        return VectordbQdrantSpec(
            url=str(self.config.url),
            collection=str(self.config.collection),
            api_key=api_key,
            search_type=self.config.search_type,
            embedder_spec=embedder_spec,
        )

    def _build_outputs(self) -> VectordbQdrantOutputs:
        """Build outputs from current config.

        Returns:
            VectordbQdrantOutputs with spec and no pip dependencies.
        """
        return VectordbQdrantOutputs(
            spec=self._build_spec(),
            pip_dependencies=[],
        )

    async def on_create(self) -> VectordbQdrantOutputs:
        """Create resource and return serializable outputs.

        Returns:
            VectordbQdrantOutputs with configuration metadata.
        """
        return self._build_outputs()

    async def on_update(self, previous_config: VectordbQdrantConfig | None) -> VectordbQdrantOutputs:  # noqa: ARG002
        """Update resource and return serializable outputs.

        Returns:
            VectordbQdrantOutputs with updated configuration metadata.
        """
        return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
