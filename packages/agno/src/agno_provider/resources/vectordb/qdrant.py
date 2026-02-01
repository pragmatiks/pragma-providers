"""Agno Qdrant VectorDB resource."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from agno.vectordb.qdrant import Qdrant, SearchType
from pragma_sdk import Config, Dependency, Field, Outputs, Resource

from agno_provider.resources.knowledge.embedder.openai import EmbedderOpenAI


class VectordbQdrantConfig(Config):
    """Configuration for Agno Qdrant VectorDB.

    Provides Agno-compatible vector store configuration using Field references
    to connect to an existing Qdrant instance.

    Attributes:
        url: Qdrant server URL. Can reference qdrant/database outputs.
        collection: Collection name. Can reference qdrant/collection outputs.
        api_key: Optional API key for authentication.
        search_type: Search type - vector, keyword, or hybrid.
        embedder: Optional embedder resource for automatic vector generation.
    """

    url: Field[str]
    collection: Field[str]
    api_key: Field[str] | None = None
    search_type: Literal["vector", "keyword", "hybrid"] = "hybrid"
    embedder: Dependency[EmbedderOpenAI] | None = None


class VectordbQdrantOutputs(Outputs):
    """Outputs from Agno Qdrant VectorDB resource.

    Attributes:
        url: Qdrant server URL.
        collection: Collection name.
        search_type: Configured search type.
        pip_dependencies: Python packages required for this configuration.
        ready: Whether the vector store is ready for use.
    """

    url: str
    collection: str
    search_type: str
    pip_dependencies: list[str]
    ready: bool


class VectordbQdrant(Resource[VectordbQdrantConfig, VectordbQdrantOutputs]):
    """Agno Qdrant VectorDB resource.

    Wraps Agno's Qdrant class to provide vector storage for knowledge bases.
    Uses Field references to connect to an existing Qdrant instance.

    Example YAML:
        provider: agno
        resource: vectordb/qdrant
        name: embeddings
        config:
          url:
            $ref: qdrant/database/main.url
          collection:
            $ref: qdrant/collection/embeddings.name
          api_key:
            $ref: qdrant/database/main.api_key
          search_type: hybrid

    Usage by dependent resources:
        vector_db: Dependency[VectordbQdrant]

        async def on_create(self):
            qdrant_resource = await self.config.vector_db.resolve()
            db = qdrant_resource.vectordb()
            knowledge = Knowledge(vector_db=db, ...)

    Lifecycle:
        - on_create: Return configuration metadata
        - on_update: Return updated metadata
        - on_delete: No-op (stateless wrapper)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "vectordb/qdrant"

    def _get_search_type(self) -> SearchType:
        """Map search type string to Agno SearchType enum.

        Returns:
            Agno SearchType enum value.
        """
        search_type_map = {
            "vector": SearchType.vector,
            "keyword": SearchType.keyword,
            "hybrid": SearchType.hybrid,
        }
        return search_type_map[self.config.search_type]

    def _get_pip_dependencies(self) -> list[str]:
        """Get pip dependencies based on search type.

        Returns:
            List of pip packages required for this configuration.
        """
        if self.config.search_type in ("hybrid", "keyword"):
            return ["fastembed>=0.6.0"]

        return []

    def vectordb(self) -> Qdrant:
        """Return configured Agno Qdrant instance.

        Called by dependent resources (e.g., agno/knowledge) that need
        the vector database instance.

        Returns:
            Configured Agno Qdrant instance ready for use.
        """
        kwargs: dict[str, Any] = {
            "collection": str(self.config.collection),
            "url": str(self.config.url),
            "search_type": self._get_search_type(),
        }

        if self.config.api_key is not None:
            kwargs["api_key"] = str(self.config.api_key)

        if self.config.embedder is not None and self.config.embedder._resolved is not None:
            kwargs["embedder"] = self.config.embedder._resolved.embedder()

        return Qdrant(**kwargs)

    async def on_create(self) -> VectordbQdrantOutputs:
        """Create resource and return serializable outputs.

        Returns:
            VectordbQdrantOutputs with configuration metadata.
        """
        return VectordbQdrantOutputs(
            url=str(self.config.url),
            collection=str(self.config.collection),
            search_type=self.config.search_type,
            pip_dependencies=self._get_pip_dependencies(),
            ready=True,
        )

    async def on_update(self, previous_config: VectordbQdrantConfig) -> VectordbQdrantOutputs:  # noqa: ARG002
        """Update resource and return serializable outputs.

        Returns:
            VectordbQdrantOutputs with updated configuration metadata.
        """
        return VectordbQdrantOutputs(
            url=str(self.config.url),
            collection=str(self.config.collection),
            search_type=self.config.search_type,
            pip_dependencies=self._get_pip_dependencies(),
            ready=True,
        )

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
