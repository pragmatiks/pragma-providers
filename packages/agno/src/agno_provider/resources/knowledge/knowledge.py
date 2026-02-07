"""Agno Knowledge resource for semantic search.

Provides vector storage configuration for semantic search capabilities.
"""

from __future__ import annotations

from typing import ClassVar

from agno.knowledge.knowledge import Knowledge as AgnoKnowledge
from pragma_sdk import Config, Dependency, Outputs

from agno_provider.resources.base import AgnoResource, AgnoSpec
from agno_provider.resources.knowledge.embedder.openai import EmbedderOpenAI, EmbedderOpenAISpec
from agno_provider.resources.vectordb.qdrant import VectordbQdrant, VectordbQdrantSpec


class KnowledgeSpec(AgnoSpec):
    """Specification for reconstructing Agno Knowledge at runtime.

    Contains all necessary information to create a Knowledge instance
    with all nested dependencies.

    Attributes:
        name: Knowledge base name.
        max_results: Maximum search results to return.
        vector_db_spec: Nested spec for vector database configuration.
        embedder_spec: Optional nested spec for embedder configuration.
    """

    name: str
    max_results: int = 10
    vector_db_spec: VectordbQdrantSpec
    embedder_spec: EmbedderOpenAISpec | None = None


class KnowledgeConfig(Config):
    """Configuration for Agno Knowledge.

    Provides vector storage configuration for semantic search.

    Attributes:
        vector_db: Required vector store for embeddings.
        embedder: Optional embedder override for custom embedding model.
        max_results: Maximum search results to return.
    """

    vector_db: Dependency[VectordbQdrant]
    embedder: Dependency[EmbedderOpenAI] | None = None
    max_results: int = 10


class KnowledgeOutputs(Outputs):
    """Outputs from Agno Knowledge resource.

    Attributes:
        pip_dependencies: Python packages required.
        spec: Specification for reconstructing the knowledge base at runtime.
    """

    pip_dependencies: list[str]
    spec: KnowledgeSpec


class Knowledge(AgnoResource[KnowledgeConfig, KnowledgeOutputs, KnowledgeSpec]):
    """Agno Knowledge resource for semantic search.

    Wraps Agno's Knowledge class to provide vector storage configuration
    for semantic search capabilities.

    The outputs include a KnowledgeSpec that can be used to reconstruct
    the knowledge base at runtime via Knowledge.from_spec().

    Example YAML:
        provider: agno
        resource: knowledge
        name: docs
        config:
          vector_db:
            provider: agno
            resource: vectordb/qdrant
            name: embeddings
          max_results: 5

    Runtime reconstruction via spec:
        knowledge = Knowledge.from_spec(spec)

    Lifecycle:
        - on_create: Resolve dependencies, return outputs with spec
        - on_update: Re-resolve dependencies, return updated outputs with spec
        - on_delete: No-op (stateless wrapper)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "knowledge"

    @staticmethod
    def from_spec(spec: KnowledgeSpec) -> AgnoKnowledge:
        """Factory: construct Agno Knowledge from spec.

        Args:
            spec: The knowledge specification.

        Returns:
            Configured AgnoKnowledge instance.
        """
        vectordb = VectordbQdrant.from_spec(spec.vector_db_spec)

        return AgnoKnowledge(
            name=spec.name,
            vector_db=vectordb,
            max_results=spec.max_results,
        )

    def _build_spec(self) -> KnowledgeSpec:
        """Build spec from resolved dependencies.

        Creates a specification that can be serialized and used to
        reconstruct the knowledge base at runtime.

        Returns:
            KnowledgeSpec with all configuration fields.

        Raises:
            RuntimeError: If vector_db dependency is not resolved.
        """
        vector_db_resource = self.config.vector_db._resolved

        if vector_db_resource is None or vector_db_resource.outputs is None:
            msg = "vector_db dependency not resolved"
            raise RuntimeError(msg)

        vector_db_spec = vector_db_resource.outputs.spec

        embedder_spec = None

        if self.config.embedder is not None:
            embedder_resource = self.config.embedder._resolved

            if embedder_resource is not None and embedder_resource.outputs is not None:
                embedder_spec = embedder_resource.outputs.spec

        return KnowledgeSpec(
            name=self.name,
            max_results=self.config.max_results,
            vector_db_spec=vector_db_spec,
            embedder_spec=embedder_spec,
        )

    def _get_pip_dependencies(self) -> list[str]:
        """Aggregate pip dependencies from vectordb and embedder.

        Returns:
            Deduplicated list of pip packages required.
        """
        deps: set[str] = set()

        vector_db = self.config.vector_db._resolved

        if vector_db is not None and vector_db.outputs is not None:
            deps.update(vector_db.outputs.pip_dependencies)

        if self.config.embedder is not None:
            embedder = self.config.embedder._resolved

            if embedder is not None and embedder.outputs is not None:
                deps.update(embedder.outputs.pip_dependencies)

        return sorted(deps)

    async def _build_outputs(self) -> KnowledgeOutputs:
        """Build outputs from resolved dependencies.

        Returns:
            KnowledgeOutputs with pip dependencies and spec.
        """
        await self.config.vector_db.resolve()

        if self.config.embedder is not None:
            await self.config.embedder.resolve()

        return KnowledgeOutputs(
            pip_dependencies=self._get_pip_dependencies(),
            spec=self._build_spec(),
        )

    async def on_create(self) -> KnowledgeOutputs:
        """Create resource and return serializable outputs.

        Returns:
            KnowledgeOutputs with pip dependencies and spec.
        """
        return await self._build_outputs()

    async def on_update(self, previous_config: KnowledgeConfig) -> KnowledgeOutputs:  # noqa: ARG002
        """Update resource and return serializable outputs.

        Returns:
            KnowledgeOutputs with updated pip dependencies and spec.
        """
        return await self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
