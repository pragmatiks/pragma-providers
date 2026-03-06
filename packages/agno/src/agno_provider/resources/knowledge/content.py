"""Content resource for Agno knowledge sources.

Represents a single content source (URL or text) for ingestion into a Knowledge base.
Content manages its own lifecycle in the vectordb through the Knowledge dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pragma_sdk import Config, Dependency, Field, Outputs
from pydantic import model_validator

from agno_provider.resources.base import AgnoResource, AgnoSpec
from agno_provider.resources.knowledge.knowledge import Knowledge, KnowledgeSpec


if TYPE_CHECKING:
    from agno.knowledge.knowledge import Knowledge as AgnoKnowledge


READER_DEPENDENCIES = [
    "pypdf",
    "beautifulsoup4",
    "python-docx",
    "python-pptx",
    "arxiv",
    "youtube-transcript-api",
    "wikipedia",
    "aiofiles",
]


class ContentSpec(AgnoSpec):
    """Specification for content insertion into a Knowledge base.

    Contains all configuration needed to insert content via Knowledge.ainsert().

    Attributes:
        name: Content name for identification.
        knowledge_spec: Nested spec for the knowledge base configuration.
        url: Website, PDF, or any remote file URL.
        text_content: Raw inline text content.
        description: Content description.
        metadata: Custom metadata key-value pairs.
        topics: Topic tags for categorization.
    """

    name: str
    knowledge_spec: KnowledgeSpec
    url: str | None = None
    text_content: str | None = None
    description: str | None = None
    metadata: dict[str, str] | None = None
    topics: list[str] | None = None


class ContentConfig(Config):
    """Configuration for a knowledge content source.

    Supports two mutually exclusive content types:
    1. URL: Website, PDF, or any remote file URL
    2. Text: Raw inline text content

    Attributes:
        knowledge: Reference to the Knowledge resource for vectordb access.
        url: Website, PDF, or any remote file URL.
        text_content: Raw inline text content.
        name: Content name for identification.
        description: Content description.
        metadata: Custom metadata key-value pairs.
        topics: Topic tags for categorization.
    """

    knowledge: Dependency[Knowledge]
    url: Field[str] | None = None
    text_content: Field[str] | None = None
    name: Field[str] | None = None
    description: Field[str] | None = None
    metadata: Field[dict[str, str]] | None = None
    topics: Field[list[str]] | None = None

    @model_validator(mode="after")
    def validate_content_source(self) -> ContentConfig:
        """Validate that exactly one of url or text_content is provided.

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If neither or both url and text_content are provided.
        """
        has_url = self.url is not None and len(self.url) > 0
        has_text = self.text_content is not None

        if has_url and has_text:
            msg = "Exactly one of url or text_content must be provided, not both"
            raise ValueError(msg)

        if not has_url and not has_text:
            msg = "Either url or text_content must be provided"
            raise ValueError(msg)

        return self


class ContentOutputs(Outputs):
    """Outputs from Content resource.

    Attributes:
        spec: The content specification for runtime insertion.
        pip_dependencies: Python packages required for reading content.
    """

    spec: ContentSpec
    pip_dependencies: list[str]


class Content(AgnoResource[ContentConfig, ContentOutputs, ContentSpec]):
    """Content resource for Agno knowledge sources.

    Represents a single content source that can be ingested into a Knowledge base.
    Content manages its own lifecycle in the vectordb through the Knowledge dependency.

    This is a provider-side resource - no from_spec() is needed as Content is not
    reconstructed at agent runtime. The Knowledge already has access to the vectordb
    with all content inserted.

    Lifecycle:
        - on_create: Insert content into knowledge's vectordb
        - on_update: Re-insert content with upsert
        - on_delete: Remove content from vectordb by name
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "knowledge/content"

    async def _resolve_knowledge(self) -> AgnoKnowledge:
        """Resolve knowledge dependency and return Agno Knowledge instance.

        Returns:
            Configured AgnoKnowledge instance.

        Raises:
            RuntimeError: If knowledge dependency is not resolved.
        """
        await self.config.knowledge.resolve()

        knowledge_resource = self.config.knowledge._resolved

        if knowledge_resource is None or knowledge_resource.outputs is None:
            msg = "knowledge dependency not resolved"
            raise RuntimeError(msg)

        return Knowledge.from_spec(knowledge_resource.outputs.spec)

    def _build_spec(self) -> ContentSpec:
        """Build the content specification for runtime insertion.

        Returns:
            ContentSpec with all configuration fields.

        Raises:
            RuntimeError: If knowledge dependency is not resolved.
        """
        knowledge_resource = self.config.knowledge._resolved

        if knowledge_resource is None or knowledge_resource.outputs is None:
            msg = "knowledge dependency not resolved"
            raise RuntimeError(msg)

        return ContentSpec(
            name=self.config.name if self.config.name else self.name,
            knowledge_spec=knowledge_resource.outputs.spec,
            url=self.config.url,
            text_content=self.config.text_content,
            description=self.config.description,
            metadata=self.config.metadata,
            topics=self.config.topics,
        )

    def _build_outputs(self) -> ContentOutputs:
        """Build outputs from current config.

        Returns:
            ContentOutputs with spec and pip dependencies.
        """
        return ContentOutputs(
            spec=self._build_spec(),
            pip_dependencies=list(READER_DEPENDENCIES),
        )

    def _content_name(self) -> str:
        """Return the content name for vectordb operations."""
        return self.config.name if self.config.name else self.name

    def _insert_kwargs(self) -> dict:
        """Build kwargs for Knowledge.ainsert() from spec.

        Returns:
            Dict of kwargs for ainsert().
        """
        spec = self._build_spec()

        return spec.model_dump(
            exclude_none=True,
            exclude={"knowledge_spec"},
        )

    async def on_create(self) -> ContentOutputs:
        """Insert content into the knowledge's vectordb.

        Returns:
            ContentOutputs with content metadata.
        """
        knowledge = await self._resolve_knowledge()
        await knowledge.ainsert(**self._insert_kwargs())

        return self._build_outputs()

    async def on_update(self, previous_config: ContentConfig) -> ContentOutputs:  # noqa: ARG002
        """Re-insert content with upsert.

        Returns:
            ContentOutputs with updated content metadata.
        """
        knowledge = await self._resolve_knowledge()
        await knowledge.ainsert(**self._insert_kwargs(), upsert=True)

        return self._build_outputs()

    async def on_delete(self) -> None:
        """Remove content from the knowledge's vectordb by name."""
        knowledge = await self._resolve_knowledge()
        knowledge.remove_vectors_by_name(self._content_name())

    @classmethod
    def upgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs

    @classmethod
    def downgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs
