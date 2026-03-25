"""Tests for Agno knowledge/content resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pragma_sdk import Dependency, LifecycleState
from pragma_sdk.provider import ProviderHarness
from pydantic import ValidationError

from agno_provider.resources.knowledge.content import (
    READER_DEPENDENCIES,
    Content,
    ContentConfig,
    ContentOutputs,
    ContentSpec,
)
from agno_provider.resources.knowledge.knowledge import (
    Knowledge,
    KnowledgeConfig,
    KnowledgeOutputs,
    KnowledgeSpec,
)
from agno_provider.resources.vectordb.qdrant import (
    VectordbQdrant,
    VectordbQdrantConfig,
    VectordbQdrantOutputs,
    VectordbQdrantSpec,
)


ContentConfig.model_rebuild()


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_vectordb_resource(name: str = "test-vectordb") -> VectordbQdrant:
    """Create a VectordbQdrant resource for testing."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="vector",
    )

    return VectordbQdrant(
        name=name,
        config=config,
        lifecycle_state=LifecycleState.READY,
        outputs=VectordbQdrantOutputs(
            spec=VectordbQdrantSpec(
                url="http://localhost:6333",
                collection="test-collection",
                search_type="vector",
            ),
            pip_dependencies=[],
        ),
    )


def create_knowledge_resource(name: str = "test-knowledge") -> Knowledge:
    """Create a Knowledge resource for testing."""
    vectordb = create_vectordb_resource()

    vectordb_dep = Dependency[VectordbQdrant](
        provider="agno",
        resource="vectordb/qdrant",
        name=vectordb.name,
    )
    vectordb_dep._resolved = vectordb

    config = KnowledgeConfig(
        vector_db=vectordb_dep,
        max_results=10,
    )

    return Knowledge(
        name=name,
        config=config,
        lifecycle_state=LifecycleState.READY,
        outputs=KnowledgeOutputs(
            pip_dependencies=[],
            spec=KnowledgeSpec(
                name=name,
                max_results=10,
                vector_db_spec=VectordbQdrantSpec(
                    url="http://localhost:6333",
                    collection="test-collection",
                    search_type="vector",
                ),
            ),
        ),
    )


def create_content(
    name: str = "test-content",
    knowledge: Knowledge | None = None,
    url: str | None = None,
    text_content: str | None = None,
    content_name: str | None = None,
    description: str | None = None,
    metadata: dict[str, str] | None = None,
    topics: list[str] | None = None,
    outputs: ContentOutputs | None = None,
) -> Content:
    """Create a Content resource for testing."""
    if knowledge is None:
        knowledge = create_knowledge_resource()

    knowledge_dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name=knowledge.name,
    )
    knowledge_dep._resolved = knowledge

    config = ContentConfig(
        knowledge=knowledge_dep,
        url=url,
        text_content=text_content,
        name=content_name,
        description=description,
        metadata=metadata,
        topics=topics,
    )

    return Content(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


def test_resource_metadata_resource_type() -> None:
    """Resource has correct resource type."""
    assert Content.resource == "knowledge/content"


def test_config_validation_requires_knowledge() -> None:
    """Config requires knowledge dependency."""
    with pytest.raises(ValidationError, match="knowledge"):
        ContentConfig(urls=["https://example.com"])


def test_config_validation_requires_urls_or_text() -> None:
    """Config requires either url or text_content."""
    knowledge_dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name="test-knowledge",
    )

    with pytest.raises(ValidationError, match="Either url or text_content must be provided"):
        ContentConfig(knowledge=knowledge_dep)


def test_config_validation_rejects_both_urls_and_text() -> None:
    """Config rejects both url and text_content together."""
    knowledge_dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name="test-knowledge",
    )

    with pytest.raises(ValidationError, match="Exactly one of url or text_content must be provided"):
        ContentConfig(knowledge=knowledge_dep, url="https://example.com", text_content="some text")


def test_config_validation_accepts_urls_only() -> None:
    """Config accepts URL as sole content source."""
    knowledge_dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name="test-knowledge",
    )

    config = ContentConfig(knowledge=knowledge_dep, url="https://example.com/docs")

    assert config.url == "https://example.com/docs"
    assert config.text_content is None


def test_config_validation_accepts_text_only() -> None:
    """Config accepts text_content as sole content source."""
    knowledge_dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name="test-knowledge",
    )

    config = ContentConfig(knowledge=knowledge_dep, text_content="This is the content")

    assert config.text_content == "This is the content"
    assert config.url is None


def test_config_with_urls_and_metadata() -> None:
    """Config accepts URL with optional metadata fields."""
    knowledge_dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name="test-knowledge",
    )

    config = ContentConfig(
        knowledge=knowledge_dep,
        url="https://example.com/docs",
        name="example-docs",
        description="Example documentation",
        metadata={"source": "web"},
        topics=["docs", "api"],
    )

    assert config.url == "https://example.com/docs"
    assert config.name == "example-docs"
    assert config.description == "Example documentation"
    assert config.metadata == {"source": "web"}
    assert config.topics == ["docs", "api"]


def test_outputs_contain_expected_fields() -> None:
    """Outputs contain spec and pip_dependencies."""
    knowledge_spec = KnowledgeSpec(
        name="test-knowledge",
        max_results=10,
        vector_db_spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
        ),
    )

    outputs = ContentOutputs(
        spec=ContentSpec(
            name="test-content",
            knowledge_spec=knowledge_spec,
            url="https://example.com",
        ),
        pip_dependencies=READER_DEPENDENCIES,
    )

    assert outputs.spec.name == "test-content"
    assert outputs.spec.url == "https://example.com"
    assert outputs.pip_dependencies == READER_DEPENDENCIES


def test_outputs_are_serializable() -> None:
    """Outputs can be serialized to JSON."""
    knowledge_spec = KnowledgeSpec(
        name="test-knowledge",
        max_results=10,
        vector_db_spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
        ),
    )

    outputs = ContentOutputs(
        spec=ContentSpec(
            name="test-content",
            knowledge_spec=knowledge_spec,
            text_content="Some text content",
        ),
        pip_dependencies=READER_DEPENDENCIES,
    )

    serialized = outputs.model_dump_json()

    assert "spec" in serialized
    assert "test-content" in serialized
    assert "pip_dependencies" in serialized


def test_spec_model_dump_for_ainsert() -> None:
    """spec.model_dump returns kwargs for Knowledge.ainsert()."""
    knowledge_spec = KnowledgeSpec(
        name="test-knowledge",
        max_results=10,
        vector_db_spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
        ),
    )

    spec = ContentSpec(
        name="test",
        knowledge_spec=knowledge_spec,
        url="https://example.com",
    )

    kwargs = spec.model_dump(exclude_none=True, exclude={"knowledge_spec"})

    assert kwargs["url"] == "https://example.com"
    assert kwargs["name"] == "test"
    assert "knowledge_spec" not in kwargs


def test_spec_model_dump_excludes_none_fields() -> None:
    """spec.model_dump excludes None fields for clean ainsert kwargs."""
    knowledge_spec = KnowledgeSpec(
        name="test-knowledge",
        max_results=10,
        vector_db_spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
        ),
    )

    spec = ContentSpec(
        name="test",
        knowledge_spec=knowledge_spec,
        url="https://example.com",
    )

    kwargs = spec.model_dump(exclude_none=True, exclude={"knowledge_spec"})

    assert "url" in kwargs
    assert "description" not in kwargs
    assert "metadata" not in kwargs
    assert "topics" not in kwargs


def test_reader_dependencies_defined() -> None:
    """READER_DEPENDENCIES is defined in content module."""
    assert "pypdf" in READER_DEPENDENCIES
    assert "beautifulsoup4" in READER_DEPENDENCIES
    assert "python-docx" in READER_DEPENDENCIES
    assert "youtube-transcript-api" in READER_DEPENDENCIES


async def test_lifecycle_on_create_with_urls(mocker: MockerFixture) -> None:
    """on_create returns outputs and inserts content into vectordb."""
    resource = create_content(
        name="docs-content",
        url="https://example.com/docs",
    )

    mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    mock_knowledge = mocker.MagicMock()
    mock_knowledge.ainsert = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "agno_provider.resources.knowledge.knowledge.Knowledge.from_spec",
        return_value=mock_knowledge,
    )

    result = await resource.on_create()

    assert isinstance(result, ContentOutputs)
    assert result.spec.name == "docs-content"
    assert result.spec.url == "https://example.com/docs"
    assert result.pip_dependencies == READER_DEPENDENCIES

    mock_knowledge.ainsert.assert_called_once()
    call_kwargs = mock_knowledge.ainsert.call_args.kwargs
    assert call_kwargs["url"] == "https://example.com/docs"


async def test_lifecycle_on_create_with_text(mocker: MockerFixture) -> None:
    """on_create returns outputs for text content."""
    resource = create_content(
        name="text-content",
        text_content="This is inline text",
    )

    mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    mock_knowledge = mocker.MagicMock()
    mock_knowledge.ainsert = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "agno_provider.resources.knowledge.knowledge.Knowledge.from_spec",
        return_value=mock_knowledge,
    )

    result = await resource.on_create()

    assert isinstance(result, ContentOutputs)
    assert result.spec.name == "text-content"
    assert result.spec.text_content == "This is inline text"

    mock_knowledge.ainsert.assert_called_once()
    call_kwargs = mock_knowledge.ainsert.call_args.kwargs
    assert call_kwargs["text_content"] == "This is inline text"


async def test_lifecycle_on_update_returns_outputs(mocker: MockerFixture) -> None:
    """on_update returns updated outputs with upsert."""
    resource = create_content(
        name="updated-content",
        url="https://new-example.com",
    )

    mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    mock_knowledge = mocker.MagicMock()
    mock_knowledge.ainsert = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "agno_provider.resources.knowledge.knowledge.Knowledge.from_spec",
        return_value=mock_knowledge,
    )

    knowledge_dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name="test-knowledge",
    )
    previous_config = ContentConfig(knowledge=knowledge_dep, url="https://old-example.com")

    result = await resource.on_update(previous_config)

    assert isinstance(result, ContentOutputs)
    assert result.spec.name == "updated-content"
    assert result.spec.url == "https://new-example.com"

    mock_knowledge.ainsert.assert_called_once()
    call_kwargs = mock_knowledge.ainsert.call_args.kwargs
    assert call_kwargs["url"] == "https://new-example.com"
    assert call_kwargs["upsert"] is True


async def test_lifecycle_on_delete_removes_vectors(mocker: MockerFixture) -> None:
    """on_delete removes content from vectordb by name."""
    resource = create_content(name="content-to-delete", url="https://example.com")

    mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    mock_knowledge = mocker.MagicMock()
    mock_knowledge.remove_vectors_by_name = mocker.MagicMock()
    mocker.patch(
        "agno_provider.resources.knowledge.knowledge.Knowledge.from_spec",
        return_value=mock_knowledge,
    )

    await resource.on_delete()

    mock_knowledge.remove_vectors_by_name.assert_called_once_with("content-to-delete")


async def test_lifecycle_on_delete_uses_config_name(mocker: MockerFixture) -> None:
    """on_delete uses config.name if set, otherwise resource name."""
    resource = create_content(
        name="resource-name",
        content_name="custom-content-name",
        url="https://example.com",
    )

    mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    mock_knowledge = mocker.MagicMock()
    mock_knowledge.remove_vectors_by_name = mocker.MagicMock()
    mocker.patch(
        "agno_provider.resources.knowledge.knowledge.Knowledge.from_spec",
        return_value=mock_knowledge,
    )

    await resource.on_delete()

    mock_knowledge.remove_vectors_by_name.assert_called_once_with("custom-content-name")


async def test_harness_create_with_urls(harness: ProviderHarness, mocker: MockerFixture) -> None:
    """on_create via harness returns correct outputs for URLs."""
    knowledge = create_knowledge_resource()

    knowledge_dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name=knowledge.name,
    )
    knowledge_dep._resolved = knowledge

    mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    mock_knowledge = mocker.MagicMock()
    mock_knowledge.ainsert = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "agno_provider.resources.knowledge.knowledge.Knowledge.from_spec",
        return_value=mock_knowledge,
    )

    config = ContentConfig(knowledge=knowledge_dep, url="https://example.com/docs")

    result = await harness.invoke_create(Content, name="test-content", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.name == "test-content"
    assert result.outputs.spec.url == "https://example.com/docs"


async def test_harness_create_with_text(harness: ProviderHarness, mocker: MockerFixture) -> None:
    """on_create via harness returns correct outputs for text."""
    knowledge = create_knowledge_resource()

    knowledge_dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name=knowledge.name,
    )
    knowledge_dep._resolved = knowledge

    mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    mock_knowledge = mocker.MagicMock()
    mock_knowledge.ainsert = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "agno_provider.resources.knowledge.knowledge.Knowledge.from_spec",
        return_value=mock_knowledge,
    )

    config = ContentConfig(knowledge=knowledge_dep, text_content="Inline content here")

    result = await harness.invoke_create(Content, name="text-content", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.name == "text-content"
    assert result.outputs.spec.text_content == "Inline content here"


async def test_harness_update_returns_outputs(harness: ProviderHarness, mocker: MockerFixture) -> None:
    """on_update via harness returns updated outputs."""
    knowledge = create_knowledge_resource()
    knowledge_spec = knowledge.outputs.spec

    knowledge_dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name=knowledge.name,
    )
    knowledge_dep._resolved = knowledge

    mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    mock_knowledge = mocker.MagicMock()
    mock_knowledge.ainsert = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "agno_provider.resources.knowledge.knowledge.Knowledge.from_spec",
        return_value=mock_knowledge,
    )

    previous = ContentConfig(knowledge=knowledge_dep, url="https://old.example.com")
    current = ContentConfig(knowledge=knowledge_dep, url="https://new.example.com")
    current_outputs = ContentOutputs(
        spec=ContentSpec(
            name="test-content",
            knowledge_spec=knowledge_spec,
            url="https://old.example.com",
        ),
        pip_dependencies=READER_DEPENDENCIES,
    )

    result = await harness.invoke_update(
        Content,
        name="test-content",
        config=current,
        previous_config=previous,
        current_outputs=current_outputs,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.name == "test-content"
    assert result.outputs.spec.url == "https://new.example.com"


async def test_harness_delete_success(harness: ProviderHarness, mocker: MockerFixture) -> None:
    """on_delete via harness removes vectors by name."""
    knowledge = create_knowledge_resource()

    knowledge_dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name=knowledge.name,
    )
    knowledge_dep._resolved = knowledge

    mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    mock_knowledge = mocker.MagicMock()
    mock_knowledge.remove_vectors_by_name = mocker.MagicMock()
    mocker.patch(
        "agno_provider.resources.knowledge.knowledge.Knowledge.from_spec",
        return_value=mock_knowledge,
    )

    config = ContentConfig(knowledge=knowledge_dep, url="https://example.com")

    result = await harness.invoke_delete(Content, name="test-content", config=config)

    assert result.success
    mock_knowledge.remove_vectors_by_name.assert_called_once_with("test-content")
