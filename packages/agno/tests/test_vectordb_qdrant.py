"""Tests for Agno vectordb/qdrant resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agno.vectordb.qdrant import Qdrant, SearchType
from pragma_sdk import Dependency
from pragma_sdk.provider import ProviderHarness

from agno_provider import (
    EmbedderOpenAI,
    VectordbQdrant,
    VectordbQdrantConfig,
    VectordbQdrantOutputs,
)
from agno_provider.resources.vectordb.qdrant import VectordbQdrantSpec


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_resource_metadata_provider_name() -> None:
    """Resource has correct provider name."""
    assert VectordbQdrant.provider == "agno"


def test_resource_metadata_resource_type() -> None:
    """Resource has correct resource type."""
    assert VectordbQdrant.resource == "vectordb/qdrant"


def test_config_required_fields() -> None:
    """Config requires url and collection."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
    )

    assert config.url == "http://localhost:6333"
    assert config.collection == "test-collection"
    assert config.api_key is None
    assert config.search_type == "hybrid"
    assert config.embedder is None


def test_config_embedder_default_is_none() -> None:
    """Config embedder field defaults to None."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
    )

    assert config.embedder is None


def test_config_with_embedder_dependency() -> None:
    """Config accepts embedder dependency."""
    embedder_dep = Dependency[EmbedderOpenAI](
        provider="agno",
        resource="knowledge/embedder/openai",
        name="my-embedder",
    )
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        embedder=embedder_dep,
    )

    assert config.embedder is not None
    assert config.embedder.name == "my-embedder"


def test_config_with_api_key() -> None:
    """Config accepts optional api_key."""
    config = VectordbQdrantConfig(
        url="https://qdrant-cloud.example.com:6333",
        collection="cloud-collection",
        api_key="qdrant-api-key-123",
    )

    assert config.api_key == "qdrant-api-key-123"


def test_config_with_search_type() -> None:
    """Config accepts search_type parameter."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="hybrid",
    )

    assert config.search_type == "hybrid"


def test_outputs_are_serializable() -> None:
    """Outputs contain only serializable data."""
    outputs = VectordbQdrantOutputs(
        spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
            search_type="vector",
        ),
        pip_dependencies=[],
    )

    assert outputs.spec.url == "http://localhost:6333"
    assert outputs.spec.collection == "test-collection"
    assert outputs.spec.search_type == "vector"
    assert outputs.pip_dependencies == []

    serialized = outputs.model_dump_json()
    assert "spec" in serialized
    assert "pip_dependencies" in serialized


def test_from_spec_returns_qdrant_instance() -> None:
    """from_spec() returns configured Agno Qdrant instance."""
    spec = VectordbQdrantSpec(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="vector",
    )

    db = VectordbQdrant.from_spec(spec)

    assert isinstance(db, Qdrant)


def test_from_spec_with_api_key() -> None:
    """from_spec() passes API key when provided."""
    spec = VectordbQdrantSpec(
        url="https://qdrant-cloud.example.com:6333",
        collection="cloud-collection",
        api_key="qdrant-api-key-123",
        search_type="vector",
    )

    db = VectordbQdrant.from_spec(spec)

    assert isinstance(db, Qdrant)


def test_get_search_type_vector() -> None:
    """_get_search_type returns correct enum for vector."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="vector",
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    assert resource._get_search_type() == SearchType.vector


def test_get_search_type_keyword() -> None:
    """_get_search_type returns correct enum for keyword."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="keyword",
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    assert resource._get_search_type() == SearchType.keyword


def test_get_search_type_hybrid() -> None:
    """_get_search_type returns correct enum for hybrid."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="hybrid",
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    assert resource._get_search_type() == SearchType.hybrid


def test_pip_dependencies_vector_is_empty() -> None:
    """Vector search requires no additional dependencies."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="vector",
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    assert resource._get_pip_dependencies() == []


def test_pip_dependencies_hybrid_requires_fastembed() -> None:
    """Hybrid search requires fastembed."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="hybrid",
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    assert resource._get_pip_dependencies() == ["fastembed>=0.6.0"]


def test_pip_dependencies_keyword_requires_fastembed() -> None:
    """Keyword search requires fastembed."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="keyword",
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    assert resource._get_pip_dependencies() == ["fastembed>=0.6.0"]


async def test_lifecycle_create_returns_outputs(harness: ProviderHarness) -> None:
    """on_create returns outputs with config values."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="hybrid",
    )

    result = await harness.invoke_create(VectordbQdrant, name="test-qdrant", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.url == "http://localhost:6333"
    assert result.outputs.spec.collection == "test-collection"
    assert result.outputs.spec.search_type == "hybrid"
    assert result.outputs.pip_dependencies == ["fastembed>=0.6.0"]


async def test_lifecycle_update_returns_outputs(harness: ProviderHarness) -> None:
    """on_update returns updated outputs."""
    previous = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="old-collection",
        search_type="vector",
    )
    current = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="new-collection",
        search_type="hybrid",
    )
    current_outputs = VectordbQdrantOutputs(
        spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="old-collection",
            search_type="vector",
        ),
        pip_dependencies=[],
    )

    result = await harness.invoke_update(
        VectordbQdrant,
        name="test-qdrant",
        config=current,
        previous_config=previous,
        current_outputs=current_outputs,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.collection == "new-collection"
    assert result.outputs.spec.search_type == "hybrid"
    assert result.outputs.pip_dependencies == ["fastembed>=0.6.0"]


async def test_lifecycle_delete_success(harness: ProviderHarness) -> None:
    """on_delete completes without error (stateless resource)."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
    )

    result = await harness.invoke_delete(VectordbQdrant, name="test-qdrant", config=config)

    assert result.success


def test_from_spec_without_embedder(mocker: MockerFixture) -> None:
    """from_spec() works when embedder is None (default behavior)."""
    spec = VectordbQdrantSpec(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="vector",
    )

    mock_init = mocker.patch("agno.vectordb.qdrant.Qdrant.__init__", return_value=None)
    VectordbQdrant.from_spec(spec)

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args.kwargs
    assert "embedder" not in call_kwargs
    assert call_kwargs["collection"] == "test-collection"
    assert call_kwargs["url"] == "http://localhost:6333"
