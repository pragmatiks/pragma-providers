"""Tests for Agno vectordb/qdrant resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from agno.vectordb.qdrant import Qdrant, SearchType
from pragma_sdk import Dependency
from pragma_sdk.provider import ProviderHarness

from agno_provider import (
    EmbedderOpenAI,
    VectordbQdrant,
    VectordbQdrantConfig,
    VectordbQdrantOutputs,
)


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
        url="http://localhost:6333",
        collection="test-collection",
        search_type="vector",
        pip_dependencies=[],
        ready=True,
    )

    assert outputs.url == "http://localhost:6333"
    assert outputs.collection == "test-collection"
    assert outputs.search_type == "vector"
    assert outputs.pip_dependencies == []
    assert outputs.ready is True

    serialized = outputs.model_dump_json()
    assert "url" in serialized
    assert "collection" in serialized
    assert "search_type" in serialized
    assert "pip_dependencies" in serialized
    assert "ready" in serialized


def test_vectordb_method_returns_qdrant_instance() -> None:
    """vectordb() returns configured Agno Qdrant instance."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="vector",
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    db = resource.vectordb()

    assert isinstance(db, Qdrant)


def test_vectordb_method_with_api_key() -> None:
    """vectordb() passes API key when provided."""
    config = VectordbQdrantConfig(
        url="https://qdrant-cloud.example.com:6333",
        collection="cloud-collection",
        api_key="qdrant-api-key-123",
        search_type="vector",
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    db = resource.vectordb()

    assert isinstance(db, Qdrant)


def test_vectordb_method_with_hybrid_search() -> None:
    """vectordb() configures hybrid search when specified.

    Note: Hybrid search requires fastembed package which needs
    onnxruntime. Skips on Python 3.14+ where onnxruntime isn't available.
    """
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="hybrid",
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    try:
        db = resource.vectordb()
        assert isinstance(db, Qdrant)
    except ImportError as e:
        error_msg = str(e).lower()
        if "fastembed" in error_msg or "onnxruntime" in error_msg:
            pytest.skip("fastembed/onnxruntime not available - required for hybrid search")
        raise


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
    assert result.outputs.url == "http://localhost:6333"
    assert result.outputs.collection == "test-collection"
    assert result.outputs.search_type == "hybrid"
    assert result.outputs.pip_dependencies == ["fastembed>=0.6.0"]
    assert result.outputs.ready is True


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
        url="http://localhost:6333",
        collection="old-collection",
        search_type="vector",
        pip_dependencies=[],
        ready=True,
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
    assert result.outputs.collection == "new-collection"
    assert result.outputs.search_type == "hybrid"
    assert result.outputs.pip_dependencies == ["fastembed>=0.6.0"]


async def test_lifecycle_delete_success(harness: ProviderHarness) -> None:
    """on_delete completes without error (stateless resource)."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
    )

    result = await harness.invoke_delete(VectordbQdrant, name="test-qdrant", config=config)

    assert result.success


def test_vectordb_without_embedder(mocker: MockerFixture) -> None:
    """vectordb() works when embedder is None (default behavior)."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="vector",
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    mock_init = mocker.patch("agno.vectordb.qdrant.Qdrant.__init__", return_value=None)
    resource.vectordb()

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args.kwargs
    assert "embedder" not in call_kwargs
    assert call_kwargs["collection"] == "test-collection"
    assert call_kwargs["url"] == "http://localhost:6333"


def test_vectordb_with_embedder_dependency(mocker: MockerFixture) -> None:
    """vectordb() passes embedder to Qdrant when dependency is resolved."""
    mock_embedder = mocker.MagicMock(spec=["get_embedding"])
    mock_embedder_resource = mocker.MagicMock()
    mock_embedder_resource.embedder.return_value = mock_embedder

    embedder_dep = Dependency[EmbedderOpenAI](
        provider="agno",
        resource="knowledge/embedder/openai",
        name="my-embedder",
    )
    embedder_dep._resolved = mock_embedder_resource

    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="vector",
        embedder=embedder_dep,
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    mock_init = mocker.patch("agno.vectordb.qdrant.Qdrant.__init__", return_value=None)
    resource.vectordb()

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args.kwargs
    assert call_kwargs["embedder"] is mock_embedder
    assert call_kwargs["collection"] == "test-collection"
    assert call_kwargs["url"] == "http://localhost:6333"

    mock_embedder_resource.embedder.assert_called_once()


def test_vectordb_with_unresolved_embedder_dependency(mocker: MockerFixture) -> None:
    """vectordb() does not pass embedder when dependency is not resolved."""
    embedder_dep = Dependency[EmbedderOpenAI](
        provider="agno",
        resource="knowledge/embedder/openai",
        name="my-embedder",
    )

    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="vector",
        embedder=embedder_dep,
    )

    resource = VectordbQdrant(name="test-qdrant", config=config)

    mock_init = mocker.patch("agno.vectordb.qdrant.Qdrant.__init__", return_value=None)
    resource.vectordb()

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args.kwargs
    assert "embedder" not in call_kwargs
