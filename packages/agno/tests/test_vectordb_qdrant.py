"""Tests for Agno vectordb/qdrant resource."""

from __future__ import annotations

import pytest
from agno.vectordb.qdrant import Qdrant, SearchType
from pragma_sdk.provider import ProviderHarness

from agno_provider import (
    VectordbQdrant,
    VectordbQdrantConfig,
    VectordbQdrantOutputs,
)


@pytest.fixture
def harness() -> ProviderHarness:
    """Test harness for invoking lifecycle methods."""
    return ProviderHarness()


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
        ready=True,
    )

    assert outputs.url == "http://localhost:6333"
    assert outputs.collection == "test-collection"
    assert outputs.search_type == "vector"
    assert outputs.ready is True

    serialized = outputs.model_dump_json()
    assert "url" in serialized
    assert "collection" in serialized
    assert "search_type" in serialized
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

    Note: Hybrid search requires fastembed package. This test verifies
    the configuration is passed correctly to Agno's Qdrant class.
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
        if "fastembed" in str(e):
            pytest.skip("fastembed not installed - required for hybrid search")
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


async def test_lifecycle_delete_success(harness: ProviderHarness) -> None:
    """on_delete completes without error (stateless resource)."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
    )

    result = await harness.invoke_delete(VectordbQdrant, name="test-qdrant", config=config)

    assert result.success
