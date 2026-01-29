"""Tests for Qdrant Collection resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pragma_sdk.provider import ProviderHarness
from qdrant_client.http import models

from qdrant_provider import Collection, CollectionConfig, CollectionOutputs, VectorConfig

if TYPE_CHECKING:
    from pytest_mock import MockerFixture, MockType


async def test_create_collection_success(
    harness: ProviderHarness,
    mock_qdrant_client: "MockType",
) -> None:
    """on_create creates collection and returns outputs."""
    config = CollectionConfig(
        url="http://localhost:6333",
        name="test-collection",
        vectors=VectorConfig(size=1536, distance="Cosine"),
    )

    result = await harness.invoke_create(Collection, name="test-collection", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.name == "test-collection"
    assert result.outputs.indexed_vectors_count == 100
    assert result.outputs.points_count == 100
    assert result.outputs.status == "green"

    mock_qdrant_client.create_collection.assert_called_once()


async def test_create_collection_already_exists(
    harness: ProviderHarness,
    mock_qdrant_client: "MockType",
) -> None:
    """on_create is idempotent when collection exists."""
    mock_qdrant_client.collection_exists.return_value = True

    config = CollectionConfig(
        url="http://localhost:6333",
        name="existing-collection",
        vectors=VectorConfig(size=1536, distance="Cosine"),
    )

    result = await harness.invoke_create(Collection, name="test-collection", config=config)

    assert result.success
    mock_qdrant_client.create_collection.assert_not_called()
    mock_qdrant_client.get_collection.assert_called_once()


async def test_create_with_api_key(
    harness: ProviderHarness,
    mocker: "MockerFixture",
) -> None:
    """on_create passes api_key to client."""
    mock_client = mocker.MagicMock()
    mock_info = mocker.MagicMock()
    mock_info.indexed_vectors_count = 0
    mock_info.points_count = 0
    mock_info.status = models.CollectionStatus.GREEN

    mock_client.collection_exists = mocker.AsyncMock(return_value=False)
    mock_client.create_collection = mocker.AsyncMock(return_value=True)
    mock_client.get_collection = mocker.AsyncMock(return_value=mock_info)
    mock_client.close = mocker.AsyncMock(return_value=None)
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mock_constructor = mocker.patch(
        "qdrant_provider.resources.collection.AsyncQdrantClient",
        return_value=mock_client,
    )

    config = CollectionConfig(
        api_key="test-api-key",
        url="https://xyz.qdrant.io:6333",
        name="cloud-collection",
        vectors=VectorConfig(size=1536, distance="Cosine"),
    )

    result = await harness.invoke_create(Collection, name="test-collection", config=config)

    assert result.success
    mock_constructor.assert_called_once_with(
        url="https://xyz.qdrant.io:6333",
        api_key="test-api-key",
    )


async def test_create_with_on_disk(
    harness: ProviderHarness,
    mock_qdrant_client: "MockType",
) -> None:
    """on_create passes on_disk parameter to collection config."""
    config = CollectionConfig(
        url="http://localhost:6333",
        name="large-collection",
        vectors=VectorConfig(size=1536, distance="Cosine"),
        on_disk=True,
    )

    result = await harness.invoke_create(Collection, name="test-collection", config=config)

    assert result.success
    call_kwargs = mock_qdrant_client.create_collection.call_args.kwargs
    assert call_kwargs["vectors_config"].on_disk is True


async def test_create_with_euclid_distance(
    harness: ProviderHarness,
    mock_qdrant_client: "MockType",
) -> None:
    """on_create uses correct distance metric."""
    config = CollectionConfig(
        url="http://localhost:6333",
        name="euclid-collection",
        vectors=VectorConfig(size=768, distance="Euclid"),
    )

    result = await harness.invoke_create(Collection, name="test-collection", config=config)

    assert result.success
    call_kwargs = mock_qdrant_client.create_collection.call_args.kwargs
    assert call_kwargs["vectors_config"].distance == models.Distance.EUCLID
    assert call_kwargs["vectors_config"].size == 768


async def test_update_no_change(
    harness: ProviderHarness,
    mock_qdrant_client: "MockType",
) -> None:
    """on_update returns info without recreation when config unchanged."""
    config = CollectionConfig(
        url="http://localhost:6333",
        name="test-collection",
        vectors=VectorConfig(size=1536, distance="Cosine"),
    )
    existing_outputs = CollectionOutputs(
        name="test-collection",
        indexed_vectors_count=50,
        points_count=50,
        status="green",
    )

    result = await harness.invoke_update(
        Collection,
        name="test-collection",
        config=config,
        previous_config=config,
        current_outputs=existing_outputs,
    )

    assert result.success
    mock_qdrant_client.delete_collection.assert_not_called()
    mock_qdrant_client.create_collection.assert_not_called()


async def test_update_recreates_on_vector_size_change(
    harness: ProviderHarness,
    mock_qdrant_client: "MockType",
) -> None:
    """on_update recreates collection when vector size changes."""
    mock_qdrant_client.collection_exists.return_value = True

    previous = CollectionConfig(
        url="http://localhost:6333",
        name="test-collection",
        vectors=VectorConfig(size=1536, distance="Cosine"),
    )
    current = CollectionConfig(
        url="http://localhost:6333",
        name="test-collection",
        vectors=VectorConfig(size=768, distance="Cosine"),
    )

    result = await harness.invoke_update(
        Collection,
        name="test-collection",
        config=current,
        previous_config=previous,
        current_outputs=CollectionOutputs(
            name="test-collection",
            indexed_vectors_count=100,
            points_count=100,
            status="green",
        ),
    )

    assert result.success
    mock_qdrant_client.delete_collection.assert_called_once_with("test-collection")
    mock_qdrant_client.create_collection.assert_called_once()


async def test_update_recreates_on_distance_change(
    harness: ProviderHarness,
    mock_qdrant_client: "MockType",
) -> None:
    """on_update recreates collection when distance metric changes."""
    mock_qdrant_client.collection_exists.return_value = True

    previous = CollectionConfig(
        url="http://localhost:6333",
        name="test-collection",
        vectors=VectorConfig(size=1536, distance="Cosine"),
    )
    current = CollectionConfig(
        url="http://localhost:6333",
        name="test-collection",
        vectors=VectorConfig(size=1536, distance="Dot"),
    )

    result = await harness.invoke_update(
        Collection,
        name="test-collection",
        config=current,
        previous_config=previous,
        current_outputs=CollectionOutputs(
            name="test-collection",
            indexed_vectors_count=100,
            points_count=100,
            status="green",
        ),
    )

    assert result.success
    mock_qdrant_client.delete_collection.assert_called_once()
    call_kwargs = mock_qdrant_client.create_collection.call_args.kwargs
    assert call_kwargs["vectors_config"].distance == models.Distance.DOT


async def test_update_rejects_name_change(
    harness: ProviderHarness,
    mock_qdrant_client: "MockType",
) -> None:
    """on_update raises error when collection name changes."""
    previous = CollectionConfig(
        url="http://localhost:6333",
        name="old-collection",
        vectors=VectorConfig(size=1536, distance="Cosine"),
    )
    current = CollectionConfig(
        url="http://localhost:6333",
        name="new-collection",
        vectors=VectorConfig(size=1536, distance="Cosine"),
    )

    result = await harness.invoke_update(
        Collection,
        name="test-collection",
        config=current,
        previous_config=previous,
        current_outputs=CollectionOutputs(
            name="old-collection",
            indexed_vectors_count=100,
            points_count=100,
            status="green",
        ),
    )

    assert result.failed
    assert "Cannot change collection name" in str(result.error)


async def test_delete_success(
    harness: ProviderHarness,
    mock_qdrant_client: "MockType",
) -> None:
    """on_delete removes the collection."""
    mock_qdrant_client.collection_exists.return_value = True

    config = CollectionConfig(
        url="http://localhost:6333",
        name="test-collection",
        vectors=VectorConfig(size=1536, distance="Cosine"),
    )

    result = await harness.invoke_delete(Collection, name="test-collection", config=config)

    assert result.success
    mock_qdrant_client.delete_collection.assert_called_once_with("test-collection")


async def test_delete_idempotent(
    harness: ProviderHarness,
    mock_qdrant_client: "MockType",
) -> None:
    """on_delete succeeds if collection doesn't exist."""
    mock_qdrant_client.collection_exists.return_value = False

    config = CollectionConfig(
        url="http://localhost:6333",
        name="nonexistent-collection",
        vectors=VectorConfig(size=1536, distance="Cosine"),
    )

    result = await harness.invoke_delete(Collection, name="test-collection", config=config)

    assert result.success
    mock_qdrant_client.delete_collection.assert_not_called()


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert Collection.provider == "qdrant"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Collection.resource == "collection"
