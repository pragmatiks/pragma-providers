"""Pytest configuration for qdrant provider tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pragma_sdk.provider import ProviderHarness
from qdrant_client.http import models

if TYPE_CHECKING:
    from pytest_mock import MockerFixture, MockType


@pytest.fixture
def harness() -> ProviderHarness:
    """Test harness for invoking lifecycle methods."""
    return ProviderHarness()


@pytest.fixture
def mock_qdrant_client(mocker: "MockerFixture") -> "MockType":
    """Mock AsyncQdrantClient with async context manager support."""
    mock_client = mocker.MagicMock()

    # Mock collection info response
    mock_info = mocker.MagicMock()
    mock_info.vectors_count = 100
    mock_info.points_count = 100
    mock_info.status = models.CollectionStatus.GREEN

    # Mock async methods
    mock_client.collection_exists = mocker.AsyncMock(return_value=False)
    mock_client.create_collection = mocker.AsyncMock(return_value=True)
    mock_client.get_collection = mocker.AsyncMock(return_value=mock_info)
    mock_client.delete_collection = mocker.AsyncMock(return_value=True)

    # Support async context manager protocol
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch(
        "qdrant_provider.resources.collection.AsyncQdrantClient",
        return_value=mock_client,
    )

    return mock_client
