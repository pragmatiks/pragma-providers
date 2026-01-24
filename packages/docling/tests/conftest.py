"""Pytest configuration for docling provider tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pragma_sdk.provider import ProviderHarness

if TYPE_CHECKING:
    from pytest_mock import MockerFixture, MockType


@pytest.fixture
def harness() -> ProviderHarness:
    """Test harness for invoking lifecycle methods."""
    return ProviderHarness()


@pytest.fixture
def mock_docling_converter(mocker: "MockerFixture") -> "MockType":
    """Mock DocumentConverter for testing without actual docling processing."""
    # Create mock document
    mock_doc = mocker.MagicMock()
    mock_doc.name = "test-document"
    mock_doc.pages = [mocker.MagicMock(), mocker.MagicMock()]  # 2 pages
    mock_doc.export_to_markdown.return_value = "# Test Document\n\nThis is test content."

    # Create mock result
    mock_result = mocker.MagicMock()
    mock_result.document = mock_doc

    # Create mock converter
    mock_converter = mocker.MagicMock()
    mock_converter.convert.return_value = mock_result

    # Patch DocumentConverter
    mocker.patch(
        "docling_provider.resources.parser.DocumentConverter",
        return_value=mock_converter,
    )

    return mock_converter


@pytest.fixture
def mock_hierarchical_chunker(mocker: "MockerFixture") -> "MockType":
    """Mock HierarchicalChunker for testing."""
    # Create mock chunks
    mock_chunk1 = mocker.MagicMock()
    mock_chunk1.text = "First chunk of text."
    mock_chunk1.meta = mocker.MagicMock()
    mock_chunk1.meta.headings = ["Test Document"]
    mock_chunk1.meta.doc_items = [mocker.MagicMock()]

    mock_chunk2 = mocker.MagicMock()
    mock_chunk2.text = "Second chunk of text."
    mock_chunk2.meta = mocker.MagicMock()
    mock_chunk2.meta.headings = ["Test Document"]
    mock_chunk2.meta.doc_items = [mocker.MagicMock(), mocker.MagicMock()]

    # Create mock chunker
    mock_chunker = mocker.MagicMock()
    mock_chunker.chunk.return_value = [mock_chunk1, mock_chunk2]

    mocker.patch(
        "docling_provider.resources.parser.HierarchicalChunker",
        return_value=mock_chunker,
    )

    return mock_chunker
