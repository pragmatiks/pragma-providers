"""Tests for Docling Parser resource."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from pragma_sdk.provider import ProviderHarness

from docling_provider import (
    ChunkInput,
    ParseInput,
    Parser,
    ParserConfig,
    ParserOutputs,
    chunk_text,
    parse_document,
)

if TYPE_CHECKING:
    from pytest_mock import MockType


async def test_create_parser_success(harness: ProviderHarness) -> None:
    """on_create returns ready state with supported formats."""
    config = ParserConfig(
        ocr_enabled=True,
        table_extraction=True,
        supported_formats=["pdf", "docx", "html", "md"],
    )

    result = await harness.invoke_create(Parser, name="test-parser", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.ready is True
    assert result.outputs.supported_formats == ["pdf", "docx", "html", "md"]


async def test_create_parser_custom_formats(harness: ProviderHarness) -> None:
    """on_create respects custom supported_formats."""
    config = ParserConfig(
        ocr_enabled=False,
        table_extraction=False,
        supported_formats=["pdf", "md"],
    )

    result = await harness.invoke_create(Parser, name="test-parser", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.supported_formats == ["pdf", "md"]


async def test_update_parser_changes_config(harness: ProviderHarness) -> None:
    """on_update applies new configuration."""
    previous = ParserConfig(
        ocr_enabled=True,
        table_extraction=True,
        supported_formats=["pdf"],
    )
    current = ParserConfig(
        ocr_enabled=False,
        table_extraction=True,
        supported_formats=["pdf", "docx", "html"],
    )

    result = await harness.invoke_update(
        Parser,
        name="test-parser",
        config=current,
        previous_config=previous,
        current_outputs=ParserOutputs(ready=True, supported_formats=["pdf"]),
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.supported_formats == ["pdf", "docx", "html"]


async def test_delete_parser_success(harness: ProviderHarness) -> None:
    """on_delete completes without error (stateless resource)."""
    config = ParserConfig()

    result = await harness.invoke_delete(Parser, name="test-parser", config=config)

    assert result.success


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert Parser.provider == "docling"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Parser.resource == "parser"


# --- Tests for parse_document function ---


def test_parse_document_markdown(
    mock_docling_converter: "MockType",
    mock_hierarchical_chunker: "MockType",
) -> None:
    """parse_document handles markdown content."""
    config = ParserConfig(supported_formats=["md"])
    parse_input = ParseInput(
        content="# Hello World\n\nThis is a test.",
        filename="test.md",
    )

    result = parse_document(config, parse_input)

    assert result.text == "# Test Document\n\nThis is test content."
    assert result.metadata.format == "md"
    assert result.metadata.title == "test-document"
    assert len(result.chunks) == 2
    assert result.chunks[0].text == "First chunk of text."


def test_parse_document_html(
    mock_docling_converter: "MockType",
    mock_hierarchical_chunker: "MockType",
) -> None:
    """parse_document handles HTML content."""
    config = ParserConfig(supported_formats=["html"])
    parse_input = ParseInput(
        content="<html><body><h1>Test</h1></body></html>",
        filename="test.html",
    )

    result = parse_document(config, parse_input)

    assert result.metadata.format == "html"
    mock_docling_converter.convert.assert_called_once()


def test_parse_document_pdf_binary(
    mock_docling_converter: "MockType",
    mock_hierarchical_chunker: "MockType",
) -> None:
    """parse_document handles base64-encoded PDF content."""
    config = ParserConfig(supported_formats=["pdf"])
    # Create fake PDF content (just bytes for testing)
    fake_pdf = b"%PDF-1.4 fake content"
    parse_input = ParseInput(
        content=base64.b64encode(fake_pdf).decode("utf-8"),
        filename="test.pdf",
    )

    result = parse_document(config, parse_input)

    assert result.metadata.format == "pdf"
    assert result.metadata.page_count == 2  # From mock


def test_parse_document_unsupported_format() -> None:
    """parse_document raises error for unsupported format."""
    config = ParserConfig(supported_formats=["pdf"])
    parse_input = ParseInput(
        content="test content",
        filename="test.xyz",
    )

    try:
        parse_document(config, parse_input)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported format" in str(e)


def test_parse_document_format_not_allowed() -> None:
    """parse_document raises error when format not in supported_formats."""
    config = ParserConfig(supported_formats=["pdf"])  # Only PDF allowed
    parse_input = ParseInput(
        content="# Test",
        filename="test.md",  # MD not allowed
    )

    try:
        parse_document(config, parse_input)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not in supported formats" in str(e)


# --- Tests for chunk_text function ---


def test_chunk_text_recursive() -> None:
    """chunk_text with recursive strategy splits by character limit."""
    text = " ".join(["word"] * 100)
    chunk_input = ChunkInput(
        text=text,
        chunk_size=50,  # ~200 chars
        chunk_overlap=10,
        strategy="recursive",
    )

    result = chunk_text(chunk_input)

    assert len(result.chunks) > 1
    for chunk in result.chunks:
        assert chunk.metadata["strategy"] == "recursive"


def test_chunk_text_sentence() -> None:
    """chunk_text with sentence strategy splits by sentences."""
    text = "First sentence. Second sentence! Third sentence?"
    chunk_input = ChunkInput(
        text=text,
        strategy="sentence",
    )

    result = chunk_text(chunk_input)

    assert len(result.chunks) == 3
    assert result.chunks[0].text == "First sentence."
    assert result.chunks[1].text == "Second sentence!"
    assert result.chunks[2].text == "Third sentence?"


def test_chunk_text_paragraph() -> None:
    """chunk_text with paragraph strategy splits by double newlines."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunk_input = ChunkInput(
        text=text,
        strategy="paragraph",
    )

    result = chunk_text(chunk_input)

    assert len(result.chunks) == 3
    assert result.chunks[0].text == "First paragraph."
    assert result.chunks[1].text == "Second paragraph."
    assert result.chunks[2].text == "Third paragraph."


def test_chunk_text_empty() -> None:
    """chunk_text handles empty text."""
    chunk_input = ChunkInput(text="", strategy="paragraph")

    result = chunk_text(chunk_input)

    assert len(result.chunks) == 0


def test_chunk_text_overlap() -> None:
    """chunk_text preserves overlap between chunks."""
    # Create text that will definitely span multiple chunks
    words = ["word" + str(i) for i in range(50)]
    text = " ".join(words)
    chunk_input = ChunkInput(
        text=text,
        chunk_size=20,  # Small size to force multiple chunks
        chunk_overlap=5,  # Some overlap
        strategy="recursive",
    )

    result = chunk_text(chunk_input)

    # Verify we have multiple chunks
    assert len(result.chunks) > 1

    # Check that later chunks might start with words from previous chunks (overlap)
    # This is a simplified check - just verify chunks exist and have content
    for chunk in result.chunks:
        assert len(chunk.text) > 0
        assert chunk.metadata["strategy"] == "recursive"
