"""Docling provider for Pragmatiks.

Provides document parsing and chunking capabilities for RAG pipelines
using IBM's Docling library.
"""

from pragma_sdk import Provider

from docling_provider.resources import (
    Chunk,
    ChunkInput,
    ChunkOutput,
    DocumentMetadata,
    ParseInput,
    ParseOutput,
    Parser,
    ParserConfig,
    ParserOutputs,
    chunk_text,
    parse_document,
)

docling = Provider(name="docling")

# Register resources
docling.resource("parser")(Parser)

__all__ = [
    "Chunk",
    "ChunkInput",
    "ChunkOutput",
    "docling",
    "DocumentMetadata",
    "ParseInput",
    "ParseOutput",
    "Parser",
    "ParserConfig",
    "ParserOutputs",
    "chunk_text",
    "parse_document",
]
