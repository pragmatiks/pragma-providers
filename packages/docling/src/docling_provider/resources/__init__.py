"""Resource definitions for docling provider.

Import and export your Resource classes here for discovery by the runtime.
"""

from docling_provider.resources.parser import (
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

__all__ = [
    "Chunk",
    "ChunkInput",
    "ChunkOutput",
    "DocumentMetadata",
    "ParseInput",
    "ParseOutput",
    "Parser",
    "ParserConfig",
    "ParserOutputs",
    "chunk_text",
    "parse_document",
]
