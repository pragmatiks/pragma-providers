"""Docling Parser resource for document parsing."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, ClassVar

import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HierarchicalChunker
from pragma_sdk import Config, Outputs, Resource

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


class ParserConfig(Config):
    """Configuration for the Docling parser resource.

    Attributes:
        ocr_enabled: Enable OCR for scanned documents. Requires OCR dependencies.
        table_extraction: Enable table structure recognition and extraction.
        supported_formats: List of document formats this parser accepts.
    """

    ocr_enabled: bool = True
    table_extraction: bool = True
    supported_formats: list[str] = ["pdf", "docx", "html", "md"]


class ParserOutputs(Outputs):
    """Outputs from the Docling parser resource.

    Attributes:
        ready: Whether the parser is ready to accept documents.
        supported_formats: List of supported document formats.
    """

    ready: bool
    supported_formats: list[str]


class Parser(Resource[ParserConfig, ParserOutputs]):
    """Document parser resource using IBM Docling.

    Parses documents (PDF, DOCX, HTML, Markdown, etc.) into structured text
    with OCR and table extraction support. Use parse_document() to convert
    documents and chunk_text() for RAG chunking.

    Args:
        config: Parser configuration specifying OCR settings and supported formats.

    Returns:
        ParserOutputs with ready status and supported format list.
    """

    provider: ClassVar[str] = "docling"
    resource: ClassVar[str] = "parser"

    async def on_create(self) -> ParserOutputs:
        """Create parser resource - verify docling is available."""
        return ParserOutputs(
            ready=True,
            supported_formats=self.config.supported_formats,
        )

    async def on_update(self, previous_config: ParserConfig) -> ParserOutputs:
        """Update parser configuration."""
        return ParserOutputs(
            ready=True,
            supported_formats=self.config.supported_formats,
        )

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""


# --- Parsing utilities (for action handlers when actions are implemented) ---


class DocumentMetadata(Outputs):
    """Metadata extracted from a parsed document.

    Attributes:
        title: Document title if detected.
        author: Document author if available.
        page_count: Number of pages for paginated documents.
        format: Detected document format (pdf, docx, etc.).
    """

    title: str | None
    author: str | None
    page_count: int | None
    format: str


class Chunk(Outputs):
    """A text chunk from a parsed document.

    Attributes:
        text: The chunk text content.
        metadata: Additional metadata about the chunk (headings, page, etc.).
    """

    text: str
    metadata: dict[str, Any]


class ParseInput(Config):
    """Input for the parse action.

    Attributes:
        content: Document content - base64 encoded for binary formats, raw for text.
        filename: Original filename with extension for format detection.
        content_type: Optional MIME type override.
    """

    content: str
    filename: str
    content_type: str | None = None


class ParseOutput(Outputs):
    """Output from the parse action.

    Attributes:
        text: Full extracted text in markdown format.
        metadata: Document metadata (title, author, etc.).
        chunks: Document split into semantic chunks for RAG.
    """

    text: str
    metadata: DocumentMetadata
    chunks: list[Chunk]


class ChunkInput(Config):
    """Input for the chunk action.

    Attributes:
        text: Plain text to chunk.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.
        strategy: Chunking strategy (recursive, sentence, paragraph).
    """

    text: str
    chunk_size: int = 512
    chunk_overlap: int = 50
    strategy: str = "recursive"


class ChunkOutput(Outputs):
    """Output from the chunk action.

    Attributes:
        chunks: List of text chunks.
    """

    chunks: list[Chunk]


# Format mapping from extension to InputFormat
FORMAT_MAP: dict[str, InputFormat] = {
    "pdf": InputFormat.PDF,
    "docx": InputFormat.DOCX,
    "html": InputFormat.HTML,
    "htm": InputFormat.HTML,
    "md": InputFormat.MD,
    "markdown": InputFormat.MD,
    "pptx": InputFormat.PPTX,
    "xlsx": InputFormat.XLSX,
    "csv": InputFormat.CSV,
}


def _get_format_from_filename(filename: str) -> InputFormat | None:
    """Get InputFormat from filename extension."""
    ext = Path(filename).suffix.lower().lstrip(".")
    return FORMAT_MAP.get(ext)


def _is_binary_format(format_str: str) -> bool:
    """Check if format requires binary content (base64 encoded)."""
    return format_str.lower() in {"pdf", "docx", "pptx", "xlsx"}


def _create_converter(config: ParserConfig) -> DocumentConverter:
    """Create a configured DocumentConverter instance."""
    allowed_formats = []
    for fmt in config.supported_formats:
        if fmt.lower() in FORMAT_MAP:
            allowed_formats.append(FORMAT_MAP[fmt.lower()])

    # Configure PDF pipeline options
    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = config.ocr_enabled
    pdf_options.do_table_structure = config.table_extraction

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
    }

    return DocumentConverter(
        allowed_formats=allowed_formats or None,
        format_options=format_options,
    )


def _extract_metadata(doc: "DoclingDocument", format_str: str) -> DocumentMetadata:
    """Extract metadata from a DoclingDocument."""
    # Try to get title from document
    title = None
    if hasattr(doc, "name") and doc.name:
        title = doc.name

    # Page count for PDFs
    page_count = None
    if hasattr(doc, "pages") and doc.pages:
        page_count = len(doc.pages)

    return DocumentMetadata(
        title=title,
        author=None,  # Docling doesn't expose author metadata directly
        page_count=page_count,
        format=format_str,
    )


def _chunk_document(doc: "DoclingDocument") -> list[Chunk]:
    """Chunk a document using hierarchical chunker."""
    chunker = HierarchicalChunker()
    chunks = []

    for chunk in chunker.chunk(dl_doc=doc):
        chunk_metadata: dict[str, Any] = {}

        if hasattr(chunk, "meta") and chunk.meta:
            if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                chunk_metadata["headings"] = chunk.meta.headings
            if hasattr(chunk.meta, "doc_items"):
                chunk_metadata["doc_items_count"] = len(chunk.meta.doc_items)

        chunks.append(
            Chunk(
                text=chunk.text,
                metadata=chunk_metadata,
            )
        )

    return chunks


def _simple_chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
) -> list[Chunk]:
    """Simple text chunking without docling document structure."""
    chunks: list[Chunk] = []

    if strategy == "paragraph":
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            chunks.append(Chunk(text=para, metadata={"index": i, "strategy": "paragraph"}))
    elif strategy == "sentence":
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for i, sent in enumerate(sentences):
            if sent.strip():
                chunks.append(Chunk(text=sent.strip(), metadata={"index": i, "strategy": "sentence"}))
    else:
        # Token-based chunking using tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)

        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={"index": len(chunks), "strategy": "recursive", "token_count": len(chunk_tokens)},
                )
            )

            # Move start forward, accounting for overlap
            start = end - chunk_overlap if end < len(tokens) else end

    return chunks


def parse_document(config: ParserConfig, parse_input: ParseInput) -> ParseOutput:
    """Parse a document into structured text and chunks.

    Accepts base64-encoded content for binary formats (PDF, DOCX)
    or raw text for text formats (HTML, MD).

    This is a standalone function that can be called from action handlers
    or used directly for testing.

    Args:
        config: Parser configuration.
        parse_input: Document content and metadata.

    Returns:
        Parsed document with text, metadata, and chunks.

    Raises:
        ValueError: If format is unsupported or not in allowed formats.
    """
    # Detect format from filename
    input_format = _get_format_from_filename(parse_input.filename)
    if input_format is None:
        ext = Path(parse_input.filename).suffix.lower().lstrip(".")
        raise ValueError(f"Unsupported format: {ext}")

    format_str = Path(parse_input.filename).suffix.lower().lstrip(".")

    # Check if format is in supported formats
    if format_str not in [f.lower() for f in config.supported_formats]:
        raise ValueError(f"Format '{format_str}' not in supported formats: {config.supported_formats}")

    # Decode content
    if _is_binary_format(format_str):
        content_bytes = base64.b64decode(parse_input.content)
    else:
        content_bytes = parse_input.content.encode("utf-8")

    # Write to temp file for docling (it needs a file path)
    suffix = Path(parse_input.filename).suffix
    tmp_path: str | None = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name

        # Create converter and parse
        converter = _create_converter(config)
        result = converter.convert(tmp_path)
        doc = result.document

        # Extract text as markdown
        text = doc.export_to_markdown()

        # Extract metadata
        metadata = _extract_metadata(doc, format_str)

        # Generate chunks
        chunks = _chunk_document(doc)

        return ParseOutput(
            text=text,
            metadata=metadata,
            chunks=chunks,
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def chunk_text(chunk_input: ChunkInput) -> ChunkOutput:
    """Chunk plain text into smaller pieces for RAG.

    Supports multiple chunking strategies:
    - recursive: Token-based chunking with overlap using tiktoken (default)
    - sentence: Split on sentence boundaries
    - paragraph: Split on paragraph boundaries

    Args:
        chunk_input: Text and chunking parameters.

    Returns:
        List of text chunks.
    """
    chunks = _simple_chunk_text(
        text=chunk_input.text,
        chunk_size=chunk_input.chunk_size,
        chunk_overlap=chunk_input.chunk_overlap,
        strategy=chunk_input.strategy,
    )

    return ChunkOutput(chunks=chunks)
