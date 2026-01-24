# Docling Provider for Pragmatiks

Document parsing and chunking for RAG pipelines using [IBM Docling](https://github.com/DS4SD/docling).

## Features

- Parse PDF, DOCX, HTML, Markdown, and other document formats
- OCR support for scanned documents
- Table structure extraction
- Semantic chunking for RAG pipelines
- Multiple chunking strategies (recursive, sentence, paragraph)

## Installation

```bash
pip install pragmatiks-docling-provider
```

## Usage

### Resource Definition

```yaml
apiVersion: docling/v1
kind: parser
metadata:
  name: doc-parser
  namespace: demo
spec:
  ocr_enabled: true
  table_extraction: true
  supported_formats:
    - pdf
    - docx
    - html
    - md
```

### Python SDK

```python
from docling_provider import Parser, ParserConfig, parse_document, ParseInput

# Create parser configuration
config = ParserConfig(
    ocr_enabled=True,
    table_extraction=True,
    supported_formats=["pdf", "docx", "html", "md"],
)

# Parse a document (markdown example)
parse_input = ParseInput(
    content="# Hello World\n\nThis is a test document.",
    filename="test.md",
)

result = parse_document(config, parse_input)
print(f"Extracted text: {result.text}")
print(f"Chunks: {len(result.chunks)}")
```

### Chunking Text

```python
from docling_provider import chunk_text, ChunkInput

chunk_input = ChunkInput(
    text="Your long text here...",
    chunk_size=512,
    chunk_overlap=50,
    strategy="recursive",  # or "sentence" or "paragraph"
)

result = chunk_text(chunk_input)
for chunk in result.chunks:
    print(f"Chunk: {chunk.text[:50]}...")
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ocr_enabled` | bool | true | Enable OCR for scanned documents |
| `table_extraction` | bool | true | Extract table structures |
| `supported_formats` | list | ["pdf", "docx", "html", "md"] | Allowed document formats |

## Supported Formats

- PDF (`.pdf`)
- Microsoft Word (`.docx`)
- HTML (`.html`, `.htm`)
- Markdown (`.md`)
- PowerPoint (`.pptx`)
- Excel (`.xlsx`)
- CSV (`.csv`)

## License

MIT
