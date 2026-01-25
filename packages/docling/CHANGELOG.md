# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-24

### Added

- Initial release of the Docling provider
- Parser resource for document parsing capabilities
- Support for PDF, DOCX, HTML, Markdown, and other formats
- OCR support for scanned documents
- Table structure extraction
- Hierarchical chunking using Docling's chunker
- Simple text chunking with recursive, sentence, and paragraph strategies
- `parse_document` function for document parsing
- `chunk_text` function for text chunking

## docling-v0.2.0 (2026-01-25)

### Feat

- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)
- **gcp**: improve module docstring with resource description
- **gcp**: require credentials from pragma/secret for multi-tenant auth
- add PyPI publishing and rename to pragmatiks-gcp-provider
- **gcp**: add GCP provider with Secret Manager resource

### Fix

- **ci**: use GitHub App token for push to bypass branch protection
- **deps**: update pragmatiks-sdk to v0.6.0
- **ci**: correct dist paths for workspace builds
- **ci**: collect built packages to central dist directory
- **gcp**: update README with actual Secret resource documentation
- **ci**: per-provider versioning with commitizen and change detection
- **deps**: update pragmatiks-sdk to v0.5.0
- **deps**: update pragmatiks-sdk to v0.4.0
- use default dist/ directory for PyPI publish
- **deps**: update pragmatiks-sdk to v0.3.1
- add pypi environment for trusted publisher
- add module-name for uv_build to find gcp_provider
- **deps**: update pragmatiks-sdk to v0.2.1
- **deps**: update pragmatiks-sdk to v0.1.3
- **ci**: pull before push to avoid race conditions
- format test file
- **deps**: update pragmatiks-sdk to v0.1.2
- **ci**: add ruff to dev dependencies

### Refactor

- **gcp**: use native async client for Secret Manager
