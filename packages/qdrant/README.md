# Qdrant Provider

Qdrant vector database provider for Pragmatiks.

## Resources

### collection

Manages Qdrant vector collections.

```yaml
apiVersion: qdrant/v1
kind: collection
metadata:
  name: company-docs
  namespace: demo
spec:
  api_key: $ref{qdrant-secret.data.api_key}  # Optional for local
  url: https://xyz.qdrant.io:6333  # Or http://localhost:6333
  name: company-docs
  vectors:
    size: 1536
    distance: Cosine  # Cosine, Euclid, Dot
  on_disk: true  # Optional
```

## Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `api_key` | `Field[str]` | No | API key for Qdrant Cloud (optional for local) |
| `url` | `str` | No | Qdrant server URL (default: `http://localhost:6333`) |
| `name` | `str` | Yes | Collection name |
| `vectors` | `VectorConfig` | Yes | Vector configuration |
| `on_disk` | `bool` | No | Store vectors on disk (default: `false`) |

### VectorConfig

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `size` | `int` | Yes | Vector dimension |
| `distance` | `str` | No | Distance metric: `Cosine`, `Euclid`, `Dot` (default: `Cosine`) |

## Outputs

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Collection name |
| `vectors_count` | `int` | Number of indexed vectors |
| `points_count` | `int` | Number of points in collection |
| `status` | `str` | Collection status (`green`, `yellow`, `red`) |

## Lifecycle

- **on_create**: Creates collection if it doesn't exist
- **on_update**: Recreates collection if vector config changes (destructive)
- **on_delete**: Deletes collection

## Installation

```bash
pip install pragmatiks-qdrant-provider
```

## Development

```bash
# Run tests
task qdrant:test

# Format code
task qdrant:format

# Lint and type check
task qdrant:check
```
