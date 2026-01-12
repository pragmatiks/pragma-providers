# Pragma Providers

Pragma-managed cloud providers for the Pragmatiks platform.

## Available Providers

| Provider | Package | Resources |
|----------|---------|-----------|
| GCP | `pragma-gcp-provider` | Secret Manager |

## Installation

```bash
pip install pragma-gcp-provider
```

## Usage

Resources are typically managed through the Pragma platform. You can reference them in your own providers using `FieldReference`:

```python
from pragma_sdk import FieldReference

config = MyAppConfig(
    database_password=FieldReference(
        provider="gcp",
        resource="secret",
        name="db-password",
        field="data"
    )
)
```

## Development

```bash
# Install dependencies
task install

# Run all tests
task test

# Run all checks
task check

# Run GCP-specific tasks
task gcp:test
task gcp:check
```

## License

MIT
