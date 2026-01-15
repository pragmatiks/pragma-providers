<p align="center">
  <img src="assets/wordmark.png" alt="Pragma-OS" width="800">
</p>

# Pragma Providers

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pragmatiks/pragma-providers)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**[Documentation](https://docs.pragmatiks.io/providers/overview)** | **[SDK](https://github.com/pragmatiks/pragma-sdk)** | **[CLI](https://github.com/pragmatiks/pragma-cli)**

Resource providers for the pragma-os platform.

## Quick Start

```yaml
# secret.yaml
provider: gcp
resource: secret
name: db-password
config:
  secret_id: db-password
  data: "super-secret-value"
```

```bash
pragma resources apply --pending secret.yaml
pragma resources get gcp/secret db-password
```

## Available Providers

### GCP Provider

Manage Google Cloud Platform resources.

```bash
pip install pragmatiks-gcp-provider
```

| Resource | Description |
|----------|-------------|
| `gcp/secret` | Secret Manager secrets |

## Using Provider Resources

Reference provider resources in your configurations:

```python
from pragma_sdk import FieldReference

config = AppConfig(
    database_password=FieldReference(
        provider="gcp",
        resource="secret",
        name="db-password",
        field="data"
    )
)
```

Or via YAML with dependency references:

```yaml
provider: myapp
resource: service
name: api
config:
  db_password:
    $ref:
      provider: gcp
      resource: secret
      name: db-password
      field: data
```

## Building Custom Providers

Create your own providers with the SDK:

```bash
# Initialize a provider project
pragma provider init mycompany

# Implement your resources
cd mycompany-provider
# Edit src/mycompany_provider/resources/

# Deploy to the platform
pragma provider sync
pragma provider push --deploy
```

See the [Building Providers Guide](https://docs.pragmatiks.io/building-providers/overview) for complete documentation.

## Provider Architecture

Each provider contains:

- **Provider namespace** - Groups related resources (e.g., `gcp`)
- **Resource types** - Individual resource definitions with Config and Outputs
- **Lifecycle methods** - `on_create`, `on_update`, `on_delete` implementations

```python
from pragma_sdk import Provider, Resource, Config, Outputs

gcp = Provider(name="gcp")

class SecretConfig(Config):
    secret_id: str
    data: str

class SecretOutputs(Outputs):
    resource_name: str
    version_id: str

@gcp.resource("secret")
class Secret(Resource[SecretConfig, SecretOutputs]):
    async def on_create(self) -> SecretOutputs:
        # Create secret in GCP Secret Manager
        ...

    async def on_update(self, previous: SecretConfig) -> SecretOutputs:
        # Update secret version
        ...

    async def on_delete(self) -> None:
        # Delete secret
        ...
```

## Development

```bash
# Install dependencies
task install

# Run all tests
task test

# Run all checks
task check

# Provider-specific tasks
task gcp:test
task gcp:check
```

## Repository Structure

```
providers/
├── packages/
│   └── gcp/              # GCP provider package
│       ├── src/
│       │   └── gcp_provider/
│       │       └── resources/
│       │           └── secret.py
│       └── tests/
├── pyproject.toml        # Workspace configuration
└── README.md
```

## License

MIT
