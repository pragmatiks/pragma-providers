<p align="center">
  <img src="assets/wordmark.png" alt="Pragmatiks" width="800">
</p>

# Pragma Providers

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pragmatiks/pragma-providers)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**[Documentation](https://docs.pragmatiks.io/providers/overview)** | **[SDK](https://github.com/pragmatiks/pragma-sdk)** | **[CLI](https://github.com/pragmatiks/pragma-cli)**

Resource providers for Pragmatiks.

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
pragma resources apply secret.yaml
pragma resources get gcp/secret db-password
```

## Available Providers

### GCP Provider

Manage Google Cloud Platform resources.

```bash
pragma providers install pragmatiks/gcp
```

| Resource | Description |
|----------|-------------|
| `gcp/secret` | Secret Manager secrets |
| `gcp/gke` | GKE Autopilot clusters |
| `gcp/cloudsql/database_instance` | Cloud SQL instances |
| `gcp/cloudsql/database` | Cloud SQL databases |
| `gcp/cloudsql/user` | Cloud SQL users |

### Supabase Provider

Manage Supabase projects.

```bash
pragma providers install pragmatiks/supabase
```

| Resource | Description |
|----------|-------------|
| `supabase/project` | Supabase projects and their authentication settings |

### Vercel Provider

Manage Vercel projects, deployments, and domains.

```bash
pragma providers install pragmatiks/vercel
```

| Resource | Description |
|----------|-------------|
| `vercel/project` | Vercel projects |
| `vercel/deployment` | Deployments of a Vercel project |
| `vercel/domain` | Custom domains on Vercel projects |

### GitHub Provider

Manage GitHub repositories, environments, and secrets.

```bash
pragma providers install pragmatiks/github
```

| Resource | Description |
|----------|-------------|
| `github/repository` | GitHub repositories |
| `github/environment` | GitHub deployment environments |
| `github/secret` | GitHub repository and environment secrets |

### Pragma Provider

Manage secrets, configuration, and files stored by Pragmatiks.

```bash
pragma providers install pragmatiks/pragma
```

| Resource | Description |
|----------|-------------|
| `pragma/secret` | Platform-managed secrets |
| `pragma/config` | Non-sensitive configuration values |
| `pragma/file` | Platform-managed file storage |

### Kubernetes Provider

Manage Kubernetes resources.

```bash
pragma providers install pragmatiks/kubernetes
```

| Resource | Description |
|----------|-------------|
| `kubernetes/config` | Authenticated Kubernetes cluster access |
| `kubernetes/deployment` | Kubernetes Deployments |
| `kubernetes/service` | Kubernetes Services |
| `kubernetes/configmap` | Kubernetes ConfigMaps |
| `kubernetes/secret` | Kubernetes Secrets |
| `kubernetes/statefulset` | Kubernetes StatefulSets |
| `kubernetes/namespace` | Kubernetes Namespaces |

### Qdrant Provider

Vector database for similarity search.

```bash
pragma providers install pragmatiks/qdrant
```

| Resource | Description |
|----------|-------------|
| `qdrant/collection` | Qdrant vector collections |
| `qdrant/database` | Qdrant databases deployed to Kubernetes |

### Agno Provider

AI agent deployment.

```bash
pragma providers install pragmatiks/agno
```

| Resource | Description |
|----------|-------------|
| `agno/agent` | Agno agent definitions |
| `agno/db/postgres` | Agno Postgres storage for agents |
| `agno/runner` | Agno runners on Kubernetes |
| `agno/knowledge` | Agno knowledge bases for semantic search |
| `agno/knowledge/content` | Agno knowledge content sources |
| `agno/knowledge/embedder/openai` | Agno OpenAI embedders |
| `agno/memory/manager` | Agno memory managers |
| `agno/models/anthropic` | Agno Anthropic Claude models |
| `agno/models/openai` | Agno OpenAI chat models |
| `agno/prompt` | Agno prompt templates |
| `agno/team` | Agno team definitions |
| `agno/tools/mcp` | Agno MCP server tools |
| `agno/tools/websearch` | Agno web search tools |
| `agno/vectordb/qdrant` | Agno Qdrant vector databases |

## Using Provider Resources

Reference provider resources in your configurations:

```python
from pragma_sdk import FieldReference

config = AppConfig(
    database_password=FieldReference(provider="gcp", resource="secret", name="db-password", field="data")
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
pragma providers init mycompany

# Implement your resources
cd mycompany-provider
# Edit src/mycompany_provider/resources/

# Publish a version, then install it in your organization
pragma providers publish
pragma providers install <org>/mycompany
```

See the [Building Providers Guide](https://docs.pragmatiks.io/building-providers/overview) for complete documentation.

## Provider Architecture

Each provider contains:

- **Provider namespace** - Groups related resources (e.g., `gcp`)
- **Resource types** - Individual resource definitions with Config and Outputs
- **Lifecycle methods** - `on_create`, `on_observe`, `on_update`, `on_delete` implementations; `on_observe` locates the external object from identity and config alone and returns `None` when it does not exist
- **Computed types** - resources with no external object of their own declare `computed = True` and need no `on_observe`

```python
from pragma_sdk import Resource, Config, Outputs


class SecretConfig(Config):
    project_id: str
    secret_id: str
    data: str


class SecretOutputs(Outputs):
    resource_name: str
    version_id: str


class Secret(Resource[SecretConfig, SecretOutputs]):
    async def on_create(self) -> SecretOutputs:
        # Create secret in GCP Secret Manager
        ...

    async def on_observe(self) -> SecretOutputs | None:
        # Read the secret from GCP Secret Manager, None when absent
        ...

    async def on_update(self, previous_config: SecretConfig | None) -> SecretOutputs:
        # Add a secret version; previous_config is None when the secret already exists at create
        ...

    async def on_delete(self) -> None:
        # Delete secret
        ...
```

## Development

```bash
# Install dependencies
task install

# Run all checks
task check

# Provider-specific tasks
task gcp:check
```

## Repository Structure

```
pragma-providers/
├── packages/
│   ├── gcp/              # GCP provider (secret, gke, cloudsql)
│   ├── supabase/         # Supabase provider (project)
│   ├── vercel/           # Vercel provider (project, deployment, domain)
│   ├── github/           # GitHub provider (repository, environment, secret)
│   ├── pragma/           # Pragma provider (secret, config, file)
│   ├── kubernetes/       # Kubernetes provider (deployment, service, secret, ...)
│   ├── qdrant/           # Qdrant provider (collection, database)
│   └── agno/             # Agno provider (agent, team, knowledge, ...)
├── pyproject.toml        # Workspace configuration
└── README.md
```

## License

MIT
