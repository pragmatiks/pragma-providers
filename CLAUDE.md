# CLAUDE.md

> Tasks tracked in Linear. SessionStart hook injects issue context from branch name.

## Project

**pragma-providers**: Official providers for the Pragmatiks platform (GCP, AWS, etc.).

## Architecture

Providers handle resource lifecycle events (CREATE, UPDATE, DELETE) by calling cloud APIs and returning results to the platform.

**Each provider is an independent PyPI package.** Providers depend on each other through PyPI, not local source. They could live in separate repositories — this monorepo is a convenience, not a coupling.

## Cross-Cutting Changes

**Never bundle changes across multiple providers in a single commit or PR.** Each provider must be modified independently.

When an SDK interface change affects all providers:
1. SDK change publishes to PyPI first
2. SDK cascade (`update-sdk.yaml`) updates lockfiles in this repo
3. Each provider adapts individually — one commit per provider
4. The publish workflow handles ordering (gcp → kubernetes/qdrant → agno)

The CI tests each provider against its published PyPI dependencies. Cross-provider commits will fail CI because sibling providers resolve from PyPI, not local source.

## Development

Always use `task` commands:

| Command | Purpose |
|---------|---------|
| `task test` | Run pytest |
| `task format` | Format with ruff |
| `task check` | Lint + type check |

## Provider Interface

Each provider implements resource classes using the SDK:

```python
from pragma_sdk import Provider, Resource, Config, Outputs

gcp = Provider()

@gcp.resource("storage")
class Bucket(Resource[BucketConfig, BucketOutputs]):
    async def on_create(self) -> BucketOutputs: ...
    async def on_update(self, previous: BucketConfig) -> BucketOutputs: ...
    async def on_delete(self) -> None: ...
```

## Testing

- Mock external cloud APIs
- Use respx for httpx mocking
- Test success and failure paths
- Use `ProviderHarness` from SDK for lifecycle testing

## Publishing to PyPI

Each provider is a separate PyPI package:

| Provider | Package | Tag Format |
|----------|---------|------------|
| GCP | `pragmatiks-gcp-provider` | `gcp-v{version}` |

**Versioning** (commitizen, per-package):
```bash
cd packages/gcp
cz bump              # Bump version based on conventional commits
```

**Publishing**:
```bash
cd packages/gcp
uv build             # Build wheel and sdist
uv publish           # Publish to PyPI (requires PYPI_TOKEN)
```

**Note**: Each provider has its own version and changelog.
