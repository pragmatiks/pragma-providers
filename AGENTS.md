# Pragmatiks Providers Codex Instructions

## Project

This repository is `pragma-providers`, the official provider monorepo for **Pragmatiks**. Providers include `gcp`, `kubernetes`, `qdrant`, `agno`, `supabase`, `vercel`, `github`, and `pragma`.

Each `packages/<name>/` directory is an independent PyPI package named `pragmatiks-<name>-provider`. Providers depend on each other through published packages, not local source.

User-facing product surfaces must say **Pragmatiks**. Do not use "pragma-os", "OS", or operating-system framing in docstrings, READMEs, error strings, OpenAPI/Pydantic descriptions, or docs unless referring specifically to a repository name or infrastructure path that already depends on it.

## Architecture

Providers handle resource lifecycle events by calling external APIs and returning results to the platform. Resource handlers implement CREATE, UPDATE, and DELETE behavior through the Pragmatiks SDK provider interface.

```python
from pragma_sdk import Provider, Resource, Config, Outputs

gcp = Provider()

@gcp.resource("storage")
class Bucket(Resource[BucketConfig, BucketOutputs]):
    async def on_create(self) -> BucketOutputs: ...
    async def on_update(self, previous: BucketConfig) -> BucketOutputs: ...
    async def on_delete(self) -> None: ...
```

Core principles:

- Each provider package can be changed, tested, versioned, and published independently.
- Providers call cloud or SaaS APIs for resource lifecycle work.
- Runtime behavior consumes published wheels; committed sibling source overrides are not allowed.
- `agno` retains a dedicated runner container while still publishing as `pragmatiks-agno-provider`.

## Repository Layout

```text
packages/
|-- agno/          PyPI: pragmatiks-agno-provider
|-- gcp/           PyPI: pragmatiks-gcp-provider
|-- github/        PyPI: pragmatiks-github-provider
|-- kubernetes/    PyPI: pragmatiks-kubernetes-provider
|-- pragma/        PyPI: pragmatiks-pragma-provider
|-- qdrant/        PyPI: pragmatiks-qdrant-provider
|-- supabase/      PyPI: pragmatiks-supabase-provider
`-- vercel/        PyPI: pragmatiks-vercel-provider
taskfiles/         Provider-specific Task includes
assets/            Shared provider assets
templates/         Shared provider templates
scripts/           Repository maintenance scripts
```

## Cross-Cutting Changes

Never bundle changes across multiple providers in a single commit or PR. Each provider must be modified independently.

CI tests each provider against its published PyPI dependencies. Cross-provider commits will fail because sibling providers resolve from PyPI, not local source.

When an SDK interface change affects all providers:

1. Publish the SDK change to PyPI first.
2. Let the SDK cascade workflow, `update-sdk.yaml`, update lockfiles in this repo.
3. Adapt each provider independently, one commit per provider.
4. Let the publish workflow handle ordering: `gcp` before `kubernetes` and `qdrant`, then `agno`.

## Commands

Use `task` as the project interface. Do not run `uv`, `pytest`, `ruff`, or package-local tooling directly for normal development validation.

Common commands:

| Command | Purpose |
|---|---|
| `task install` | Sync dependencies |
| `task test` | Run pytest with respx mocks and ProviderHarness |
| `task format` | Format with ruff |
| `task check` | Run linting and type checks |

Provider-specific task groups follow the package name:

- `task gcp:check`
- `task kubernetes:test`
- `task qdrant:format`
- `task agno:check`
- `task supabase:test`
- `task vercel:check`
- `task github:test`
- `task pragma:check`

## Testing Policy

This repository has a test suite. The sanctioned full validation command is `task test`, which runs pytest across provider packages.

Testing conventions:

- Mock external cloud and SaaS APIs.
- Use `respx` for `httpx` mocking.
- Use `ProviderHarness` from `pragmatiks-sdk` for lifecycle testing without deployment.
- Test success and failure paths.
- Do not add e2e tests here; e2e coverage lives in `pragma-os/tests/e2e/`.

Before considering provider changes ready for review, run `task check` and `task test` from the repository root.

## Dependency Policy

Dependencies must resolve from registries in committed configuration.

Do not commit sibling-repo path overrides such as:

- `pragmatiks-sdk = { path = "../../pragma-sdk", editable = true }`
- `extra-paths = ["../pragma-sdk/src"]`

Those paths break in agent worktrees because `../../pragma-sdk` resolves under `.claude/worktrees/`, not the workspace root.

For local iteration against a sibling repo, use an ad hoc editable install against the active worktree `.venv`, such as `uv pip install -e ../pragma-sdk`, and keep committed project files registry-based.

## Secrets And Local Files

Never commit local secrets, personal Codex config, MCP auth, hooks, or machine-specific files.

Important local values and files:

- `PYPI_TOKEN` for publishing belongs in shell environment or CI secrets, never in the repository.
- Cloud credentials for live provider testing, including GCP service account JSON and Kubernetes kubeconfig files, must stay in user-controlled paths.
- Do not commit fixture files containing real credentials.
- Do not commit `.env` values or local credential files.

## Claude Code Compatibility

Keep `CLAUDE.md` as-is. Codex-specific durable instructions live in `AGENTS.md`.

When changing shared guidance, keep `CLAUDE.md` and `AGENTS.md` consistent where the rule is shared, but do not replace one system's files with the other's format.

## Git And Worktrees

Codex typically works inside a per-thread worktree. Treat the current worktree as the working area for the current thread. Do not create additional worktrees unless the user asks.

Before editing, inspect relevant files and current git status. Never revert user changes or unrelated generated changes.

Per-provider commit discipline applies inside worktrees too. If a single thread touches multiple providers, split the work into separate commits and PRs per provider before pushing.

## Linear And Issue Work

Use Linear as the source of truth for planned work and follow-ups. Prefer the `linear`/`linearis` CLI when MCP tools are unavailable or the user asks for CLI usage; the two commands are equivalent on this machine and return JSON-friendly output.

Autonomy expectations:

- When the user asks to work on a Linear issue, read it first with comments and attachments, then map the issue to provider package files and validation commands.
- If the user clearly starts work on a specific issue, move it to an active status when appropriate.
- If a bug, follow-up, cleanup, or feature is deferred for later and the scope is clear, create a Linear issue instead of leaving only chat context.
- Add obvious relationships when creating or updating issues: `--blocked-by`, `--blocks`, `--relates-to`, `--duplicate-of`, or `--parent-ticket`.
- When a PR is merged, identify associated Linear issues from branch names, commits, PR title/body, or user context, then mark fully completed issues Done and add a concise validation/merge comment.
- If merged work only partially addresses an issue, leave it open, comment with current status, and create/link follow-up issues as needed.

Escalate to the user before changing Linear when there is a real product or planning choice: ambiguous team/project, unclear priority, competing dependency direction, uncertain status, or whether a partial fix should close an issue.

Useful commands:

- `linear issues read PRA-123 --with-comments --with-attachments`
- `linear issues search "<query>" --limit 20`
- `linear issues create "<title>" --team PRA --description "<markdown>" --priority 3`
- `linear issues update PRA-123 --status "In Progress"`
- `linear issues update PRA-123 --status "Done"`
- `linear issues discuss PRA-123 --body "<markdown>"`

Do not mark issues Done for local-only work. Completion means the requested outcome is implemented, validated, and merged or explicitly accepted by the user.

## Publishing To PyPI

Each provider is its own package with its own version, changelog, and tag.

| Provider | PyPI package | Tag format |
|---|---|---|
| agno | `pragmatiks-agno-provider` | `agno-v{version}` |
| gcp | `pragmatiks-gcp-provider` | `gcp-v{version}` |
| github | `pragmatiks-github-provider` | `github-v{version}` |
| kubernetes | `pragmatiks-kubernetes-provider` | `kubernetes-v{version}` |
| pragma | `pragmatiks-pragma-provider` | `pragma-v{version}` |
| qdrant | `pragmatiks-qdrant-provider` | `qdrant-v{version}` |
| supabase | `pragmatiks-supabase-provider` | `supabase-v{version}` |
| vercel | `pragmatiks-vercel-provider` | `vercel-v{version}` |

Versioning is per package:

```bash
cd packages/<name>
cz bump
```

Publishing is per package and requires `PYPI_TOKEN`:

```bash
cd packages/<name>
uv build
uv publish
```

Wheel-based provider publishing is the default runtime path. Runtime dynamically installs published wheels. Docker-to-wheels migration is complete for provider packages; `agno` keeps a dedicated container.
