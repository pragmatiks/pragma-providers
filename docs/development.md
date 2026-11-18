# Development

Cross-provider changes, mise tasks, provider environments, the toolchain and the lockfiles.

## Cross-cutting changes

**Never bundle changes across multiple providers in a single commit or PR.** Each provider must be modified independently.

Exception: a workspace-level change (build, publishing, naming) lands as one commit per provider in one rebase-merged PR.

When an SDK interface change affects all providers:
1. SDK change publishes to PyPI first
2. SDK cascade (`update-sdk.yaml`) updates the root lockfile
3. Each provider adapts individually — one commit per provider
4. The publish workflow handles ordering (gcp → kubernetes → qdrant/agno)

## Tasks

Always use mise tasks (`mise tasks` lists them all):

| Command | Purpose |
|---------|---------|
| `mise run install` | Install the toolchain from `mise.lock`, every provider from `uv.lock`, and the git hooks |
| `mise run format` | Format with ruff |
| `mise run check` | Lint + type check, each provider in its own environment |
| `mise run check:wheels` | Build every provider's wheel once with `--no-sources` and the hash-locked `uv_build` (as publishing does), install each with only the sibling wheels its `Requires-Dist` reaches (a floor above a sibling's in-tree, last released, version warns and that in-tree wheel stands in as a uv override; on the bumped release tree no floor is above, so the check is fully strict there), and fail on a failed `pragma_sdk.introspection` report or `pragma.embeds` keys that differ from the providers among them; CI runs it on every pull request and push |
| `mise run check:floors` | Resolve the workspace with `--no-sources` (a provider floor above a sibling's in-tree, last released, version warns; any other resolve failure fails), then install each provider's wheel with every sibling it reaches at the lowest `<provider>-v<version>` tag released under the required distribution name that its floor allows (recursively at that tag, built from the tag with `uv build --wheel --no-sources`; a floor no tag satisfies yet warns, and the sibling's in-tree wheel stands in as a uv override that satisfies every requirement on it, read in the working tree for its own siblings, while siblings at a tag get no override) and fail on a failed `pragma_sdk.introspection` report or embeds mismatch; needs the release tags fetched; CI runs it on every pull request and push |
| `mise run lock:toolchain` | Regenerate `mise.lock` and `.mise/locks/` (macOS arm64 or Linux x64 only; never bumps) |
| `mise run check:mise-pins` | Fail when a `jdx/mise-action` step's `version:` or `sha256:` differs from `min_version` and its runner's mise binary checksum |
| `mise run check:toolchain-scripts` | Fail when a shared toolchain script under `scripts/tasks/` or the git hook `scripts/git_hooks/hook.sh` differs from the public repositories' copies (same-named branch, else `main`) |

## Provider environments

`mise run check` and `mise run format` run each provider in its own environment, `.venvs/<distribution>/` at the repository root (gitignored), built by `uv run --package <distribution> --exact --locked --group dev` from `uv.lock`. It holds only the provider, its declared and transitive dependencies (workspace siblings it declares, editable) and its `dev` group, matching what the provider sees when installed alone, so ty fails an import of a sibling the provider does not declare. The distribution name is read from `project.name` in the provider's `pyproject.toml`. The root `.venv` from `mise run install` still holds every provider and serves editors and the git hooks.

## Toolchain

mise is the only toolchain source. `mise.toml` pins python, uv, pre-commit, git-cliff, commitizen, the pragma CLI (`pragmatiks-cli`, with `pragmatiks-lint[check]` and semgrep) and yq; `mise.lock` and `.mise/locks/` are committed. Install the pinned mise version (`min_version` in `mise.toml`); the install and the one-sweep bump rule across repositories live in the hub's `docs/runbooks/local-dev-setup.md`. A toolchain change ships with its `mise run lock:toolchain` output in the same commit. On a fresh clone run `mise trust`, then `mise run install`.

## Lockfile

The root `uv.lock` is committed and must match the manifests; so must `scripts/publish_tooling/uv.lock`, the separate lock of the pragma CLI and `uv_build` backend that publishing installs (not a workspace member). Any change to a `pyproject.toml` ships with its `uv lock` in the same commit. Relock both with `mise run lock`; `mise run check` and CI fail on a stale tooling lock. Bumping the publish CLI or backend pin is an edit to `scripts/publish_tooling/pyproject.toml` plus `mise run lock`, in a `build:` PR. `[tool.uv] required-version` must equal the uv version in `mise.toml`; uv refuses to run at any other version.
