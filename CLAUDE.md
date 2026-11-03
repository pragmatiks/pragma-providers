# CLAUDE.md

## Project

**pragma-providers**: Official providers for the Pragmatiks platform (GCP, AWS, etc.).

## Shared rules

Delivery lives in pragma-kit `project-instructions.md`. Evidence-based development, the solution preference order, the new-dependency proposal, the 13 engineering principles and the reviewer and developer protocols live in the pragma-kit plugin's `pragma-kit:engineering-principles` skill: load it before planning, writing or reviewing a change here. This file holds only what is specific to this repository.

## Architecture

Providers handle resource lifecycle events (CREATE, UPDATE, DELETE) by calling cloud APIs and returning results to the platform.

**The providers form one uv workspace.** Each provider is its own package with its own version and changelog. Providers that build on a sibling depend on it through a `workspace = true` source, never a path. Every provider publishes to the Pragmatiks registry only.

## Cross-Cutting Changes

**Never bundle changes across multiple providers in a single commit or PR.** Each provider must be modified independently.

Exception: a workspace-level change (build, publishing, naming) lands as one commit per provider in one rebase-merged PR.

When an SDK interface change affects all providers:
1. SDK change publishes to PyPI first
2. SDK cascade (`update-sdk.yaml`) updates the root lockfile
3. Each provider adapts individually — one commit per provider
4. The publish workflow handles ordering (gcp → kubernetes → qdrant/agno)

## Development

Always use mise tasks (`mise tasks` lists them all):

| Command | Purpose |
|---------|---------|
| `mise run install` | Install the toolchain from `mise.lock`, every provider from `uv.lock`, and the git hooks |
| `mise run format` | Format with ruff |
| `mise run check` | Lint + type check |
| `mise run lock:toolchain` | Regenerate `mise.lock` and `.mise/locks/` (macOS arm64 or Linux x64 only; never bumps) |

### Toolchain

mise is the only toolchain source. `mise.toml` pins python, uv, pre-commit, git-cliff, commitizen and yq; `mise.lock` and `.mise/locks/` are committed. Install the pinned mise version (`min_version` in `mise.toml`); the install and the one-sweep bump rule across repositories live in the hub's `docs/runbooks/local-dev-setup.md`. A toolchain change ships with its `mise run lock:toolchain` output in the same commit. On a fresh clone run `mise trust`, then `mise run install`.

### Lockfile

The root `uv.lock` is committed and must match the manifests. Any change to a `pyproject.toml` ships with its `uv lock` in the same commit. Relock with `mise run lock`. `[tool.uv] required-version` must equal the uv version in `mise.toml`; uv refuses to run at any other version.

Humans change `main` only through pull requests. The pragma-ci-bot GitHub App pushes release commits to it directly, from the `push` job of `publish.yaml` in the main-only `release` environment. CI runs `uv lock --check` and `uv sync --all-packages --all-groups --locked` on every pull request and on every push to `main`, except release pushes: they carry `[skip ci]`, so GitHub starts no workflow for them (`ci.yaml` also skips a push whose head commit starts with `bump:`).

### Releases

Every push to `main` that starts workflows, and a manual dispatch from `main`, runs `publish.yaml`. Runs queue in order and run one at a time. A pull request that `update-sdk.yaml` merges with `GITHUB_TOKEN` starts no workflow, so its providers release with the next run, or dispatch `publish.yaml` from `main`.

- `bump` checks out the latest `main` and handles every provider in dependency order (gcp → kubernetes → qdrant/agno). git-cliff, configured by the root `cliff.toml`, reads only the commits since the provider's tag that touch its own path (`packages/<provider>`), computes the next version and prepends the changelog entry; commitizen then writes the version, commits and tags. A provider releases only when one of those commits is a `feat`, `fix`, `refactor` or `perf` commit or a breaking change, and its changelog lists only those commits. A provider that depends on one bumped in the same run is bumped too, as a PATCH with a heading-only changelog entry when it has no such commit of its own. The root `pyproject.toml` and `ci:`/`chore:`/`docs:` commits release nothing. `bump` then relocks `uv.lock` as `build: relock the provider workspace after version bumps [skip ci]`. It runs all third-party tooling and holds no secrets.
- `push` pushes `main` and every new tag in one atomic push. When `main` moved after the bump, it pushes nothing and starts a new `publish.yaml` run on `main`, which bumps from the latest `main`; any other rejection fails the run.
- Each provider then publishes from the commit its tag points to, after verifying the tag on GitHub, and gets a GitHub Release, created last (for agno, after its runner image). A provider publishes only when every provider it depends on, directly or through the chain, published in this run or had nothing to publish; one that failed stops its dependents. Any later run completes a release a failed run left behind: a provider with nothing to bump whose current tag has no GitHub Release is published from that tag without a new bump or push. A version the registry already published and an existing release are skipped, so **Re-run failed jobs**, **Re-run all jobs**, a new dispatch and the next merge's run all recover. To re-publish every provider's current version, for example after a registry reset, dispatch `publish.yaml` from `main` with `publish_only`.

`git-cliff`, `commitizen`, uv and python are pinned in `mise.toml`; every workflow job that runs them installs them through the pinned mise-action step, and `environment: release` jobs install nothing.

## Provider Interface

Each provider implements resource classes using the SDK:

```python
from pragma_sdk import Provider, Resource, Config, Outputs

gcp = Provider()


@gcp.resource("storage")
class Bucket(Resource[BucketConfig, BucketOutputs]):
    async def on_create(self) -> BucketOutputs: ...
    async def on_observe(self) -> BucketOutputs | None: ...
    async def on_update(self, previous_config: BucketConfig | None) -> BucketOutputs: ...
    async def on_delete(self) -> None: ...
```

## Testing

- No mocks, no `tests/`

## Findability audit

Pragmatiks can stop a handler at any moment, including after a create is sent and before its answer arrives; the retry calls `on_observe` first. So the object a create sends must be findable by `on_observe` from the resource's own identity from the moment the create is sent, and every handler must be safe to rerun. Each provider records how every resource type meets this in `packages/<provider>/FINDABILITY.md`, one row per type:

| Column | Content |
|---|---|
| Type | `<provider>/<type>`; `computed` for a type that writes nothing outside Pragmatiks |
| Identity | Request fields that name the object; each field `on_observe` reads must be `ImmutableField` or `ImmutableDependency` |
| `on_observe` lookup | Call and key, cited as `Class.method` |
| Uniqueness | Does the vendor refuse a second object with that identity, or does the create carry an idempotency key that returns the original object? Vendor URL |
| Findable from send | Once the vendor has stored the object, even for a create whose answer never arrived, does that lookup find it? One of: sync / LRO visible while pending / eventually consistent / unknown. Vendor URL |
| Conflict on create | What the create does when the vendor reports the identity taken: idempotency key returns the original / marker re-observe (recognizes its own object by the resource marker the create request carried, else fails) / raises / adopts any object |
| Partial create | Writes after the first one (or between child applies); does `on_update(previous_config=None)` finish every prefix? |
| Rerun | `on_update` converges from a partial earlier update; `on_delete` succeeds on an absent object |
| Verdict | `holds`, or `gap (PRA-n)` naming the filed issue |
| Evidence | Symbols, vendor URL and the date it was read |

A verdict is `holds` only when Uniqueness is yes, Findable from send is `sync` or `LRO visible while pending`, every lookup field is immutable, Conflict on create is the idempotency key or the marker re-observe, Partial create is finishable and Rerun converges. Where the vendor is silent, assume not unique and not findable; no timing argument replaces a check, since a vendor can store the object after its client gave up. A type that cannot hold is a gap with a filed issue; a fix lands with the row it turns to `holds`.

**Resource marker.** The marker a create request carries is the resource marker: the key `pragma_resource` holding the canonical resource ID (`self.id`), written in the vendor's own medium (label, annotation, metadata). A provider whose create recognizes its own object by the marker keeps it in one `marker.py` module (`RESOURCE_MARKER_KEY`, `build_resource_marker`, `match_resource_marker`, `MarkerMatch`) and logs `<provider>_create_conflict_recognized` or `<provider>_create_conflict_refused ... reason=<absent|unmarked|other_resource>` on a conflict; kubernetes, which recreates an object gone when read back and refuses its own object while it is being deleted, logs those two cases as `kubernetes_create_conflict_absent` and `kubernetes_create_conflict_terminating`. A vendor naming rule that forces another key spelling or value encoding is named in that provider's `FINDABILITY.md`. The marker tells this resource's earlier create from an object made outside Pragmatiks or by another resource; it is not proof against someone who can write the object, and it names the resource identity, so an object left by an earlier incarnation of the same resource ID is recognized as its own.

**Review check.** A diff that changes a resource type's create, observe, update, delete or conflict path, or adds a type, updates that type's row in `packages/<provider>/FINDABILITY.md` with the columns and verdict rule above. A missing or stale row = blocker. A row whose verdict is not `holds` names its gap issue.

## Publishing to the Pragmatiks registry

Each provider publishes to the Pragmatiks registry as its own distribution, never to PyPI:

| Provider | Distribution | Tag Format |
|----------|--------------|------------|
| GCP | `pragmatiks-gcp` | `gcp-v{version}` |
| Supabase | `pragmatiks-supabase` | `supabase-v{version}` |
| Vercel | `pragmatiks-vercel` | `vercel-v{version}` |
| GitHub | `pragmatiks-github` | `github-v{version}` |
| Pragma | `pragmatiks-pragma` | `pragma-v{version}` |
| Kubernetes | `pragmatiks-kubernetes` | `kubernetes-v{version}` |
| Qdrant | `pragmatiks-qdrant` | `qdrant-v{version}` |
| Agno | `pragmatiks-agno` | `agno-v{version}` |

**Versioning**: CI bumps each provider with git-cliff from the conventional commits on its own paths and pushes the tag. Never bump versions or edit changelogs by hand.

**Publishing**: `scripts/publish_platform_provider.sh` builds the wheel with `uv build --wheel --no-sources`, which ignores `[tool.uv.sources]` so build requirements resolve from the package index, not the workspace, and publishes it with `pragma providers publish --wheel`.

**Embedded providers**: a provider declares every Pragmatiks provider among its dependencies, direct or transitive, under `[project.entry-points."pragma.embeds"]`, keyed by distribution name. The platform reads only the key; by convention the value is the embedded provider's import package:

```toml
[project.entry-points."pragma.embeds"]
pragmatiks-gcp = "gcp_provider"
```
