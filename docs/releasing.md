# Releasing

How providers are versioned, released and published to the Pragmatiks registry.

## Main branch and CI

Humans change `main` only through pull requests. The pragma-ci-bot GitHub App pushes release commits to it directly, from the `push` job of `publish.yaml` in the main-only `release` environment. CI runs `uv lock --check`, `mise run check:tooling-lock` and `mise run check:ci` (every provider's check in its own `--locked` environment, the pre-commit hooks and the Pragmatiks lint rules) on every pull request and on every push to `main`, except release pushes: they carry `[skip ci]`, so GitHub starts no workflow for them (`ci.yaml` also skips a push whose head commit starts with `bump:`).

## Releases

Every push to `main` that starts workflows, and a manual dispatch from `main`, runs `publish.yaml`. Runs queue in order and run one at a time. `update-sdk.yaml` merges with `GITHUB_TOKEN`, which starts no push-triggered run, so after it merges an SDK bump it dispatches `publish.yaml` on `main` itself; a bump that crosses an SDK major version it opens but leaves for review, and the human merge starts `publish.yaml`.

- `bump` checks out the latest `main` and handles every provider in dependency order (gcp → kubernetes → qdrant/agno). git-cliff, configured by the root `cliff.toml`, reads only the commits since the provider's tag that touch its own path (`packages/<provider>`), computes the next version and prepends the changelog entry; commitizen then writes the version, commits and tags. A provider releases only when one of those commits is a `feat`, `fix`, `refactor` or `perf` commit or a breaking change, and its changelog lists only those commits. A provider that depends on one bumped in the same run is bumped too, as a PATCH with a heading-only changelog entry when it has no such commit of its own. agno, which ships the runner image, is bumped the same way when the third-party runtime dependencies `uv.lock` resolves for it (`uv export --no-dev --no-emit-workspace`) differ from those at its release tag, whatever commit changed the lock. The root `pyproject.toml` and `ci:`/`chore:`/`docs:` commits release nothing. `bump` then relocks `uv.lock` as `build: relock the provider workspace after version bumps [skip ci]`. When anything was bumped, `bump` runs `mise run check:wheels` on the bumped tree before `push`, so a failure tags and pushes nothing. It runs all third-party tooling and holds no secrets.
- `push` pushes `main` and every new tag in one atomic push. When `main` moved after the bump, it pushes nothing and starts a new `publish.yaml` run on `main`, which bumps from the latest `main`; any other rejection fails the run.
- Each provider then publishes from the commit its tag points to, after verifying the tag on GitHub, and gets a GitHub Release, created last (agno publishes only after its runner image: `agno-runner-image.yaml` verifies the tag, pushes `ghcr.io/pragmatiks/agno-runner:<agno version>` unless it exists, then points the transport tag and `latest` at its digest, so the Runner's default image, that version tag, exists before any agno version that names it can be installed; on a pull request that touches agno, its siblings or the lock it only builds the image). A provider publishes only when every provider it depends on, directly or through the chain, published in this run or had nothing to publish; one that failed stops its dependents. Any later run completes a release a failed run left behind: a provider with nothing to bump whose current tag has no GitHub Release is published from that tag without a new bump or push. A version the registry already published and an existing release are skipped, so **Re-run failed jobs**, **Re-run all jobs**, a new dispatch and the next merge's run all recover. To re-publish every provider's current version, for example after a registry reset, dispatch `publish.yaml` from `main` with `publish_only`.

`git-cliff`, `commitizen`, uv and python are pinned in `mise.toml`; every workflow job that runs them installs them through the pinned mise-action step, and `environment: release` jobs install nothing.

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

**Publishing**: `publish-provider.yaml` takes its tooling (scripts, `mise.toml`, the tooling lock) from the workflow's own commit and only the provider package from the tag commit, checked out under `release-tag/`, in three jobs. `build` runs with no environment, no secret and read-only contents: `scripts/build_provider_wheel.sh` builds the wheel with `uv build --no-config --wheel --no-sources` (so the tag commit's `[tool.uv]` settings and sources are ignored and any tag builds under the pinned uv) and `uv_build` constrained by hash to the version `scripts/publish_tooling/uv.lock` pins, refusing a `[build-system]` that locked backend cannot satisfy, and uploads it as an artifact; `mise run check:wheels` builds the same way. `publish` runs in the main-only `pypi` environment, which holds `PRAGMA_PUBLISH_API_KEY`, on a fresh runner that runs no tag-commit code: it downloads the wheel, writes the key to a file, and `scripts/publish_platform_provider.sh` moves the key into the hash-locked pragma CLI's isolated credentials file (never an environment variable or argument), checks the wheel's distribution and version, and publishes with `pragma providers publish --wheel`. `release`, when `create_release` is set, then creates the GitHub Release through `release-provider.yaml`.

**Embedded providers**: a provider declares every Pragmatiks provider among its dependencies, direct or transitive, under `[project.entry-points."pragma.embeds"]`, keyed by distribution name. The platform reads only the key; by convention the value is the embedded provider's import package:

```toml
[project.entry-points."pragma.embeds"]
pragmatiks-gcp = "gcp_provider"
```
