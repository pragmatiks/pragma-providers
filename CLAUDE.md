# pragma-providers

Official Pragmatiks providers (GCP, Kubernetes, Supabase, Vercel, GitHub, Pragma, Qdrant, Agno). Each handles resource lifecycle events by calling a vendor API and returning outputs to the platform. The providers form one uv workspace: each `packages/<provider>` is its own package, version and changelog, published to the Pragmatiks registry only, never PyPI.

Delivery: pragma-kit `project-instructions.md`. Engineering rules: the `pragma-kit:engineering-principles` skill; load it before planning, writing or reviewing a change. Tasks, environments and lockfiles: `docs/development.md`. Releases and publishing: `docs/releasing.md`.

## Gotchas

- **One provider per change.** A commit and a PR touch one provider, so each releases on its own; a workspace-level change (build, publishing, naming) is one commit per provider in one PR, and an SDK interface change lands per provider after the SDK cascade relocks (`docs/development.md`).
- **Siblings through the workspace.** A provider that builds on a sibling depends on it through a `workspace = true` source, never a path, and declares every Pragmatiks provider it reaches under `[project.entry-points."pragma.embeds"]`; `mise run check:wheels` fails on a mismatch (`docs/releasing.md`).
- **Lock with the manifest.** A `pyproject.toml` change ships with `mise run lock` output in the same commit; a `mise.toml` change with `mise run lock:toolchain` output. `[tool.uv] required-version` must equal the uv pin in `mise.toml` (`docs/development.md`).
- **Releases are CI's.** Never bump a version or edit a changelog by hand; git-cliff bumps each provider from the conventional commits on its own path (`docs/releasing.md`).
- **No tests.** No mocks and no `tests/`.
- **Setup.** Fresh clone: `mise trust`, then `mise run install`. `mise run check` runs each provider in its own `.venvs/<distribution>/`, so ty fails an import of an undeclared sibling.

## Findability audit

A handler can stop after a create is sent and before its answer arrives, so the object a create sends must be findable by `on_observe` from the resource's own identity, every handler must be safe to rerun, and `on_observe` raises `AlreadyExistsError` for an object this resource did not create. The contract, the resource marker, the `FINDABILITY.md` columns and the verdict rule are in `docs/findability.md`.

Review check: a diff that changes a type's create, observe, update, delete or conflict path, or adds a type, updates its row in `packages/<provider>/FINDABILITY.md`. A missing or stale row, or a marker type whose `on_observe` reports an unmarked object instead of raising, is a blocker; a row not `holds` names its gap issue.
