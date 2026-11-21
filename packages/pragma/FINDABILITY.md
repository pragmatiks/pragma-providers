# Findability audit: pragma

Each row records how a resource type meets the Pragmatiks findability and idempotency contract; the contract, the columns and the verdict rule are in `## Findability audit` of the repository's `CLAUDE.md`. Audited 2026-11-21 against `main@61cc45d`.

| Type | Identity | `on_observe` lookup | Uniqueness | Findable from send | Conflict on create | Partial create | Rerun | Verdict | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| `pragma/file` | `computed` | — | — | — | — | — | — | holds: no vendor write; outputs read through the address the hub signs for `config.content`, which must name the resource's own file; the hub deletes that file after the resource is deleted | `File.on_create` |
| `pragma/config` | `computed` | — | — | — | — | — | — | holds: no vendor write; outputs are `config.data` | `ConfigResource.on_create` |
| `pragma/secret` | `computed` | — | — | — | — | — | — | holds: no vendor write; outputs are the configured values | `Secret.on_create` |
