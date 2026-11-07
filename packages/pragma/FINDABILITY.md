# Findability audit: pragma

Each row records how a resource type meets the Pragmatiks findability and idempotency contract; the contract, the columns and the verdict rule are in `## Findability audit` of the repository's `CLAUDE.md`. Audited 2026-11-02 against `main@8754f36`; vendor documentation read 2026-11-02.

| Type | Identity | `on_observe` lookup | Uniqueness | Findable from send | Conflict on create | Partial create | Rerun | Verdict | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| `pragma/file` | Project (`self.project_id`, part of the resource identity, immutable) and name. No create request: the object is `files/{organization_id}/{project_id}/{name}` in the file bucket, uploaded through the Pragmatiks API, whose file service owns that key; this handler only reads it and follows it; `on_create` only reads it | `File.on_observe` → `head_async` of `files/{organization_id}/{project_id}/{name}`, then its `.meta` sidecar | Yes: an object key names one object | sync: object read-after-write is strongly consistent, "the object is immediately available for reading and metadata operations" | None: `on_create` writes nothing and fails when the content is not uploaded | No write | `on_update` rereads; `on_delete` deletes the sidecar and the content, each ignoring a missing object | holds | `File.on_observe`, `File.on_create`, `File.on_delete`; https://docs.cloud.google.com/storage/docs/consistency |
| `pragma/config` | `computed` | — | — | — | — | — | — | holds: no vendor write; outputs are `config.data` | `ConfigResource.on_create` |
| `pragma/secret` | `computed` | — | — | — | — | — | — | holds: no vendor write; outputs are the configured values | `Secret.on_create` |
