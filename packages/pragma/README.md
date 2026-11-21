# Pragma Provider

Platform resources for [Pragmatiks](https://pragmatiks.io).

Declaratively manage secrets, non-sensitive configuration, and file storage alongside the rest of your infrastructure. Install it in your organization to use its resources, which other providers commonly reference through `FieldReference`.

## Installation

```bash
pragma providers install pragmatiks/pragma
```

## Resources

| Resource | Type Slug | Description |
|----------|-----------|-------------|
| [Secret](#secret) | `pragma/secret` | Platform-managed secrets referenced by other resources |
| [Config](#config) | `pragma/config` | Non-sensitive configuration values shared across resources |
| [File](#file) | `pragma/file` | Platform-managed file storage with download URLs |

---

### Secret

Stores sensitive key-value data and exposes each entry as an output so that other resources can reference individual values through `FieldReference`.

This is the resource type for a sensitive value written as a literal. A field that other providers mark as a secret accepts only a reference, so its literal belongs here and the field references `outputs.<key>`. Every value in `data` and every output is a secret: `describe`, `list`, the web graph and the live event stream show it as `********`, and a resource that references it receives the real value.

**Config:**
- `data` (`dict[str, SecretStr]`, required, mutable) -- Key-value pairs of secret data. Values must be strings.

**Outputs:**
- Each key-value pair from `data` is exposed as a separately referenceable secret output field (e.g., `outputs.username`, `outputs.password`).

```yaml
resources:
  database-credentials:
    provider: pragmatiks/pragma
    resource: secret
    config:
      data:
        username: app_service
        password: hunter2
```

**Behavior:**
- Create: Stores each key as a separate output for downstream references.
- Update: Replaces the full output set with the new `data` keys and values.
- Delete: No-op. Secret state is held alongside the resource itself.

---

### Config

Non-sensitive configuration store. Unlike `pragma/secret`, values may be any JSON-serializable type, which makes it a natural place to keep shared defaults (project IDs, regions, feature flags) that many resources reference.

**Config:**
- `data` (`dict[str, Any]`, required, mutable) -- Key-value pairs of configuration data. Values may be strings, numbers, booleans, lists, or nested objects.

**Outputs:**
- Each key-value pair from `data` is exposed as a separately referenceable output field (e.g., `outputs.project_id`, `outputs.region`), preserving its original type.

```yaml
resources:
  gcp-defaults:
    provider: pragmatiks/pragma
    resource: config
    config:
      data:
        project_id: my-gcp-project
        region: us-central1
        tags:
          - production
          - primary
```

**Behavior:**
- Create: Stores each key as a separate output for downstream references.
- Update: Replaces the full output set with the new `data` keys and values.
- Delete: No-op. Config state is held alongside the resource itself.

---

### File

Platform-managed file storage. The file content is uploaded into the resource's project by `pragma resources apply` (from a `content: "@./path"` field) or the SDK's `upload_file`; this resource reads what storage records for the stored file and exposes its URLs, size, content type, checksum and upload time as outputs.

**Config:** none -- the resource tracks the file stored under the resource's name.

**Outputs:**
- `url` -- Internal `pragma://` URL used for cross-resource references.
- `public_url` -- Download address under the project's file routes. It needs Pragmatiks credentials and redirects to the file's content.
- `size` -- File size in bytes.
- `content_type` -- Media type fixed at upload.
- `checksum` -- Hex MD5 of the file content, which storage verified when the file was stored (it matches `md5sum`).
- `uploaded_at` -- When the file's current content was stored.

```yaml
resources:
  onboarding-doc:
    provider: pragmatiks/pragma
    resource: file
    config: {}
```

**Behavior:**
- Create: Reads the stored file and exposes its outputs. Fails with a clear error if no file is stored under the name, and fails as already existing if the object under the name was not stored by a Pragmatiks upload (it carries no checksum); uploading the file again replaces it.
- Update: Re-reads the stored file so outputs reflect its current content.
- Observe: Uploading new content under the name leaves the configuration unchanged, so the outputs change only when the resource is next observed.
- Delete: Deletes the file from storage. Idempotent -- a missing file is treated as already deleted. An object a Pragmatiks upload did not store is left in place.

## Runtime Environment

The `pragma/file` resource reads storage configuration from environment variables populated by the Pragmatiks runtime. Standalone use is not supported.

| Variable | Purpose |
|----------|---------|
| `PRAGMA_FILE_GCS_BUCKET` | Object-store bucket holding uploaded file content |
| `PRAGMA_FILE_PUBLIC_URL` | Base URL used to compose download addresses |
| `PRAGMA_RUNTIME_ORGANIZATION_ID` | Identifies the organization's file prefix in the bucket |

## Development

```bash
# Lint and type check
mise run pragma:check

# Format
mise run pragma:format
```
