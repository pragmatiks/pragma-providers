# Pragma Provider

Built-in platform resources for [Pragmatiks](https://pragmatiks.io).

Declaratively manage secrets, non-sensitive configuration, and file storage alongside the rest of your infrastructure. The resources in this provider are available out of the box in every organization and are commonly referenced by other providers through `FieldReference`.

## Installation

```bash
pip install pragmatiks-pragma-provider
```

or with uv:

```bash
uv add pragmatiks-pragma-provider
```

## Resources

| Resource | Type Slug | Description |
|----------|-----------|-------------|
| [Secret](#secret) | `pragma/secret` | Platform-managed secrets referenced by other resources |
| [Config](#config) | `pragma/config` | Non-sensitive configuration values shared across resources |
| [File](#file) | `pragma/file` | Platform-managed file storage with public download URLs |

---

### Secret

Stores sensitive key-value data and exposes each entry as an output so that other resources can reference individual values through `FieldReference`.

**Config:**
- `data` (`dict[str, str]`, required, mutable) -- Key-value pairs of secret data. Values must be strings.

**Outputs:**
- Each key-value pair from `data` is exposed as a separately referenceable output field (e.g., `outputs.username`, `outputs.password`).

```yaml
resources:
  database-credentials:
    provider: pragma
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
    provider: pragma
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

Platform-managed file storage. The actual file content is uploaded through the Pragmatiks API (`POST /files/{name}/upload`); this resource reads the uploaded metadata and exposes URLs, size, and checksum as outputs.

**Config:** none -- the resource tracks a file already uploaded through the Pragmatiks API under the resource's name.

**Outputs:**
- `url` -- Internal `pragma://` URL used for cross-resource references.
- `public_url` -- Public HTTP URL for external download.
- `size` -- File size in bytes.
- `content_type` -- MIME type stored at upload time.
- `checksum` -- SHA256 hash of the file content.
- `uploaded_at` -- Timestamp the file was uploaded.

```yaml
resources:
  onboarding-doc:
    provider: pragma
    resource: file
    config: {}
```

**Behavior:**
- Create: Reads the uploaded file metadata and exposes URLs, size, and checksum as outputs. Fails with a clear error if the file content has not been uploaded yet.
- Update: Re-reads the metadata from storage so outputs reflect the latest uploaded version.
- Delete: Removes both the file content and its metadata from the underlying object store. Idempotent -- missing files are treated as already deleted.

## Runtime Environment

The `pragma/file` resource reads storage configuration from environment variables populated by the Pragmatiks runtime. Standalone use is not supported.

| Variable | Purpose |
|----------|---------|
| `PRAGMA_FILE_GCS_BUCKET` | Object-store bucket holding uploaded file content |
| `PRAGMA_FILE_PUBLIC_URL` | Base URL used to compose public download URLs |
| `PRAGMA_RUNTIME_ORGANIZATION_ID` | Identifies the organization's file prefix in the bucket |

## Development

```bash
# Run tests
task pragma:test

# Lint and type check
task pragma:check

# Format
task pragma:format
```
