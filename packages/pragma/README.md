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

Platform-managed file: a read-only view of the file uploaded under the resource's name in its project. The file content is uploaded by `pragma resources apply` (from a `content: "@./path"` field, which it uploads under the resource's name and replaces with the file's `pragma://files/<name>` reference) or the SDK's `upload_file`. On every operation Pragmatiks gives the resource a short-lived download address for its `content` reference; the resource reads the file's size, content type, checksum and upload time from that address, without downloading the content, and exposes them as outputs. It needs no storage credentials and no environment variables.

**Config:**
- `content` -- The resource's own file, as `pragma://files/<name>` with the resource's name. A reference to any other file fails the operation.
- `content_type` -- Media type of the upload, required with `@<path>`. The CLI uses it for the upload and removes it from the applied config.

**Outputs:**
- `url` -- The file's `pragma://files/<name>` reference, which other resources' configs take.
- `size` -- File size in bytes.
- `content_type` -- Media type fixed at upload.
- `checksum` -- Hex MD5 of the file content, which storage verified when the file was stored (it matches `md5sum`).
- `uploaded_at` -- When the file's current content was stored.

```yaml
resources:
  onboarding.pdf:
    provider: pragmatiks/pragma
    resource: file
    config:
      content: "@./onboarding.pdf"
      content_type: application/pdf
```

**Behavior:**
- Create: Reads the stored file and exposes its outputs. Fails with a clear error if no file is stored under the name, or if the stored file was not stored by a Pragmatiks upload (it carries no checksum); uploading the file again replaces it. A download address that expired before the operation read it fails the operation, and applying again retries it.
- Update: Re-reads the stored file so outputs reflect its current content.
- Drift: The resource is computed, so it is not observed: uploading new content under the name, or deleting the file through the files API, leaves the outputs unchanged until the resource is applied again.
- Deactivate: Keeps the stored content.
- Delete: Deleting the resource deletes its uploaded content shortly afterwards; the provider itself never deletes. Resources that depend on the file resource fail as their dependency is deleted.

## Development

```bash
# Lint and type check
mise run pragma:check

# Format
mise run pragma:format
```
