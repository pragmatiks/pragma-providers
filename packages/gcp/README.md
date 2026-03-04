# GCP Provider

Manage Google Cloud Platform resources declaratively through the Pragmatiks platform. This provider uses user-provided service account credentials (multi-tenant SaaS pattern) to create and manage GCP infrastructure.

## Supported Services

| Service | Resource | Description |
|---------|----------|-------------|
| Secret Manager | `gcp/secret` | Store and version sensitive data |
| GKE | `gcp/gke` | Kubernetes clusters (Autopilot and Standard) |
| Cloud SQL | `gcp/cloudsql/database_instance` | Managed database instances (PostgreSQL, MySQL, SQL Server) |
| Cloud SQL | `gcp/cloudsql/database` | Databases within a Cloud SQL instance |
| Cloud SQL | `gcp/cloudsql/user` | Database users within a Cloud SQL instance |

## Prerequisites

1. **GCP Project** with billing enabled.
2. **Service account** with appropriate IAM roles per resource:
   - Secret Manager: `roles/secretmanager.admin`
   - GKE: `roles/container.admin`
   - Cloud SQL: `roles/cloudsql.admin`
3. **GCP APIs enabled** on the project:
   - `secretmanager.googleapis.com`
   - `container.googleapis.com`
   - `sqladmin.googleapis.com`
   - `logging.googleapis.com` (for health checks and log streaming)
4. **Service account key** exported as JSON (passed via `credentials` field on each resource).

## Installation

```bash
pragma providers install gcp
```

## Resources

### Secret (`gcp/secret`)

Manages secrets in GCP Secret Manager. Creates versioned secrets with automatic replication.

**Config fields:**

| Field | Type | Required | Mutable | Description |
|-------|------|----------|---------|-------------|
| `project_id` | string | yes | no | GCP project ID |
| `secret_id` | string | yes | no | Secret identifier (unique per project) |
| `data` | string | yes | yes | Secret payload to store |
| `credentials` | object/string | yes | yes | GCP service account credentials JSON |

**Outputs:** `resource_name`, `version_name`, `version_id`

**Example:**

```yaml
resources:
  - name: api-key
    provider: gcp
    type: secret
    config:
      project_id: my-gcp-project
      secret_id: api-key
      data: "sk-my-secret-value"
      credentials:
        $ref: gcp-credentials
```

**Behavior:**
- Create: Creates the secret and an initial version. Idempotent -- if the secret already exists, adds a new version.
- Update: Adds a new secret version when `data` changes. Previous versions are retained.
- Delete: Deletes the secret and all its versions.

---

### GKE Cluster (`gcp/gke`)

Manages GKE clusters in either Autopilot (default) or Standard mode. Includes health checks and log streaming from Cloud Logging.

**Config fields:**

| Field | Type | Required | Mutable | Default | Description |
|-------|------|----------|---------|---------|-------------|
| `project_id` | string | yes | no | -- | GCP project ID |
| `credentials` | object/string | yes | yes | -- | GCP service account credentials JSON |
| `location` | string | yes | no | -- | Region or zone (e.g., `europe-west4`) |
| `name` | string | yes | no | -- | Cluster name (lowercase, 1-40 chars) |
| `autopilot` | bool | no | no | `true` | Use Autopilot mode |
| `network` | string | no | no | `"default"` | VPC network name |
| `subnetwork` | string | no | yes | -- | VPC subnetwork name |
| `release_channel` | string | no | yes | `"REGULAR"` | Release channel: `RAPID`, `REGULAR`, `STABLE` |
| `initial_node_count` | int | no | yes | `1` | Nodes in default pool (Standard only) |
| `machine_type` | string | no | yes | `"e2-medium"` | Node machine type (Standard only) |
| `disk_size_gb` | int | no | yes | `100` | Boot disk size in GB (Standard only) |

**Outputs:** `name`, `endpoint`, `cluster_ca_certificate`, `location`, `status`, `console_url`, `logs_url`

**Example (Autopilot):**

```yaml
resources:
  - name: prod-cluster
    provider: gcp
    type: gke
    config:
      project_id: my-gcp-project
      location: europe-west4
      name: prod-cluster
      autopilot: true
      release_channel: STABLE
      credentials:
        $ref: gcp-credentials
```

**Example (Standard):**

```yaml
resources:
  - name: dev-cluster
    provider: gcp
    type: gke
    config:
      project_id: my-gcp-project
      location: europe-west4-a
      name: dev-cluster
      autopilot: false
      initial_node_count: 3
      machine_type: e2-standard-4
      disk_size_gb: 200
      credentials:
        $ref: gcp-credentials
```

**Behavior:**
- Create: Creates the cluster and polls until it reaches RUNNING state (up to 20 minutes). Idempotent -- if the cluster already exists, waits for RUNNING.
- Update: Returns current cluster state. Immutable fields (name, location, autopilot, network) require delete and recreate.
- Delete: Deletes the cluster and polls until fully removed.
- Health: Reports `healthy` (RUNNING), `degraded` (PROVISIONING/RECONCILING), or `unhealthy` (ERROR/not found).
- Logs: Streams cluster logs from Cloud Logging.

---

### Cloud SQL Database Instance (`gcp/cloudsql/database_instance`)

Manages Cloud SQL instances for PostgreSQL, MySQL, and SQL Server. Supports configurable tiers, high availability, backups, and network access. Includes health checks and log streaming.

**Config fields:**

| Field | Type | Required | Mutable | Default | Description |
|-------|------|----------|---------|---------|-------------|
| `project_id` | string | yes | no | -- | GCP project ID |
| `credentials` | object/string | yes | yes | -- | GCP service account credentials JSON |
| `region` | string | yes | no | -- | GCP region (e.g., `europe-west4`) |
| `instance_name` | string | yes | no | -- | Instance name (unique per project, 1-98 chars) |
| `database_version` | string | no | no | `"POSTGRES_15"` | Engine version (e.g., `POSTGRES_15`, `MYSQL_8_0`) |
| `tier` | string | no | yes | `"db-f1-micro"` | Machine tier (e.g., `db-custom-1-3840`) |
| `availability_type` | string | no | yes | `"ZONAL"` | `ZONAL` or `REGIONAL` (high availability) |
| `backup_enabled` | bool | no | yes | `true` | Enable automatic backups |
| `deletion_protection` | bool | no | yes | `false` | Prevent accidental deletion |
| `authorized_networks` | list[string] | no | yes | `[]` | CIDR ranges allowed to connect |
| `enable_public_ip` | bool | no | yes | `true` | Assign a public IP address |

**Outputs:** `connection_name`, `public_ip`, `private_ip`, `ready`, `console_url`, `logs_url`

**Example:**

```yaml
resources:
  - name: prod-db-instance
    provider: gcp
    type: cloudsql/database_instance
    config:
      project_id: my-gcp-project
      region: europe-west4
      instance_name: prod-postgres
      database_version: POSTGRES_15
      tier: db-custom-2-7680
      availability_type: REGIONAL
      backup_enabled: true
      deletion_protection: true
      authorized_networks:
        - "10.0.0.0/8"
      credentials:
        $ref: gcp-credentials
```

**Behavior:**
- Create: Creates the instance and polls until RUNNABLE (up to 15 minutes). Generates a random root password. Idempotent.
- Update: Patches mutable settings (tier, availability, backups, network config) and waits for RUNNABLE.
- Delete: Deletes the instance. Respects `deletion_protection` -- disable it first to allow deletion.
- Health: Reports `healthy` (RUNNABLE), `degraded` (PENDING_CREATE/MAINTENANCE), or `unhealthy`.
- Logs: Streams instance logs from Cloud Logging.

---

### Cloud SQL Database (`gcp/cloudsql/database`)

Creates a database within a Cloud SQL instance. Requires a dependency on a `gcp/cloudsql/database_instance` resource.

**Config fields:**

| Field | Type | Required | Mutable | Description |
|-------|------|----------|---------|-------------|
| `instance` | Dependency | yes | yes | Reference to a `cloudsql/database_instance` resource |
| `database_name` | string | yes | no | Name of the database to create |

**Outputs:** `database_name`, `instance_name`, `host`, `port`, `url`

**Example:**

```yaml
resources:
  - name: prod-db-instance
    provider: gcp
    type: cloudsql/database_instance
    config:
      project_id: my-gcp-project
      region: europe-west4
      instance_name: prod-postgres
      credentials:
        $ref: gcp-credentials

  - name: app-database
    provider: gcp
    type: cloudsql/database
    config:
      instance:
        $ref: prod-db-instance
      database_name: myapp
```

**Behavior:**
- Create: Creates the database in the target instance. Idempotent.
- Update: If the instance dependency changes, deletes from the old instance and creates in the new one.
- Delete: Drops the database from the instance.
- Outputs include a connection URL in the format `postgresql://host:port/database_name`.

---

### Cloud SQL User (`gcp/cloudsql/user`)

Creates a database user within a Cloud SQL instance. Requires a dependency on a `gcp/cloudsql/database_instance` resource.

**Config fields:**

| Field | Type | Required | Mutable | Description |
|-------|------|----------|---------|-------------|
| `instance` | Dependency | yes | yes | Reference to a `cloudsql/database_instance` resource |
| `username` | string | yes | no | Database username |
| `password` | string | yes | yes | Database password |

**Outputs:** `username`, `instance_name`, `host`

**Example:**

```yaml
resources:
  - name: prod-db-instance
    provider: gcp
    type: cloudsql/database_instance
    config:
      project_id: my-gcp-project
      region: europe-west4
      instance_name: prod-postgres
      credentials:
        $ref: gcp-credentials

  - name: app-user
    provider: gcp
    type: cloudsql/user
    config:
      instance:
        $ref: prod-db-instance
      username: app_service
      password:
        $ref: db-password-secret
```

**Behavior:**
- Create: Creates the user in the target instance. Idempotent.
- Update: Password changes are applied in-place. If the instance dependency changes, deletes from the old instance and creates in the new one.
- Delete: Drops the user from the instance.

---

## Full Stack Example

A complete Cloud SQL setup with instance, database, user, and credentials stored in Secret Manager:

```yaml
resources:
  - name: db-password
    provider: gcp
    type: secret
    config:
      project_id: my-gcp-project
      secret_id: db-password
      data: "my-secure-password"
      credentials:
        $ref: gcp-credentials

  - name: prod-instance
    provider: gcp
    type: cloudsql/database_instance
    config:
      project_id: my-gcp-project
      region: europe-west4
      instance_name: prod-postgres
      database_version: POSTGRES_15
      tier: db-custom-2-7680
      availability_type: REGIONAL
      backup_enabled: true
      credentials:
        $ref: gcp-credentials

  - name: app-db
    provider: gcp
    type: cloudsql/database
    config:
      instance:
        $ref: prod-instance
      database_name: myapp

  - name: app-user
    provider: gcp
    type: cloudsql/user
    config:
      instance:
        $ref: prod-instance
      username: app_service
      password:
        $ref: db-password.outputs.version_id
```

## Credentials

All resources require a `credentials` field containing GCP service account credentials as either a JSON object or a JSON string. In production, use a `$ref` to a secret resource to inject credentials securely.

The provider uses explicit credentials (not Application Default Credentials) to support multi-tenant deployments where each user operates in their own GCP project.

## Development

```bash
# Run tests
task gcp:test

# Lint and type check
task gcp:check

# Format
task gcp:format
```
