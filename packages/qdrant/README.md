# Qdrant Provider

Manage [Qdrant](https://qdrant.tech) vector database deployments and collections through the Pragmatiks platform. Deploy self-hosted Qdrant instances to any Kubernetes cluster, or manage collections on existing Qdrant Cloud or local instances.

## Resources

| Resource | Type Slug | Description |
|----------|-----------|-------------|
| [Database](#database-qdrantdatabase) | `qdrant/database` | Self-hosted Qdrant deployment with persistent storage and LoadBalancer access |
| [Collection](#collection-qdrantcollection) | `qdrant/collection` | Vector collection for similarity search on any Qdrant instance |

## Prerequisites

- **For Database resources:** A `kubernetes/config` resource pointing at the target cluster. The cluster must support LoadBalancer services for external access.
- **For Collection resources:** A running Qdrant instance -- either a Database resource from this provider, a [Qdrant Cloud](https://cloud.qdrant.io) cluster, or any self-hosted Qdrant server accessible via HTTP.
- **API key** (optional): Required for Qdrant Cloud. For self-hosted, can be generated automatically or provided explicitly.

## Installation

```bash
pragma providers install qdrant
```

---

## Database (`qdrant/database`)

Deploys a Qdrant vector database to a Kubernetes cluster as a StatefulSet with persistent storage. Creates a headless Service for pod DNS, a StatefulSet with configurable replicas and storage, and a LoadBalancer Service for external HTTP and gRPC access.

**Config:**

| Field | Type | Required | Mutable | Default | Description |
|-------|------|----------|---------|---------|-------------|
| `config` | dependency | yes | no | -- | `kubernetes/config` resource for cluster access |
| `replicas` | int | no | yes | `1` | Number of Qdrant StatefulSet pods |
| `image` | string | no | yes | `"qdrant/qdrant:latest"` | Qdrant Docker image |
| `api_key` | string | no | yes | -- | Explicit API key for authentication (mutually exclusive with `generate_api_key`) |
| `generate_api_key` | bool | no | yes | `false` | Generate a secure 32-character hex API key |
| `storage` | object | no | yes | -- | Persistent volume configuration (see below) |
| `resources` | object | no | yes | -- | CPU and memory limits (see below) |

**StorageConfig:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `size` | string | `"10Gi"` | Persistent volume size |
| `class` | string | `"standard-rwo"` | Kubernetes storage class name |

**ResourceConfig:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `memory` | string | `"2Gi"` | Memory limit for each Qdrant pod |
| `cpu` | string | `"1"` | CPU limit for each Qdrant pod |

**Outputs:** `url`, `grpc_url`, `api_key`, `ready`

**Example:**

```yaml
resources:
  - name: my-qdrant
    provider: qdrant
    type: database
    config:
      config:
        $ref: my-kubernetes-config
      replicas: 1
      generate_api_key: true
      storage:
        size: 20Gi
        class: premium-rwo
      resources:
        memory: 4Gi
        cpu: "2"
```

**Behavior:**
- Create: Deploys headless Service, StatefulSet, and LoadBalancer Service sequentially. Waits for each to be ready and for the LoadBalancer to receive an external IP (up to 5 minutes).
- Update: Reapplies all child Kubernetes resources with updated configuration. Skips reapply if config is unchanged. Changing the `config` dependency requires delete and recreate.
- Delete: Explicitly removes child Kubernetes resources (LoadBalancer Service, StatefulSet, headless Service).
- Health: Delegates to the underlying StatefulSet health check.
- Logs: Streams pod logs from the underlying StatefulSet.

---

## Collection (`qdrant/collection`)

Manages a vector collection on any Qdrant instance for similarity search. Works with Qdrant Cloud (with API key), self-hosted instances, or Database resources from this provider.

**Config:**

| Field | Type | Required | Mutable | Default | Description |
|-------|------|----------|---------|---------|-------------|
| `url` | string | no | yes | `"http://localhost:6333"` | Qdrant server URL |
| `api_key` | string | no | yes | -- | API key for Qdrant Cloud or secured instances |
| `name` | string | yes | no | -- | Collection name within Qdrant (immutable) |
| `vectors` | object | yes | yes | -- | Vector configuration (see below) |
| `on_disk` | bool | no | yes | `false` | Store vectors on disk instead of in memory |

**VectorConfig:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `size` | int | -- | Vector dimension (must match your embedding model output) |
| `distance` | string | `"Cosine"` | Distance metric: `Cosine`, `Euclid`, or `Dot` |

**Outputs:** `name`, `indexed_vectors_count`, `points_count`, `status`

**Example (Qdrant Cloud):**

```yaml
resources:
  - name: company-docs
    provider: qdrant
    type: collection
    config:
      url: https://xyz-abc.eu-central.aws.cloud.qdrant.io:6333
      api_key:
        $ref: qdrant-api-key
      name: company-docs
      vectors:
        size: 1536
        distance: Cosine
      on_disk: true
```

**Example (self-hosted via Database resource):**

```yaml
resources:
  - name: embeddings
    provider: qdrant
    type: collection
    config:
      url: ${{ my-qdrant.outputs.url }}
      api_key: ${{ my-qdrant.outputs.api_key }}
      name: embeddings
      vectors:
        size: 768
        distance: Cosine
```

**Behavior:**
- Create: Creates the collection if it does not already exist. Idempotent -- if the collection exists, returns its current info.
- Update: Recreates the collection if vector configuration changes (size, distance, on_disk). This is destructive and deletes all existing vectors. Collection name changes are not allowed.
- Delete: Deletes the collection and all its vectors. Idempotent -- succeeds if the collection does not exist.

---

## Common Patterns

### Knowledge Base with Embeddings

Deploy a Qdrant database and create collections for a RAG pipeline:

```yaml
resources:
  # 1. GKE cluster (from GCP provider)
  - name: my-cluster
    provider: gcp
    type: gke
    config:
      project_id: my-project
      location: europe-west4
      name: my-cluster
      credentials:
        $ref: gcp-credentials

  # 2. Kubernetes config authenticating against the GKE cluster
  - name: my-kubernetes-config
    provider: kubernetes
    type: config
    config:
      mode: gke_cluster
      cluster:
        $ref: my-cluster

  # 3. Self-hosted Qdrant database
  - name: vector-db
    provider: qdrant
    type: database
    config:
      config:
        $ref: my-kubernetes-config
      generate_api_key: true
      storage:
        size: 50Gi

  # 3. Collection for document embeddings
  - name: documents
    provider: qdrant
    type: collection
    config:
      url: ${{ vector-db.outputs.url }}
      api_key: ${{ vector-db.outputs.api_key }}
      name: documents
      vectors:
        size: 1536
        distance: Cosine
      on_disk: true

  # 4. Collection for FAQ embeddings (smaller model)
  - name: faq
    provider: qdrant
    type: collection
    config:
      url: ${{ vector-db.outputs.url }}
      api_key: ${{ vector-db.outputs.api_key }}
      name: faq
      vectors:
        size: 384
        distance: Cosine
```

### Qdrant Cloud with Multiple Collections

Use Qdrant Cloud instead of self-hosting:

```yaml
resources:
  - name: product-search
    provider: qdrant
    type: collection
    config:
      url: https://my-cluster.eu-central.aws.cloud.qdrant.io:6333
      api_key:
        $ref: qdrant-cloud-key
      name: products
      vectors:
        size: 768
        distance: Dot

  - name: image-search
    provider: qdrant
    type: collection
    config:
      url: https://my-cluster.eu-central.aws.cloud.qdrant.io:6333
      api_key:
        $ref: qdrant-cloud-key
      name: images
      vectors:
        size: 512
        distance: Cosine
```

## Configuration

### API Keys

Qdrant supports optional API key authentication. For the Database resource, you have three options:

1. **No authentication** -- omit both `api_key` and `generate_api_key` (suitable for development)
2. **Generated key** -- set `generate_api_key: true` to auto-generate a secure 32-character hex key
3. **Explicit key** -- set `api_key` to a specific value (useful when rotating keys or matching existing config)

For Collection resources connecting to Qdrant Cloud, the `api_key` field is required and should reference a secret.

### Vector Dimensions

Choose the `vectors.size` to match your embedding model:

| Embedding Model | Dimensions |
|----------------|------------|
| OpenAI `text-embedding-3-large` | 3072 |
| OpenAI `text-embedding-3-small` | 1536 |
| OpenAI `text-embedding-ada-002` | 1536 |
| Cohere `embed-english-v3.0` | 1024 |
| Sentence Transformers (all-MiniLM-L6-v2) | 384 |
| Google `text-embedding-004` | 768 |

### Distance Metrics

| Metric | Best For |
|--------|----------|
| `Cosine` | Text embeddings, normalized vectors (most common) |
| `Euclid` | Spatial data, when absolute distance matters |
| `Dot` | Pre-normalized embeddings, maximum inner product search |

## Development

```bash
# Run tests
task qdrant:test

# Lint and type check
task qdrant:check

# Format
task qdrant:format
```
