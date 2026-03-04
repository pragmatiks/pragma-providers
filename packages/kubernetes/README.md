# Kubernetes Provider

Generic Kubernetes resource management for [Pragmatiks](https://pragmatiks.io) using [lightkube](https://github.com/gtsystem/lightkube).

Declaratively manage Kubernetes workloads, networking, configuration, and cluster-scoped resources. The provider authenticates via a GKE cluster dependency and uses server-side apply for idempotent operations.

## Prerequisites

- A running Kubernetes cluster managed by the GCP provider (`gcp/gke`)
- GCP service account credentials with `container.clusters.get` and `container.clusters.getCredentials` permissions
- RBAC permissions on the target cluster for the resources you want to manage (typically `cluster-admin` or namespace-scoped roles)

## Installation

```bash
pragma providers install kubernetes
```

## Resources

| Resource | Type Slug | Description |
|----------|-----------|-------------|
| [Namespace](#namespace) | `kubernetes/namespace` | Cluster-scoped namespace isolation |
| [Deployment](#deployment) | `kubernetes/deployment` | Stateless workload with rolling updates |
| [StatefulSet](#statefulset) | `kubernetes/statefulset` | Stateful workload with persistent storage and stable pod identity |
| [Service](#service) | `kubernetes/service` | Network exposure (ClusterIP, NodePort, LoadBalancer, Headless) |
| [ConfigMap](#configmap) | `kubernetes/configmap` | Non-sensitive configuration data |
| [Secret](#secret) | `kubernetes/secret` | Sensitive data (credentials, tokens, TLS certs) |

All resources require a `cluster` dependency pointing to a `gcp/gke` resource for authentication.

---

### Namespace

Cluster-scoped resource for workload isolation. Namespaces do not belong to another namespace.

**Config:**
- `cluster` (dependency) -- GKE cluster for authentication
- `labels` (dict, optional) -- Labels to apply to the namespace

**Outputs:**
- `name` -- Namespace name

```yaml
resources:
  dev-namespace:
    provider: kubernetes
    resource: namespace
    config:
      cluster: ${{ my-cluster }}
      labels:
        environment: development
        team: platform
```

---

### Deployment

Manages stateless workloads with configurable replicas, rolling update strategy, health probes, environment variables, and resource limits. Waits for all replicas to be ready before reporting success (default timeout: 300s).

**Config:**
- `cluster` (dependency) -- GKE cluster for authentication
- `namespace` (string, default: `"default"`) -- Target namespace (immutable)
- `replicas` (int, default: `1`) -- Desired pod replicas
- `selector` (dict) -- Label selector for pods (immutable)
- `labels` (dict, optional) -- Pod labels; defaults to `selector` if not set
- `containers` (list) -- Container specs: image, ports, env, probes, resources
- `strategy` (`"RollingUpdate"` | `"Recreate"`, default: `"RollingUpdate"`) -- Update strategy

**Outputs:**
- `name`, `namespace`, `replicas`, `ready_replicas`, `available_replicas`

```yaml
resources:
  api-deployment:
    provider: kubernetes
    resource: deployment
    config:
      cluster: ${{ my-cluster }}
      namespace: production
      replicas: 3
      selector:
        app: api
      containers:
        - name: api
          image: gcr.io/my-project/api:latest
          ports:
            - container_port: 8080
              name: http
          env:
            LOG_LEVEL: info
          env_from_secret:
            DATABASE_URL: db-credentials.url
          resources:
            cpu: "250m"
            memory: "512Mi"
            cpu_limit: "1000m"
            memory_limit: "1Gi"
          readiness_probe:
            http_get:
              path: /healthz
              port: 8080
            initial_delay_seconds: 5
            period_seconds: 10
```

---

### StatefulSet

Manages stateful workloads with stable pod identity, persistent storage via PVC templates, and ordered deployment. Associates with a headless service for DNS-based pod discovery. Waits for all replicas to be ready before reporting success.

**Config:**
- `cluster` (dependency) -- GKE cluster for authentication
- `namespace` (string, default: `"default"`) -- Target namespace (immutable)
- `replicas` (int, default: `1`) -- Desired pod replicas
- `service_name` (string) -- Headless service for pod DNS (immutable)
- `selector` (dict, optional) -- Label selector; defaults to `{"app": "<name>"}`
- `containers` (list) -- Container specs: image, ports, env, volume mounts, probes
- `volume_claim_templates` (list, optional) -- PVC templates for persistent storage

**Outputs:**
- `name`, `namespace`, `replicas`, `ready_replicas`, `service_name`

```yaml
resources:
  postgres:
    provider: kubernetes
    resource: statefulset
    config:
      cluster: ${{ my-cluster }}
      namespace: data
      replicas: 3
      service_name: postgres-headless
      containers:
        - name: postgres
          image: postgres:16
          ports:
            - container_port: 5432
              name: postgres
          env:
            - name: POSTGRES_DB
              value: myapp
          volume_mounts:
            - name: data
              mount_path: /var/lib/postgresql/data
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
          readiness_probe:
            tcp_socket_port: 5432
            initial_delay_seconds: 15
            period_seconds: 10
      volume_claim_templates:
        - name: data
          storage_class: premium-rwo
          access_modes:
            - ReadWriteOnce
          storage: 50Gi
```

---

### Service

Exposes workloads via ClusterIP, NodePort, LoadBalancer, or Headless service types. Services are immediately ready after apply (no polling). Headless services automatically set `clusterIP: None`.

**Config:**
- `cluster` (dependency) -- GKE cluster for authentication
- `namespace` (string, default: `"default"`) -- Target namespace (immutable)
- `type` (`"ClusterIP"` | `"NodePort"` | `"LoadBalancer"` | `"Headless"`, default: `"ClusterIP"`) -- Service type
- `selector` (dict) -- Label selector for target pods
- `ports` (list) -- Port mappings: port, target_port, protocol, name
- `cluster_ip` (string, optional) -- Explicit cluster IP

**Outputs:**
- `name`, `namespace`, `cluster_ip`, `type`

```yaml
resources:
  api-service:
    provider: kubernetes
    resource: service
    config:
      cluster: ${{ my-cluster }}
      namespace: production
      type: ClusterIP
      selector:
        app: api
      ports:
        - name: http
          port: 80
          target_port: 8080

  postgres-headless:
    provider: kubernetes
    resource: service
    config:
      cluster: ${{ my-cluster }}
      namespace: data
      type: Headless
      selector:
        app: postgres
      ports:
        - name: postgres
          port: 5432
```

---

### ConfigMap

Stores non-sensitive configuration data as key-value pairs. ConfigMaps can be mounted as files or exposed as environment variables in pods.

**Config:**
- `cluster` (dependency) -- GKE cluster for authentication
- `namespace` (string, default: `"default"`) -- Target namespace (immutable)
- `data` (dict) -- Key-value pairs to store

**Outputs:**
- `name`, `namespace`, `data`

```yaml
resources:
  app-config:
    provider: kubernetes
    resource: configmap
    config:
      cluster: ${{ my-cluster }}
      namespace: production
      data:
        APP_ENV: production
        LOG_FORMAT: json
        MAX_CONNECTIONS: "100"
```

---

### Secret

Stores sensitive data (credentials, tokens, TLS certificates). Data values are automatically base64-encoded. Supports both pre-encoded `data` and plain-text `string_data` fields.

**Config:**
- `cluster` (dependency) -- GKE cluster for authentication
- `namespace` (string, default: `"default"`) -- Target namespace (immutable)
- `type` (string, default: `"Opaque"`) -- Secret type (e.g., `Opaque`, `kubernetes.io/tls`)
- `data` (dict, optional) -- Key-value pairs (will be base64-encoded)
- `string_data` (dict, optional) -- Plain-text key-value pairs (Kubernetes encodes them)

**Outputs:**
- `name`, `namespace`, `type`, `data`

```yaml
resources:
  db-credentials:
    provider: kubernetes
    resource: secret
    config:
      cluster: ${{ my-cluster }}
      namespace: production
      type: Opaque
      string_data:
        url: postgresql://user:pass@postgres:5432/myapp
        username: user
        password: pass
```

---

## Cross-Provider Usage

The Kubernetes provider is designed to work alongside the GCP provider. A typical pattern is: GCP provisions the cluster, Kubernetes deploys workloads into it.

```yaml
resources:
  # GCP creates the cluster
  my-cluster:
    provider: gcp
    resource: gke
    config:
      project_id: my-project
      location: europe-west4
      name: prod-cluster
      credentials: ${{ secrets.gcp_credentials }}

  # Kubernetes resources depend on the cluster
  app-namespace:
    provider: kubernetes
    resource: namespace
    config:
      cluster: ${{ my-cluster }}
      labels:
        environment: production

  app-config:
    provider: kubernetes
    resource: configmap
    config:
      cluster: ${{ my-cluster }}
      namespace: ${{ app-namespace.name }}
      data:
        APP_ENV: production

  app-secrets:
    provider: kubernetes
    resource: secret
    config:
      cluster: ${{ my-cluster }}
      namespace: ${{ app-namespace.name }}
      string_data:
        api_key: ${{ secrets.api_key }}

  app:
    provider: kubernetes
    resource: deployment
    config:
      cluster: ${{ my-cluster }}
      namespace: ${{ app-namespace.name }}
      replicas: 3
      selector:
        app: my-app
      containers:
        - name: app
          image: gcr.io/my-project/app:latest
          ports:
            - container_port: 8080
          env_from_secret:
            API_KEY: app-secrets.api_key

  app-service:
    provider: kubernetes
    resource: service
    config:
      cluster: ${{ my-cluster }}
      namespace: ${{ app-namespace.name }}
      type: LoadBalancer
      selector:
        app: my-app
      ports:
        - port: 80
          target_port: 8080
```

Resources are applied in dependency order. The platform resolves `${{ my-cluster }}` and `${{ app-namespace.name }}` references automatically, ensuring the GKE cluster is ready before any Kubernetes resources are created.
