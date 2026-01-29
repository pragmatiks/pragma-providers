# Kubernetes Provider

Generic Kubernetes resources for Pragmatiks using [lightkube](https://github.com/gtsystem/lightkube).

## Resources

| Resource | Description |
|----------|-------------|
| `kubernetes/service` | Kubernetes Service (ClusterIP, NodePort, LoadBalancer) |
| `kubernetes/configmap` | Kubernetes ConfigMap |
| `kubernetes/secret` | Kubernetes Secret |
| `kubernetes/statefulset` | Kubernetes StatefulSet with PVC templates |

## Usage

Resources require a GKE cluster dependency for authentication:

```yaml
resources:
  my-cluster:
    provider: gcp
    resource: gke
    config:
      project_id: my-project
      location: europe-west4
      name: my-cluster
      credentials: ${{ secrets.gcp_credentials }}

  my-service:
    provider: kubernetes
    resource: service
    config:
      cluster: ${{ my-cluster }}
      namespace: default
      type: ClusterIP
      selector:
        app: my-app
      ports:
        - port: 80
          target_port: 8080
```

## Installation

```bash
pip install pragmatiks-kubernetes-provider
```
