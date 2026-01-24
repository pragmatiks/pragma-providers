# Agno Provider for Pragmatiks

Deploys Agno AI agents to Kubernetes clusters with reactive dependency management.

## Resources

### `agno/agent`

Deploys an Agno agent as a Kubernetes Deployment with a Service.

**Config:**
- `cluster` - GKE cluster dependency (required)
- `model` - Model dependency for LLM (required)
- `embeddings` - Embeddings dependency for RAG (optional)
- `vector_store` - Vector store dependency for RAG (optional)
- `instructions` - System instructions for the agent (optional)
- `image` - Container image (default: `ghcr.io/agno-ai/agno:latest`)
- `replicas` - Number of replicas (default: 1)

**Outputs:**
- `url` - In-cluster URL for agent API
- `ready` - Whether the deployment is ready

## Usage

```yaml
apiVersion: agno/v1
kind: agent
metadata:
  name: company-assistant
  namespace: demo
spec:
  cluster: $ref{ai-cluster}
  model: $ref{claude}
  embeddings: $ref{my-embeddings}
  vector_store: $ref{qdrant-collection}
  instructions: |
    You are a helpful assistant.
```
