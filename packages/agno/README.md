# Agno Provider for Pragmatiks

Deploys [Agno](https://docs.agno.com) AI agents and teams to Kubernetes with reactive dependency management. Define models, tools, knowledge bases, and memory as declarative resources -- the platform handles wiring, deployment, and change propagation automatically.

## Architecture

The Agno provider follows a **spec pattern**: most resources are stateless configuration wrappers that produce serializable specifications. Only the `runner` resource creates actual infrastructure.

```
models/openai ──┐
models/anthropic┤
                ├─→ agent ──┐
tools/mcp ──────┤           ├─→ runner ──→ Kubernetes Deployment + Service
tools/websearch─┤           │
knowledge ──────┤     team ─┘
  vectordb/qdrant       │
  knowledge/content     │
  knowledge/embedder    │
memory/manager ─┤       │
db/postgres ────┤       │
prompt ─────────┘       │
                        │
cluster (gcp/gke) ──────┘
namespace (k8s) ────────┘
```

**How it works:**

1. Configuration resources (models, tools, knowledge, etc.) resolve their dependencies and produce a serializable `Spec`
2. The `agent` or `team` resource aggregates all specs from its dependencies into a single `AgentSpec` or `TeamSpec`
3. The `runner` resource deploys the agent/team to Kubernetes by passing the spec as JSON environment variables
4. The container image reconstructs the full agent/team from the spec at startup using `from_spec()` factory methods
5. When any dependency changes (e.g., a model API key rotates), Pragma propagates the change through the dependency graph and redeploys automatically

## Prerequisites

- A Kubernetes cluster managed by the `gcp` provider (`gcp/gke` resource)
- A Kubernetes namespace managed by the `kubernetes` provider
- API keys for your chosen model provider (OpenAI, Anthropic)
- An Agno runner container image (default: `ghcr.io/pragmatiks/agno-runner:latest`)
- For knowledge/RAG: a Qdrant vector database instance
- For memory/sessions: a PostgreSQL database instance

## Installation

```bash
pragma providers install agno
```

## Resources

| Resource | Type Slug | Description |
|----------|-----------|-------------|
| Agent | `agent` | AI agent definition with model, tools, knowledge, and memory |
| Team | `team` | Coordinated group of agents with shared resources |
| Runner | `runner` | Deploys an agent or team to Kubernetes as a Deployment + Service |
| Prompt | `prompt` | Reusable instruction template with variable interpolation |
| OpenAI Model | `models/openai` | OpenAI model configuration (GPT-4o, etc.) |
| Anthropic Model | `models/anthropic` | Anthropic model configuration (Claude, etc.) |
| MCP Tools | `tools/mcp` | Model Context Protocol server integration (stdio, SSE, streamable-http) |
| Web Search Tools | `tools/websearch` | Web and news search toolkit (DuckDuckGo, Google, Bing, etc.) |
| Knowledge | `knowledge` | Semantic search configuration backed by a vector database |
| Content | `knowledge/content` | Content source (URL or text) for ingestion into a knowledge base |
| OpenAI Embedder | `knowledge/embedder/openai` | OpenAI embedding model configuration |
| Qdrant VectorDB | `vectordb/qdrant` | Qdrant vector database adapter for Agno knowledge |
| Memory Manager | `memory/manager` | Agent memory management with PostgreSQL storage |
| PostgreSQL DB | `db/postgres` | PostgreSQL database connection for sessions, memory, and storage |

## Example: Full Agent Deployment

A realistic example showing a model, tools, knowledge base, and agent deployed to Kubernetes.

```yaml
# 1. Model
provider: agno
resource: models/anthropic
name: claude
config:
  id: claude-sonnet-4-20250514
  api_key:
    ref: secrets/anthropic-key
    field: value

---
# 2. MCP tool server
provider: agno
resource: tools/mcp
name: search-tool
config:
  url: http://mcp-search.tools.svc.cluster.local/sse
  transport: sse

---
# 3. Web search tool
provider: agno
resource: tools/websearch
name: web-search
config:
  backend: duckduckgo
  enable_news: true

---
# 4. Embedder for knowledge
provider: agno
resource: knowledge/embedder/openai
name: embedder
config:
  id: text-embedding-3-small
  api_key:
    ref: secrets/openai-key
    field: value

---
# 5. Vector database adapter
provider: agno
resource: vectordb/qdrant
name: doc-vectors
config:
  url:
    ref: qdrant/database/main
    field: url
  collection:
    ref: qdrant/collection/docs
    field: name
  api_key:
    ref: qdrant/database/main
    field: api_key
  search_type: hybrid
  embedder:
    ref: agno/knowledge/embedder/openai/embedder

---
# 6. Knowledge base
provider: agno
resource: knowledge
name: docs-kb
config:
  vector_db:
    ref: agno/vectordb/qdrant/doc-vectors
  max_results: 5

---
# 7. Content ingestion
provider: agno
resource: knowledge/content
name: product-docs
config:
  knowledge:
    ref: agno/knowledge/docs-kb
  url: https://docs.example.com/product
  description: Product documentation

---
# 8. Database for sessions and memory
provider: agno
resource: db/postgres
name: agent-db
config:
  connection_url:
    ref: cloudsql/instance/main
    field: connection_url
  username:
    ref: secrets/db-creds
    field: username
  password:
    ref: secrets/db-creds
    field: password

---
# 9. Memory manager
provider: agno
resource: memory/manager
name: agent-memory
config:
  db:
    ref: agno/db/postgres/agent-db
  add_memories: true
  update_memories: true

---
# 10. Prompt template
provider: agno
resource: prompt
name: system-prompt
config:
  template: |
    You are {{role}}, a helpful assistant for {{company}}.
    Always be concise and accurate.
  variables:
    role: Senior Support Engineer
    company: Acme Corp

---
# 11. Agent definition
provider: agno
resource: agent
name: support-agent
config:
  model:
    ref: agno/models/anthropic/claude
  tools:
    - ref: agno/tools/mcp/search-tool
    - ref: agno/tools/websearch/web-search
  knowledge:
    ref: agno/knowledge/docs-kb
  memory:
    ref: agno/memory/manager/agent-memory
  db:
    ref: agno/db/postgres/agent-db
  prompt:
    ref: agno/prompt/system-prompt
  markdown: true
  enable_agentic_memory: true

---
# 12. Deploy to Kubernetes
provider: agno
resource: runner
name: support-agent
config:
  agent:
    ref: agno/agent/support-agent
  cluster:
    ref: gcp/gke/main-cluster
  namespace:
    ref: kubernetes/namespace/agents
  replicas: 2
  cpu: 500m
  memory: 2Gi
  security_key: my-secret-key
```

## Common Patterns

### Team of Agents

Compose multiple specialized agents into a coordinated team.

```yaml
provider: agno
resource: agent
name: researcher
config:
  model:
    ref: agno/models/openai/gpt4o
  tools:
    - ref: agno/tools/websearch/search
  instructions:
    - "You are a research specialist. Find and summarize information."

---
provider: agno
resource: agent
name: writer
config:
  model:
    ref: agno/models/anthropic/claude
  instructions:
    - "You are a technical writer. Create clear, structured content."

---
provider: agno
resource: team
name: content-team
config:
  members:
    - ref: agno/agent/researcher
    - ref: agno/agent/writer
  model:
    ref: agno/models/anthropic/claude
  delegate_to_all_members: true
  markdown: true

---
provider: agno
resource: runner
name: content-team
config:
  team:
    ref: agno/team/content-team
  cluster:
    ref: gcp/gke/main-cluster
  namespace:
    ref: kubernetes/namespace/agents
```

### Knowledge-Augmented Agent (RAG)

An agent with access to a vector knowledge base for semantic search.

```yaml
provider: agno
resource: vectordb/qdrant
name: kb-vectors
config:
  url: http://qdrant.databases.svc.cluster.local:6333
  collection: knowledge
  search_type: hybrid

---
provider: agno
resource: knowledge
name: product-kb
config:
  vector_db:
    ref: agno/vectordb/qdrant/kb-vectors
  max_results: 10

---
provider: agno
resource: knowledge/content
name: faq
config:
  knowledge:
    ref: agno/knowledge/product-kb
  text_content: |
    Q: What are the supported regions?
    A: US-East, EU-West, and APAC.

---
provider: agno
resource: agent
name: support-bot
config:
  model:
    ref: agno/models/openai/gpt4o
  knowledge:
    ref: agno/knowledge/product-kb
  instructions:
    - "Answer questions using the knowledge base. Cite sources."
```

### Multi-Tool Agent

An agent with multiple tool integrations.

```yaml
provider: agno
resource: tools/mcp
name: github-mcp
config:
  command: npx -y @modelcontextprotocol/server-github
  env:
    GITHUB_TOKEN:
      ref: secrets/github-token
      field: value

---
provider: agno
resource: tools/mcp
name: slack-mcp
config:
  url: http://mcp-slack.tools.svc.cluster.local/sse
  transport: sse
  headers:
    Authorization: "Bearer my-token"
  include_tools:
    - send_message
    - read_channel

---
provider: agno
resource: tools/websearch
name: web
config:
  backend: auto
  enable_news: true

---
provider: agno
resource: agent
name: ops-agent
config:
  model:
    ref: agno/models/anthropic/claude
  tools:
    - ref: agno/tools/mcp/github-mcp
    - ref: agno/tools/mcp/slack-mcp
    - ref: agno/tools/websearch/web
  instructions:
    - "You are an operations assistant with access to GitHub, Slack, and web search."
```
