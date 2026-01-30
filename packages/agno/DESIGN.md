# Agno Provider Design

This document describes the complete resource architecture for the Agno provider, enabling users to build full agentic solutions using the Pragmatiks paradigm.

## Design Principles

1. **Mirror Agno SDK** - Resource paths follow Agno's Python import paths for familiarity
2. **Layered Architecture** - Infrastructure resources wrap with Agno-specific resources
3. **Flexibility** - Support both inline config and separate resources where it makes sense
4. **Dependency Propagation** - Changes flow through the resource graph automatically

## Resource Architecture

```
                              agno/team
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
               agno/agent    agno/agent    agno/agent
                    │
    ┌───────┬───────┼───────┬───────┬───────┐
    ▼       ▼       ▼       ▼       ▼       ▼
agno/     agno/   agno/   agno/   agno/   agno/
models/   prompt  tools/  tools/  knowledge memory/
anthropic         function mcp             manager
    │                       │       │       │
    ▼                       ▼       ▼       ▼
pragma/                  pragma/ agno/    agno/
secret                   secret  vectordb/ db/
                                 qdrant   postgres
                                    │       │
                                    ▼       ▼
                                 qdrant/ postgresql/
                                collection database
```

## Complete Resource Catalog

### Core Agent Resources

#### `agno/agent`
Deploys an Agno agent as a Kubernetes pod with FastAPI service.

```yaml
provider: agno
resource: agent
name: research-assistant
config:
  # Identity
  name: "Research Assistant"
  role: "Information gathering and synthesis"

  # Model (required) - reference to model resource
  model:
    $ref: agno/models/anthropic/claude-sonnet

  # Instructions - inline OR reference
  instructions:
    - "You are a research specialist"
    - "Always cite sources"
  # OR
  prompt:
    $ref: agno/prompt/researcher

  # Tools (optional) - unified list
  tools:
    - $ref: agno/tools/function/web-search
    - $ref: agno/tools/mcp/github

  # Knowledge (optional)
  knowledge:
    $ref: agno/knowledge/product-docs

  # Storage (optional)
  db:
    $ref: agno/db/postgres/agent-storage

  # Memory (optional)
  memory_manager:
    $ref: agno/memory/manager/custom
  update_memory_on_run: true
  # OR simpler:
  enable_agentic_memory: true

  # Deployment
  cluster:
    $ref: gcp/gke/main
  replicas: 1
```

**Outputs:**
- `url`: In-cluster service URL
- `ready`: Deployment status

---

#### `agno/team`
Multi-agent orchestration with configurable collaboration modes.

```yaml
provider: agno
resource: team
name: content-team
config:
  name: "Content Creation Team"

  # Members (required)
  members:
    - $ref: agno/agent/researcher
    - $ref: agno/agent/writer
    - $ref: agno/agent/editor

  # Collaboration mode
  mode: coordinate  # route | coordinate | collaborate

  # Instructions - inline OR reference (same as agent)
  instructions:
    - "Researcher gathers information first"
    - "Writer creates draft based on research"
    - "Editor polishes final output"

  # Model for leader/coordinator (optional)
  model:
    $ref: agno/models/anthropic/claude-sonnet

  # Deployment
  cluster:
    $ref: gcp/gke/main
```

**Collaboration Modes:**
| Mode | Behavior |
|------|----------|
| `route` | Leader analyzes task, routes to most appropriate agent |
| `coordinate` | Sequential: each agent's output becomes next agent's input |
| `collaborate` | Parallel: all agents tackle task, leader synthesizes |

**Outputs:**
- `url`: Team endpoint URL
- `member_count`: Number of agents
- `ready`: All members deployed

---

#### `agno/prompt`
Reusable instruction templates. *Pragmatiks addition, not in Agno SDK.*

```yaml
provider: agno
resource: prompt
name: researcher
config:
  instructions:
    - "You are a research specialist"
    - "Always cite your sources"
    - "Be thorough and accurate"

  # Optional: variable interpolation
  variables:
    domain: "technology"
    tone: "professional"

  template: |
    Focus on {{domain}} topics.
    Maintain a {{tone}} tone.
```

**Outputs:**
- `text`: Rendered prompt text
- `instruction_count`: Number of instruction lines

---

### Model Resources

#### `agno/models/anthropic`
Anthropic Claude model configuration.

```yaml
provider: agno
resource: models/anthropic
name: claude-sonnet
config:
  id: claude-sonnet-4-20250514
  api_key:
    $ref: pragma/secret/anthropic-key.value

  # Optional parameters
  max_tokens: 4096
  temperature: 0.7
```

**Outputs:**
- `provider`: "anthropic"
- `model_id`: Model identifier
- `ready`: API key validated

---

#### `agno/models/openai`
OpenAI model configuration.

```yaml
provider: agno
resource: models/openai
name: gpt4
config:
  id: gpt-4o
  api_key:
    $ref: pragma/secret/openai-key.value
```

---

### Tool Resources

#### `agno/tools/function`
Custom Python function tools.

```yaml
provider: agno
resource: tools/function
name: web-search
config:
  name: search_web
  description: "Search the web for information"
  parameters:
    - name: query
      type: string
      description: "Search query"
      required: true
    - name: max_results
      type: integer
      description: "Maximum results to return"
      default: 10
  code: |
    import requests
    response = requests.get(
        "https://api.search.com/search",
        params={"q": query, "limit": max_results}
    )
    return response.json()
```

**Outputs:**
- `name`: Function name
- `schema`: JSON schema for the tool

---

#### `agno/tools/mcp`
MCP (Model Context Protocol) server integration.

```yaml
provider: agno
resource: tools/mcp
name: github
config:
  command: "npx"
  args: ["-y", "@modelcontextprotocol/server-github"]
  env:
    GITHUB_TOKEN:
      $ref: pragma/secret/github-token.value
```

**Outputs:**
- `name`: Server identifier
- `tools`: List of available tool names
- `ready`: Connection status

---

### Storage Resources

#### `agno/db/postgres`
PostgreSQL storage wrapper for Agno.

```yaml
provider: agno
resource: db/postgres
name: agent-storage
config:
  database:
    $ref: postgresql/database/agents

  # OR direct connection string
  url:
    $ref: pragma/secret/postgres-url.value
```

**Outputs:**
- `url`: Connection string (redacted)
- `ready`: Connection verified

---

#### `agno/db/sqlite`
SQLite storage wrapper for Agno.

```yaml
provider: agno
resource: db/sqlite
name: local-storage
config:
  db_file: "agno.db"
```

---

### Vector Store Resources

#### `agno/vectordb/qdrant`
Qdrant vector store wrapper for Agno.

```yaml
provider: agno
resource: vectordb/qdrant
name: embeddings
config:
  collection:
    $ref: qdrant/collection/agent-embeddings
```

**Outputs:**
- `url`: Qdrant URL
- `collection`: Collection name
- `ready`: Connection verified

---

#### `agno/vectordb/pgvector`
PgVector store wrapper for Agno.

```yaml
provider: agno
resource: vectordb/pgvector
name: embeddings
config:
  database:
    $ref: postgresql/database/vectors
  table_name: "embeddings"
```

---

### Knowledge Resources

#### `agno/knowledge`
Knowledge base with RAG capabilities.

```yaml
provider: agno
resource: knowledge
name: product-docs
config:
  # Vector store (required)
  vector_db:
    $ref: agno/vectordb/qdrant/embeddings

  # Embedder - default, inline, or reference
  embedder:
    $ref: agno/knowledge/embedder/openai/ada
  # OR inline:
  # embedder:
  #   provider: openai
  #   model: text-embedding-3-small
  # OR omit for default

  # Inline sources
  sources:
    - type: website
      url: "https://docs.example.com"
      max_depth: 2
    - type: pdf_url
      url: "https://example.com/manual.pdf"
    - type: text
      content: "Additional context..."

  # File references (uploaded via pragma)
  files:
    - $ref: pragma/file/product-manual
    - $ref: pragma/file/user-guide

  # Chunking configuration
  chunking:
    chunk_size: 1000
    overlap: 200
```

**Outputs:**
- `document_count`: Number of indexed documents
- `chunk_count`: Number of chunks
- `ready`: Indexing complete

---

#### `agno/knowledge/embedder/openai`
OpenAI embedder configuration.

```yaml
provider: agno
resource: knowledge/embedder/openai
name: ada
config:
  model: text-embedding-ada-002
  api_key:
    $ref: pragma/secret/openai-key.value
```

---

### Memory Resources

#### `agno/memory/manager`
Memory manager for customizing how agent memories are created.

```yaml
provider: agno
resource: memory/manager
name: custom
config:
  db:
    $ref: agno/db/postgres/agent-storage

  # Optional: override model for memory operations
  model:
    $ref: agno/models/openai/gpt4

  # Additional instructions for memory creation
  additional_instructions: "Don't store the user's real name"
```

---

### Infrastructure Resources (Separate Providers)

#### `pragma/file`
Managed file storage. *Core platform resource in pragma-os.*

```yaml
provider: pragma
resource: file
name: product-manual
# Content uploaded via CLI: pragma files upload ./manual.pdf --name product-manual
```

**Outputs:**
- `url`: Pragma-hosted URL
- `size`: File size in bytes
- `content_type`: MIME type
- `checksum`: SHA256 hash

---

#### `postgresql/database`
PostgreSQL database. *Separate provider.*

```yaml
provider: postgresql
resource: database
name: agents
config:
  host: "localhost"
  port: 5432
  database: "agents"
  credentials:
    $ref: pragma/secret/postgres-creds.value
```

**Outputs:**
- `host`: Database host
- `port`: Database port
- `database`: Database name
- `url`: Connection string
- `ready`: Connection verified

---

## Complete Example

A full agentic solution with multiple agents, knowledge base, and memory:

```yaml
# === Infrastructure ===

provider: postgresql
resource: database
name: agent-db
config:
  host: "db.example.com"
  port: 5432
  database: "agents"
  credentials:
    $ref: pragma/secret/postgres-creds.value

---
# === Agno Storage ===

provider: agno
resource: db/postgres
name: storage
config:
  database:
    $ref: postgresql/database/agent-db

---
provider: agno
resource: vectordb/qdrant
name: vectors
config:
  collection:
    $ref: qdrant/collection/agent-embeddings

---
# === Model ===

provider: agno
resource: models/anthropic
name: claude
config:
  id: claude-sonnet-4-20250514
  api_key:
    $ref: pragma/secret/anthropic-key.value

---
# === Knowledge ===

provider: agno
resource: knowledge
name: docs
config:
  vector_db:
    $ref: agno/vectordb/qdrant/vectors
  sources:
    - type: website
      url: "https://docs.example.com"
  files:
    - $ref: pragma/file/product-manual

---
# === Tools ===

provider: agno
resource: tools/mcp
name: github
config:
  command: "npx"
  args: ["-y", "@modelcontextprotocol/server-github"]
  env:
    GITHUB_TOKEN:
      $ref: pragma/secret/github-token.value

---
# === Prompts ===

provider: agno
resource: prompt
name: researcher
config:
  instructions:
    - "You are a research specialist"
    - "Always cite your sources"
    - "Be thorough and accurate"

---
provider: agno
resource: prompt
name: writer
config:
  instructions:
    - "You are a technical writer"
    - "Write clear, concise documentation"
    - "Use examples to illustrate concepts"

---
# === Agents ===

provider: agno
resource: agent
name: researcher
config:
  name: "Research Agent"
  role: "Information gathering"
  model:
    $ref: agno/models/anthropic/claude
  prompt:
    $ref: agno/prompt/researcher
  tools:
    - $ref: agno/tools/mcp/github
  knowledge:
    $ref: agno/knowledge/docs
  db:
    $ref: agno/db/postgres/storage
  enable_agentic_memory: true
  cluster:
    $ref: gcp/gke/main

---
provider: agno
resource: agent
name: writer
config:
  name: "Writer Agent"
  role: "Content creation"
  model:
    $ref: agno/models/anthropic/claude
  prompt:
    $ref: agno/prompt/writer
  db:
    $ref: agno/db/postgres/storage
  cluster:
    $ref: gcp/gke/main

---
# === Team ===

provider: agno
resource: team
name: content-team
config:
  name: "Content Creation Team"
  members:
    - $ref: agno/agent/researcher
    - $ref: agno/agent/writer
  mode: coordinate
  instructions:
    - "Researcher gathers information first"
    - "Writer creates content based on research"
  cluster:
    $ref: gcp/gke/main
```

---

## Implementation Order

### Phase 0: Infrastructure (Blocking)
1. **PRA-157** - `postgresql` provider (Urgent)
2. **PRA-158** - `pragma/file` resource (Urgent)

### Phase 1: Design
3. **PRA-149** - Design resource architecture (this document)

### Phase 2: Leaf Resources (Parallelizable)
4. **PRA-150** - `agno/prompt`
5. **PRA-151** → `agno/tools/function`
6. **PRA-152** → `agno/tools/mcp`
7. `agno/models/anthropic`, `agno/models/openai`
8. `agno/db/postgres`, `agno/db/sqlite`
9. `agno/vectordb/qdrant`, `agno/vectordb/pgvector`
10. `agno/knowledge/embedder/openai`

### Phase 3: Composite Resources
11. **PRA-153** → `agno/knowledge`
12. **PRA-154** → `agno/memory/manager`
13. **PRA-155** - `agno/agent` (rebuild)

### Phase 4: Orchestration
14. **PRA-156** - `agno/team`

---

## Open Questions

1. **Additional model providers** - Do we need `agno/models/google`, `agno/models/ollama`, etc.?
2. **Workflow resource** - Should we add `agno/workflow` for deterministic orchestration?
3. **AgentOS features** - Control plane UI, monitoring, etc.?

---

## References

- [Agno Documentation](https://docs.agno.com)
- [Agno GitHub](https://github.com/agno-agi/agno)
- [Pragmatiks SDK](../../../pragma-sdk/)
