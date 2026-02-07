## gcp-v0.116.0 (2026-02-07)

### Feat

- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)

## gcp-v0.115.0 (2026-02-07)

### Feat

- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)

## gcp-v0.114.0 (2026-02-07)

### Feat

- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)

## gcp-v0.113.0 (2026-02-07)

### Feat

- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)

## gcp-v0.112.0 (2026-02-07)

### Feat

- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)

## gcp-v0.111.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.110.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.109.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.108.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.107.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.106.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.105.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.104.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.103.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.102.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.101.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.100.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## gcp-v0.99.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.98.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.97.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.96.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.95.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.94.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.93.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.92.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.91.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.90.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.89.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.1 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## gcp-v0.88.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.87.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.86.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.85.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.84.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.83.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.82.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.81.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-02-02)

### Feat

- **kubernetes**: add Deployment resource (#25)
- **agno**: add memory/manager resource and abstract Model interface (#22)
- **agno**: implement knowledge/embedder/openai resource (#21)
- **agno**: add vectordb/qdrant resource for Qdrant vector store (#20)
- **agno**: add tools/mcp resource for MCP server integration
- **agno**: add tools/websearch resource wrapping DuckDuckGoTools (#19)
- **agno**: add prompt resource for reusable instruction templates (#18)
- **agno**: add db/postgres resource for agent storage (#17)
- **gcp**: add cloudsql resource for Cloud SQL instances (#14)
- **agno**: add models/openai resource (#13)
- **agno**: add models/anthropic resource (#12)
- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.80.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.79.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.78.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## gcp-v0.77.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

## gcp-v0.76.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.75.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.74.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.73.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.72.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.71.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.70.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

## gcp-v0.69.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

## gcp-v0.68.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

## gcp-v0.67.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)

## gcp-v0.66.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.65.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.64.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.63.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.62.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.61.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.60.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.59.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.58.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.57.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.56.0 (2026-01-29)

### Feat

- **kubernetes**: add kubernetes provider with lightkube

## gcp-v0.55.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.54.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.53.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.52.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.51.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.50.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.49.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.48.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.47.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.46.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.45.0 (2026-01-29)

### Feat

- **gcp**: add logs() and health() methods to GKE resource

## gcp-v0.44.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.43.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.42.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.41.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.40.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.39.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.38.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.37.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.36.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.35.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.34.0 (2026-01-29)

### Feat

- **gcp**: rename region to location for zonal cluster support

## gcp-v0.33.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.32.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.31.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.30.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.29.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.28.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.27.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.26.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.25.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.24.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.23.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.22.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.21.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.20.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.19.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.18.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.17.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.16.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.15.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.14.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.13.0 (2026-01-29)

### Feat

- **gcp**: add standard cluster support to GKE resource
- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.12.0 (2026-01-28)

### Feat

- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.11.0 (2026-01-25)

### Feat

- **agno**: add agent resource for deploying AI agents to GKE (#9)
- **qdrant**: add database resource that deploys to GKE via Helm (#8)
- **gcp**: add GKE Autopilot cluster resource (#7)
- **docling**: add docling provider for document parsing (#6)
- **qdrant**: add qdrant provider with collection resource (#5)
- **openai**: add embeddings resource for text embedding generation (#4)
- add provider template for pragma providers init (#3)
- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)

### Fix

- **ci**: use GitHub App token for push to bypass branch protection

## gcp-v0.10.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.9.0 (2026-01-18)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.8.0 (2026-01-16)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.7.0 (2026-01-15)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.6.0 (2026-01-15)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.5.0 (2026-01-15)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.4.0 (2026-01-15)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.3.0 (2026-01-14)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.2.1 (2026-01-14)

### Fix

- **deps**: update pragmatiks-sdk to v0.6.0

## gcp-v0.2.0 (2026-01-14)

### Feat

- **gcp**: require credentials from pragma/secret for multi-tenant auth
- add PyPI publishing and rename to pragmatiks-gcp-provider
- **gcp**: add GCP provider with Secret Manager resource

### Fix

- **ci**: per-provider versioning with commitizen and change detection
- **deps**: update pragmatiks-sdk to v0.5.0
- **deps**: update pragmatiks-sdk to v0.4.0
- use default dist/ directory for PyPI publish
- **deps**: update pragmatiks-sdk to v0.3.1
- add pypi environment for trusted publisher
- add module-name for uv_build to find gcp_provider
- **deps**: update pragmatiks-sdk to v0.2.1
- **deps**: update pragmatiks-sdk to v0.1.3
- **ci**: pull before push to avoid race conditions
- format test file
- **deps**: update pragmatiks-sdk to v0.1.2
- **ci**: add ruff to dev dependencies

### Refactor

- **gcp**: use native async client for Secret Manager
