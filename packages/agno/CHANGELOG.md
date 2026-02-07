## agno-v0.64.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## agno-v0.55.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.44.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.63.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## agno-v0.55.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.44.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.33.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.62.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## agno-v0.55.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.44.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.33.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.61.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## agno-v0.55.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.44.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.33.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.60.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## agno-v0.59.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## agno-v0.58.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## agno-v0.57.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## agno-v0.56.0 (2026-02-07)

### Feat

- **kubernetes**: add startup probe support and authorized_user credentials (#28)

## agno-v0.55.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.54.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.44.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.53.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.44.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.33.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.52.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.44.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.33.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.51.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.44.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.33.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.50.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.49.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.48.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.47.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.46.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.45.0 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.44.1 (2026-02-05)

### Fix

- **agno**: runtime dependencies and import fix (#27)

## agno-v0.44.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.43.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.33.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.42.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.33.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.41.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.33.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.40.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.39.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.38.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.37.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.36.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.35.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.34.0 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.33.1 (2026-02-02)

### Fix

- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs

## agno-v0.33.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.32.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.31.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.30.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.29.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.28.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.27.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.26.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.25.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.24.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.23.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.22.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.21.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.20.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.19.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.18.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.17.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.16.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.15.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.14.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.13.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-02-02)

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
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

### Fix

- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)

### Refactor

- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py

## agno-v0.12.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.11.0 (2026-02-02)

### Feat

- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **ci**: use PyPI API for availability check instead of pip index

## agno-v0.10.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## agno-v0.9.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## agno-v0.8.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## agno-v0.7.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## agno-v0.6.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## agno-v0.5.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## agno-v0.4.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## agno-v0.3.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## agno-v0.2.0 (2026-01-25)

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
- **gcp**: improve module docstring with resource description
- **gcp**: require credentials from pragma/secret for multi-tenant auth
- add PyPI publishing and rename to pragmatiks-gcp-provider
- **gcp**: add GCP provider with Secret Manager resource

### Fix

- **ci**: use GitHub App token for push to bypass branch protection
- **deps**: update pragmatiks-sdk to v0.6.0
- **ci**: correct dist paths for workspace builds
- **ci**: collect built packages to central dist directory
- **gcp**: update README with actual Secret resource documentation
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
