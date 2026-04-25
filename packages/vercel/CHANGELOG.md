## vercel-v0.33.0 (2026-04-25)

### Feat

- **ci**: add allow_no_commit dispatch input for catalog repopulation (PRA-369) (#76)
- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **ci**: detect new commitizen no-commits output (#77)
- **ci**: forward provider metadata from pyproject to /console publish (PRA-369) (#75)
- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.32.0 (2026-04-25)

### Feat

- **ci**: add allow_no_commit dispatch input for catalog repopulation (PRA-369) (#76)
- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **ci**: forward provider metadata from pyproject to /console publish (PRA-369) (#75)
- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.31.0 (2026-04-25)

### Feat

- **ci**: add allow_no_commit dispatch input for catalog repopulation (PRA-369) (#76)
- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **ci**: forward provider metadata from pyproject to /console publish (PRA-369) (#75)
- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.30.0 (2026-04-25)

### Feat

- **ci**: add allow_no_commit dispatch input for catalog repopulation (PRA-369) (#76)
- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **ci**: forward provider metadata from pyproject to /console publish (PRA-369) (#75)
- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.29.0 (2026-04-25)

### Feat

- **ci**: add allow_no_commit dispatch input for catalog repopulation (PRA-369) (#76)
- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **ci**: forward provider metadata from pyproject to /console publish (PRA-369) (#75)
- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.28.0 (2026-04-25)

### Feat

- **ci**: add allow_no_commit dispatch input for catalog repopulation (PRA-369) (#76)
- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **ci**: forward provider metadata from pyproject to /console publish (PRA-369) (#75)
- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.27.0 (2026-04-25)

### Feat

- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **ci**: forward provider metadata from pyproject to /console publish (PRA-369) (#75)
- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.26.0 (2026-04-25)

### Feat

- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.25.0 (2026-04-25)

### Feat

- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.24.0 (2026-04-25)

### Feat

- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.23.0 (2026-04-23)

### Feat

- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.22.0 (2026-04-21)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.21.0 (2026-04-21)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.20.0 (2026-04-21)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.19.0 (2026-04-21)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.18.0 (2026-04-21)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.17.0 (2026-04-20)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)
- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.16.0 (2026-04-20)

### Feat

- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

### Fix

- **pragma**: correct FieldReference syntax in README (#68)

## vercel-v0.15.0 (2026-04-18)

### Feat

- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

## vercel-v0.14.0 (2026-04-18)

### Feat

- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

## vercel-v0.13.0 (2026-04-18)

### Feat

- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.12.0 (2026-04-18)

### Feat

- **pragma**: publish pragma provider via standard pipeline (#66)
- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

## vercel-v0.11.0 (2026-04-16)

### Feat

- **agno**: multi-entity AgentOS support on runner (#65)
- **kubernetes**: config resource with multi-cluster auth modes (#64)

## vercel-v0.10.0 (2026-04-16)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.9.0 (2026-04-13)

### Feat

- **agno**: add output_schema for structured agent responses
- **agno**: add HITL fields to ToolsMCP and approvals_table to DbPostgres
- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- **agno**: mark api_key fields as SensitiveField for API response redaction (#61)
- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.8.0 (2026-04-07)

### Feat

- **ci**: skip builds for unchanged providers
- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.7.0 (2026-04-04)

### Feat

- accept workflow_dispatch trigger for CLI cascade
- add icon_url to all provider pyproject.toml

### Fix

- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.6.0 (2026-04-03)

### Feat

- add icon_url to all provider pyproject.toml

### Fix

- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.5.0 (2026-04-03)

### Fix

- remove redundant PRAGMA_API_URL env var from publish workflow
- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.4.0 (2026-04-03)

### Fix

- use PRAGMA_AUTH_TOKEN secret instead of CLERK_SECRET_KEY for store publish
- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.3.0 (2026-04-03)

### Fix

- set PRAGMA_API_URL in publish workflow for pragma store
- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.2.1 (2026-04-03)

### Fix

- install pragma CLI and add git push retry to publish workflow
- restructure publish workflow as per-provider DAG with API token auth

## vercel-v0.2.0 (2026-04-03)

### Feat

- add GitHub provider with Repository, Environment, and Secret resources (#59)
- add explicit upgrade/downgrade methods to all resources (PRA-226) (#41)
- migrate provider configs to Field[T] and ImmutableField[T] (PRA-225) (#36)
- add pragma store publish to CI/CD pipeline (#35)
- add kubernetes/namespace resource type (PRA-177) (#34)
- add store metadata and seed script (PRA-211) (#33)
- add store metadata fields to provider copier template (PRA-193) (#32)
- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)
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

- add supabase, vercel, github to publish and CI workflows
- **ci**: update per-provider lockfiles in update-sdk workflow
- **ci**: pull --rebase before push and fix update-sdk auto-merge
- use PyPI JSON API for version availability polling (#46)
- add PyPI availability polling to update-sdk workflow (#45)
- **gcp**: replace fake $ref syntax with actual reference format in docstrings (#42)
- remove editable SDK source overrides from provider pyproject.toml files (#38)
- **gcp**: handle CloudSQL 400 error when deleting non-existent user
- **agno**: remove wait_ready calls from runner resource application
- **gcp**: handle HTTP 400 for already-existing CloudSQL databases
- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index
- **ci**: output builds to workspace dist directory
- **ci**: use env vars instead of dynamic expressions in publish workflow
- **gcp**: add defaults for optional outputs to ensure serialization
- **gcp**: use Dependency.resolve() for instance access (#16)
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

- remove provider identity from Python classes (PRA-269)
- **agno**: DRY refactor with base classes and spec pattern (#23)
- **agno**: use pytest-mock MockType instead of Any for mock typing
- **agno**: move mock_mcp_tools fixture to conftest.py
- **gcp**: use native async client for Secret Manager
