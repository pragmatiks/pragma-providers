# Changelog

All notable changes to this project will be documented in this file.

## 0.1.0 (2026-01-24)

### Features

- Initial release
- Collection resource with create, update, delete lifecycle
- Support for Qdrant Cloud and local instances

## qdrant-v0.33.1 (2026-03-25)

### Fix

- **ci**: update per-provider lockfiles in update-sdk workflow

## qdrant-v0.33.0 (2026-03-25)

### Feat

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

### Fix

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

### Refactor

- remove provider identity from Python classes (PRA-269)

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.32.0 (2026-03-25)

### Feat

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

### Fix

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

### Refactor

- remove provider identity from Python classes (PRA-269)

## qdrant-v0.31.0 (2026-03-19)

### Feat

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

### Fix

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

## qdrant-v0.30.0 (2026-03-19)

### Feat

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

### Fix

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

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.29.0 (2026-03-17)

### Feat

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

### Fix

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

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.28.0 (2026-03-16)

### Feat

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

### Fix

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

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.27.0 (2026-03-16)

### Feat

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

### Fix

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

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.26.0 (2026-03-07)

### Feat

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

### Fix

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

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.25.0 (2026-03-06)

### Feat

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

### Fix

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

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.24.0 (2026-03-04)

### Feat

- migrate provider configs to Field[T] and ImmutableField[T] (PRA-225) (#36)
- add pragma store publish to CI/CD pipeline (#35)
- add kubernetes/namespace resource type (PRA-177) (#34)
- add store metadata and seed script (PRA-211) (#33)
- add store metadata fields to provider copier template (PRA-193) (#32)
- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

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

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.23.0 (2026-03-04)

### Feat

- migrate provider configs to Field[T] and ImmutableField[T] (PRA-225) (#36)
- add pragma store publish to CI/CD pipeline (#35)
- add kubernetes/namespace resource type (PRA-177) (#34)
- add store metadata and seed script (PRA-211) (#33)
- add store metadata fields to provider copier template (PRA-193) (#32)
- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

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

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.22.0 (2026-03-04)

### Feat

- migrate provider configs to Field[T] and ImmutableField[T] (PRA-225) (#36)
- add pragma store publish to CI/CD pipeline (#35)
- add kubernetes/namespace resource type (PRA-177) (#34)
- add store metadata and seed script (PRA-211) (#33)
- add store metadata fields to provider copier template (PRA-193) (#32)
- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

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

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.21.0 (2026-03-04)

### Feat

- migrate provider configs to Field[T] and ImmutableField[T] (PRA-225) (#36)
- add pragma store publish to CI/CD pipeline (#35)
- add kubernetes/namespace resource type (PRA-177) (#34)
- add store metadata and seed script (PRA-211) (#33)
- add store metadata fields to provider copier template (PRA-193) (#32)
- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **gcp**: handle CloudSQL 400 error when deleting non-existent user
- **agno**: remove wait_ready calls from runner resource application
- **gcp**: handle HTTP 400 for already-existing CloudSQL databases
- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.20.0 (2026-03-01)

### Feat

- migrate provider configs to Field[T] and ImmutableField[T] (PRA-225) (#36)
- add pragma store publish to CI/CD pipeline (#35)
- add kubernetes/namespace resource type (PRA-177) (#34)
- add store metadata and seed script (PRA-211) (#33)
- add store metadata fields to provider copier template (PRA-193) (#32)
- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **gcp**: handle CloudSQL 400 error when deleting non-existent user
- **agno**: remove wait_ready calls from runner resource application
- **gcp**: handle HTTP 400 for already-existing CloudSQL databases
- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## qdrant-v0.19.0 (2026-02-27)

### Feat

- add pragma store publish to CI/CD pipeline (#35)
- add kubernetes/namespace resource type (PRA-177) (#34)
- add store metadata and seed script (PRA-211) (#33)
- add store metadata fields to provider copier template (PRA-193) (#32)
- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **gcp**: handle CloudSQL 400 error when deleting non-existent user
- **agno**: remove wait_ready calls from runner resource application
- **gcp**: handle HTTP 400 for already-existing CloudSQL databases
- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## qdrant-v0.18.0 (2026-02-25)

### Feat

- add kubernetes/namespace resource type (PRA-177) (#34)
- add store metadata and seed script (PRA-211) (#33)
- add store metadata fields to provider copier template (PRA-193) (#32)
- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **gcp**: handle CloudSQL 400 error when deleting non-existent user
- **agno**: remove wait_ready calls from runner resource application
- **gcp**: handle HTTP 400 for already-existing CloudSQL databases
- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## qdrant-v0.17.0 (2026-02-25)

### Feat

- add store metadata and seed script (PRA-211) (#33)
- add store metadata fields to provider copier template (PRA-193) (#32)
- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **gcp**: handle CloudSQL 400 error when deleting non-existent user
- **agno**: remove wait_ready calls from runner resource application
- **gcp**: handle HTTP 400 for already-existing CloudSQL databases
- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## qdrant-v0.16.0 (2026-02-24)

### Feat

- add store metadata fields to provider copier template (PRA-193) (#32)
- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **gcp**: handle CloudSQL 400 error when deleting non-existent user
- **agno**: remove wait_ready calls from runner resource application
- **gcp**: handle HTTP 400 for already-existing CloudSQL databases
- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## qdrant-v0.15.0 (2026-02-23)

### Feat

- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **gcp**: handle CloudSQL 400 error when deleting non-existent user
- **agno**: remove wait_ready calls from runner resource application
- **gcp**: handle HTTP 400 for already-existing CloudSQL databases
- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## qdrant-v0.14.0 (2026-02-09)

### Feat

- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **gcp**: handle CloudSQL 400 error when deleting non-existent user
- **agno**: remove wait_ready calls from runner resource application
- **gcp**: handle HTTP 400 for already-existing CloudSQL databases
- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## qdrant-v0.13.0 (2026-02-09)

### Feat

- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: remove wait_ready calls from runner resource application
- **gcp**: handle HTTP 400 for already-existing CloudSQL databases
- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## qdrant-v0.10.0 (2026-02-02)

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

## qdrant-v0.12.0 (2026-02-09)

### Feat

- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **gcp**: handle HTTP 400 for already-existing CloudSQL databases
- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## qdrant-v0.11.0 (2026-02-07)

### Feat

- **agno**: add knowledge and content resource support (#30)
- **agno**: add team resource, runner auth, model discriminators, and memory config (#29)
- **kubernetes**: add startup probe support and authorized_user credentials (#28)
- **agno**: rebuild agent, add team and deployment resources (#24)

### Fix

- **agno**: drop --frozen from Dockerfile uv sync (incompatible with --no-sources)
- **ci**: prevent infinite publish loop on bump commits
- **agno**: runtime dependencies and import fix (#27)
- **agno**: rewrite agent tests for current AgentConfig implementation
- **gcp**: convert db_port to int in CloudSQL database outputs
- **ci**: use PyPI API for availability check instead of pip index

## qdrant-v0.10.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## qdrant-v0.9.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## qdrant-v0.8.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## qdrant-v0.7.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## qdrant-v0.6.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## qdrant-v0.5.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## qdrant-v0.4.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## qdrant-v0.3.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## qdrant-v0.2.0 (2026-01-25)

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
