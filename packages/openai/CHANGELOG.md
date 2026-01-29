## openai-v0.8.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## openai-v0.7.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## openai-v0.6.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## openai-v0.5.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## openai-v0.4.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## openai-v0.3.0 (2026-01-29)

### Feat

- **qdrant**: add LoadBalancer exposure and API key authentication (#11)
- **kubernetes**: add kubernetes provider with lightkube
- **gcp**: add logs() and health() methods to GKE resource
- **gcp**: rename region to location for zonal cluster support
- **gcp**: add standard cluster support to GKE resource

## openai-v0.2.0 (2026-01-22)

### Feat

- **openai**: add openai provider with chat_completions resource
- **anthropic**: add anthropic provider with messages resource (#1)
- **gcp**: improve module docstring with resource description
- **gcp**: require credentials from pragma/secret for multi-tenant auth
- add PyPI publishing and rename to pragmatiks-gcp-provider
- **gcp**: add GCP provider with Secret Manager resource

### Fix

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
