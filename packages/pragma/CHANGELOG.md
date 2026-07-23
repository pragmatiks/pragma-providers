## pragma-v5.1.4 (2026-07-23)

### Fix

- **deps**: update pragmatiks-sdk to v10.0.0 (#102)

## pragma-v5.1.3 (2026-07-22)

### Fix

- **ci**: run pragma publish on shared changes
- **ci**: publish providers through the unified publish route
- **deps**: update pragmatiks-sdk to v9.0.1 (#101)

## pragma-v5.1.2 (2026-07-22)

### Fix

- **ci**: republish providers on automated sdk updates

## pragma-v5.1.1 (2026-07-22)

### Fix

- require pragmatiks-sdk >= 8.0.0 across providers

## pragma-v5.1.0 (2026-05-11)

### Feat

- **publish**: add register_only dispatch mode (#94)

### Fix

- **publish**: push tags reliably so cz finds previous version (#96)

## pragma-v5.0.2 (2026-05-09)

## pragma-v5.0.1 (2026-05-09)

### Feat

- **agno**: declare explicit runtime entrypoint in [tool.pragma] (#86)
- **qdrant**: declare explicit runtime entrypoint in [tool.pragma] (#85)
- **kubernetes**: declare explicit runtime entrypoint in [tool.pragma] (#84)
- **gcp**: declare explicit runtime entrypoint in [tool.pragma] (#83)
- register PyPI wheel via pragma CLI, switch PyPI auth to OIDC (PRA-382) (#82)
- **agno**: add thinking-mode support to models/anthropic (#78)

### Fix

- **publish**: pin --allow-no-commit dispatches to PATCH increment (#92)
- **publish**: hoist provider lookup out of f-string (#91)
- **publish**: post directly to Console wheel-register endpoint (#90)
- **publish**: mint JWT M2M tokens instead of opaque (#88)
- **publish-script**: pin provider wheel via PEP 508 direct reference (#87)
- **ci**: detect new commitizen no-commits output (#77)

## pragma-v1.11.0 (2026-04-25)

### Feat

- **ci**: add allow_no_commit dispatch input for catalog repopulation (PRA-369) (#76)
- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- **ci**: forward provider metadata from pyproject to /console publish (PRA-369) (#75)
- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v5.0.0 (2026-05-08)

### Feat

- **agno**: declare explicit runtime entrypoint in [tool.pragma] (#86)
- **qdrant**: declare explicit runtime entrypoint in [tool.pragma] (#85)
- **kubernetes**: declare explicit runtime entrypoint in [tool.pragma] (#84)
- **gcp**: declare explicit runtime entrypoint in [tool.pragma] (#83)
- register PyPI wheel via pragma CLI, switch PyPI auth to OIDC (PRA-382) (#82)
- **agno**: add thinking-mode support to models/anthropic (#78)

### Fix

- **publish**: hoist provider lookup out of f-string (#91)
- **publish**: post directly to Console wheel-register endpoint (#90)
- **publish**: mint JWT M2M tokens instead of opaque (#88)
- **publish-script**: pin provider wheel via PEP 508 direct reference (#87)
- **ci**: detect new commitizen no-commits output (#77)

## pragma-v1.11.0 (2026-04-25)

### Feat

- **ci**: add allow_no_commit dispatch input for catalog repopulation (PRA-369) (#76)
- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- **ci**: forward provider metadata from pyproject to /console publish (PRA-369) (#75)
- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v4.0.0 (2026-05-08)

### Feat

- **agno**: declare explicit runtime entrypoint in [tool.pragma] (#86)
- **qdrant**: declare explicit runtime entrypoint in [tool.pragma] (#85)
- **kubernetes**: declare explicit runtime entrypoint in [tool.pragma] (#84)
- **gcp**: declare explicit runtime entrypoint in [tool.pragma] (#83)
- register PyPI wheel via pragma CLI, switch PyPI auth to OIDC (PRA-382) (#82)
- **agno**: add thinking-mode support to models/anthropic (#78)

### Fix

- **publish**: post directly to Console wheel-register endpoint (#90)
- **publish**: mint JWT M2M tokens instead of opaque (#88)
- **publish-script**: pin provider wheel via PEP 508 direct reference (#87)
- **ci**: detect new commitizen no-commits output (#77)

## pragma-v1.11.0 (2026-04-25)

### Feat

- **ci**: add allow_no_commit dispatch input for catalog repopulation (PRA-369) (#76)
- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- **ci**: forward provider metadata from pyproject to /console publish (PRA-369) (#75)
- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v3.0.0 (2026-05-08)

### Feat

- **agno**: declare explicit runtime entrypoint in [tool.pragma] (#86)
- **qdrant**: declare explicit runtime entrypoint in [tool.pragma] (#85)
- **kubernetes**: declare explicit runtime entrypoint in [tool.pragma] (#84)
- **gcp**: declare explicit runtime entrypoint in [tool.pragma] (#83)
- register PyPI wheel via pragma CLI, switch PyPI auth to OIDC (PRA-382) (#82)
- **agno**: add thinking-mode support to models/anthropic (#78)

### Fix

- **publish**: mint JWT M2M tokens instead of opaque (#88)
- **publish-script**: pin provider wheel via PEP 508 direct reference (#87)
- **ci**: detect new commitizen no-commits output (#77)

## pragma-v1.11.0 (2026-04-25)

### Feat

- **ci**: add allow_no_commit dispatch input for catalog repopulation (PRA-369) (#76)
- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- **ci**: forward provider metadata from pyproject to /console publish (PRA-369) (#75)
- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v2.0.0 (2026-05-08)

### Feat

- **agno**: declare explicit runtime entrypoint in [tool.pragma] (#86)
- **qdrant**: declare explicit runtime entrypoint in [tool.pragma] (#85)
- **kubernetes**: declare explicit runtime entrypoint in [tool.pragma] (#84)
- **gcp**: declare explicit runtime entrypoint in [tool.pragma] (#83)
- register PyPI wheel via pragma CLI, switch PyPI auth to OIDC (PRA-382) (#82)
- **agno**: add thinking-mode support to models/anthropic (#78)

### Fix

- **publish-script**: pin provider wheel via PEP 508 direct reference (#87)
- **ci**: detect new commitizen no-commits output (#77)

## pragma-v1.15.0 (2026-05-07)

### Feat

- **agno**: add thinking-mode support to models/anthropic (#78)

### Fix

- **ci**: detect new commitizen no-commits output (#77)

## pragma-v1.14.0 (2026-04-25)

### Fix

- **ci**: detect new commitizen no-commits output (#77)

## pragma-v1.13.0 (2026-04-25)

### Fix

- **ci**: detect new commitizen no-commits output (#77)

## pragma-v1.12.0 (2026-04-25)

## pragma-v1.11.1 (2026-04-25)

## pragma-v1.11.0 (2026-04-25)

### Feat

- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- **ci**: mint opaque M2M token (mt_*) instead of JWT-format (PRA-369) (#74)
- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v1.10.0 (2026-04-25)

### Feat

- **ci**: migrate provider publish to ConsoleMachine + /console publish endpoint (PRA-369) (#73)
- rename canonical platform providers from pragmatiks/* to platform/* (PRA-368) (#72)
- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- **agno**: remove workflows placeholder field to satisfy sdk type validation (#71)
- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v1.9.0 (2026-04-21)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v1.8.0 (2026-04-21)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v1.7.0 (2026-04-21)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v1.6.0 (2026-04-21)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v1.5.0 (2026-04-21)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v1.4.0 (2026-04-20)

### Feat

- **agno**: honest readiness for runner + credential validation (#69)

### Fix

- refresh provider lockfiles to resolve cross-provider deps (#70)
- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v1.3.0 (2026-04-20)

### Fix

- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v1.2.0 (2026-04-20)

### Fix

- **pragma**: correct FieldReference syntax in README (#68)

## pragma-v1.1.0 (2026-04-18)

## pragma-v1.0.1 (2026-04-18)

### Fix

- Republish to land an up-to-date provider version in the platform catalog.

## pragma-v1.0.0 (2026-04-18)

### Feat

- Initial public release.
- **secret**: manage secret values exposed as resource outputs
- **config**: define typed configuration blocks consumed by other resources
- **file**: track files uploaded via the Pragmatiks API, exposing URLs and metadata as outputs
