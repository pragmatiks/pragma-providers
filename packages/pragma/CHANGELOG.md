## pragma-v6.2.0 (2026-10-07)

### Feat

- **template**: scaffold and publish providers for wheel admission
- **vercel**: carry the catalog description and keywords in the project metadata
- **supabase**: carry the catalog description and keywords in the project metadata
- **qdrant**: carry the catalog description and keywords in the project metadata
- **pragma**: carry the catalog description and keywords in the project metadata
- **kubernetes**: carry the catalog description and keywords in the project metadata
- **github**: carry the catalog description and keywords in the project metadata
- **gcp**: carry the catalog description and keywords in the project metadata
- **agno**: carry the catalog description and keywords in the project metadata

## pragma-v6.1.1 (2026-10-06)

### Fix

- **deps**: update pragmatiks-sdk to v16.0.0 (#111)

## pragma-v6.1.0 (2026-10-02)

### Feat

- **template**: declare the pragma.provider entry point in scaffolded providers
- **vercel**: declare the pragma.provider entry point
- **supabase**: declare the pragma.provider entry point
- **qdrant**: declare the pragma.provider entry point
- **pragma**: declare the pragma.provider entry point
- **kubernetes**: declare the pragma.provider entry point
- **github**: declare the pragma.provider entry point
- **gcp**: declare the pragma.provider entry point
- **agno**: declare the pragma.provider entry point

### Fix

- **deps**: update pragmatiks-sdk to v15.0.0 (#109)
- **qdrant,agno**: require kubernetes-provider 2.0.1 and gcp-provider 7 for observe-then-act
- **kubernetes**: require pragmatiks-gcp-provider 7 for observe-then-act

## pragma-v6.0.0 (2026-10-01)

### BREAKING CHANGE

- requires pragmatiks-sdk>=14.0.0.
- requires pragmatiks-sdk>=14.0.0.
- requires pragmatiks-sdk>=14.0.0.
- requires pragmatiks-sdk>=14.0.0.
- requires pragmatiks-sdk>=14.0.0.
- requires pragmatiks-sdk>=14.0.0.
- requires pragmatiks-sdk>=14.0.0.
- requires pragmatiks-sdk>=14.0.0.
- providers require pragmatiks-sdk>=14.0.0.

### Feat

- **vercel**: observe-then-act handlers on pragmatiks-sdk 14
- **supabase**: observe-then-act handlers on pragmatiks-sdk 14
- **pragma**: observe-then-act handlers on pragmatiks-sdk 14
- **github**: observe-then-act handlers on pragmatiks-sdk 14
- **agno**: observe-then-act handlers on pragmatiks-sdk 14
- **qdrant**: observe-then-act handlers on pragmatiks-sdk 14
- **kubernetes**: observe-then-act handlers on pragmatiks-sdk 14
- **gcp**: observe-then-act handlers on pragmatiks-sdk 14


- drop provider test suites, pin ruff/ty/uv, template observe-then-act on SDK 14

## pragma-v5.1.8 (2026-10-01)

### Fix

- **deps**: update pragmatiks-sdk to v14.0.0 (#107)

## pragma-v5.1.7 (2026-09-01)

### Fix

- **deps**: update pragmatiks-sdk to v13.0.0 (#105)

## pragma-v5.1.6 (2026-08-28)

### Fix

- **deps**: update pragmatiks-sdk to v12.0.0 (#104)

## pragma-v5.1.5 (2026-07-24)

### Fix

- **deps**: update pragmatiks-sdk to v11.0.0 (#103)

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
