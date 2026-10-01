# GitHub Provider for Pragmatiks

Manage GitHub repositories, environments, and secrets through declarative resources using the GitHub REST API.

## Resources

- **Repository** - Create and manage GitHub repositories with visibility, features, and branch settings
- **Environment** - Configure deployment environments on repositories with protection rules
- **Secret** - Manage repository-level and environment-level secrets with automatic encryption

## Renamed or transferred repositories

A resource is bound to the repository at its declared owner and repository name. When that repository is renamed or transferred, the provider treats it as absent there: it never reads, changes, or deletes the repository at its new location. Deleting the resource succeeds without touching the moved repository, and a create or update fails with a message asking you to set the owner and repository name to the new location.

## Authentication

This provider uses GitHub Personal Access Tokens (classic or fine-grained) for authentication. Generate a token at https://github.com/settings/tokens.
