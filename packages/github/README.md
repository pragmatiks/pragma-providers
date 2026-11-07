# GitHub Provider for Pragmatiks

Manage GitHub repositories, environments, and secrets through declarative resources using the GitHub REST API.

## Installation

```bash
pragma providers install pragmatiks/github
```

## Resources

- **Repository** - Create and manage GitHub repositories with visibility, features, and branch settings
- **Environment** - Configure deployment environments on repositories with protection rules
- **Secret** - Manage repository-level and environment-level secrets with automatic encryption

## Renamed or transferred repositories

A resource is bound to the repository at its declared owner and repository name. When that repository is renamed or transferred, GitHub redirects its old path, and the provider never follows the redirect: it never reads, changes, or deletes the repository at its new location.

- **Create** fails with a reason saying the repository moved, and no repository is created at the old name.
- **Update** fails with the same reason.
- **Delete** succeeds, leaves the moved repository untouched, and logs the old path.

To manage the moved repository, delete the resource and declare a new one at the new owner and name.

## Authentication

This provider uses GitHub Personal Access Tokens (classic or fine-grained) for authentication. Generate a token at https://github.com/settings/tokens.
