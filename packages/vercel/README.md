# Vercel Provider for Pragmatiks

Manage Vercel projects, deployments, and domains through declarative resources using the Vercel REST API.

## Installation

```bash
pragma providers install pragmatiks/vercel
```

## Resources

- **Project** - Create and manage Vercel projects with framework, build settings, and environment variables
- **Deployment** - Trigger and manage deployments for a Vercel project
- **Domain** - Configure custom domains on Vercel projects

## Authentication

This provider uses Vercel API tokens for authentication. Generate a token at https://vercel.com/account/tokens.

The `access_token` of every resource is a secret: it takes a reference to a secret output, such as a `pragma/secret` entry, and a literal token is refused. It is shown as `********` on `describe`, `list`, the web graph and the live event stream, and the provider receives the real token.
