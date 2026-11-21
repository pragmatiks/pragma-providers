# Supabase Provider for Pragmatiks

Manage Supabase projects and configuration through declarative resources using the Supabase Management API.

## Installation

```bash
pragma providers install pragmatiks/supabase
```

## Resources

- **Project** - Create and manage Supabase projects, including their authentication settings (providers, signups, JWT expiry)

## Authentication

This provider uses Supabase Personal Access Tokens (PAT) for the Management API. Generate a token at https://supabase.com/dashboard/account/tokens.

The project's `access_token` and `database_password` are secrets: each takes a reference to a secret output, such as a `pragma/secret` entry, and a literal value is refused. Both are shown as `********` on `describe`, `list`, the web graph and the live event stream, and the provider receives the real values.
