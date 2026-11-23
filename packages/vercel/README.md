# Vercel Provider for Pragmatiks

Manage Vercel projects, with the deployments Pragmatiks makes of their Git link, and domains through declarative resources using the Vercel REST API.

## Installation

```bash
pragma providers install pragmatiks/vercel
```

## Resources

- **Project** - Create and manage Vercel projects with framework, build settings, environment variables and a Git repository link, and deploy a Git ref of that repository
- **Domain** - Configure custom domains on Vercel projects

## Deploying a project

A `vercel/project` with a `deployment` block deploys a Git ref of its linked repository and reports that deployment as `deployment_id`, `deployment_url` and `deployment_target`. Without the block, Pragmatiks deploys nothing.

```yaml
config:
  name: web
  git_repository:
    repo:
      provider: github
      resource: repository
      name: web
      field: outputs.full_name
  git_push_deploys: false
  deployment:
    ref: v1.4.0
    target: production
```

- `git_repository.repo` takes a plain `owner/repo` string, or a reference to a `github/repository` resource's `outputs.full_name`, which makes the project wait for the repository.
- `deployment.ref` is a branch, tag or commit SHA; by default, the production branch of the project's Git link. A branch builds its head when the deployment's inputs change; a tag or SHA pins a release.
- Pragmatiks redeploys when the `deployment` block, the environment variables whose targets include the deployment's target, or the build settings change. Rotating a secret environment variable redeploys; changing one the target does not see does not. A reapply with nothing changed deploys nothing.
- Each deployment records a digest of its inputs in its `meta`, which anyone who can read the project's deployments sees: an HMAC-SHA256 keyed by the resource ID over the deployment ref and target, the build settings and the key, type and value of every variable whose targets include the deployment target. The key is not secret, so a low-entropy secret value is guessable by someone who can read deployment `meta`. Values, not Vercel's variable versions, go in because Vercel bumps a variable's `updatedAt` on every upsert, an unchanged one included.
- Deploying needs a GitHub Login Connection on the Vercel account that owns the token, and the Vercel GitHub app must reach the repository; without them, Vercel refuses to link the repository.
- A new project's first deployment is production whatever `target` says: Pragmatiks deploys it and `deployment_target` reports `production`.
- A build that ends `ERROR`, `CANCELED` or `BLOCKED` (refused by a deployment policy or check) fails the project. Resources that depend on it and are not yet READY wait until a reapply builds successfully; the previous good deployment keeps serving.
- A renamed repository keeps deploying only once its link is refreshed on Vercel (reconnect the repository), or the project is recreated.

### Git push deploys

`git_push_deploys` (default `true`) keeps Vercel building every push to the linked repository. Pragmatiks does not report those builds: a push to the production branch replaces a pinned release on Vercel while `deployment_*` still name the pin. A pinned release (`ref` a tag or SHA) therefore sets `git_push_deploys: false`.

`false` writes a project deployment policy that allows every deployment source but Git pushes (Vercel counts the deployments Pragmatiks makes as REST API deployments, so they still build), so it overrides the team's deployment-source rules for this project; `true` returns the project to the team's policy. A new project with `git_push_deploys: false` is created without its repository, given the policy, and only then linked, so no push builds before the policy is in place. Vercel offers deployment policies on Pro and Enterprise plans, and only those with the Owner role can change them (https://vercel.com/docs/deployments/deployment-policy).

Not yet verified against a linked repository: whether Vercel builds a tag or a commit SHA sent as `ref`, which string returns a project to its team's policy, whether a Pragmatiks deployment counts as `rest-api` rather than `git` under the policy, and the project link request.

## Authentication

This provider uses Vercel API tokens for authentication. Generate a token at https://vercel.com/account/tokens.

The `access_token` of every resource is a secret: it takes a reference to a secret output, such as a `pragma/secret` entry, and a literal token is refused. It is shown as `********` on `describe`, `list`, the web graph and the live event stream, and the provider receives the real token.
