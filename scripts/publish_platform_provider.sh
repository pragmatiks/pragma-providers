#!/usr/bin/env bash
#
# Publish a platform provider wheel to api.pragmatiks.io and to the
# Pragmatiks Artifact Registry Python repo via the Pragmatiks CLI.
#
# Mints a fresh short-lived Clerk M2M token (mt_...) from the long-lived
# Clerk Machine Secret (ak_...) and exports it as PRAGMA_AUTH_TOKEN so
# the CLI authenticates as the ConsoleMachine identity. The CLI then:
#   - Builds the wheel from the provider source tree with `uv build`.
#   - Uploads the wheel to GCP Artifact Registry with `uv publish`,
#     using `keyrings.google-artifactregistry-auth` to resolve
#     credentials from Application Default Credentials.
#   - Posts a metadata record to POST /provider-versions on the API.
#
# Required env vars:
#   PRAGMA_CONSOLE_MACHINE_SECRET_KEY  Clerk Machine Secret (ak_...) for
#                                      the Pragmatiks Console "ci-release"
#                                      Machine.
#
# The runner must also have Application Default Credentials available
# for the Artifact Registry upload (Workload Identity Federation in
# CI; `gcloud auth application-default login` locally).
#
# Optional env vars:
#   MINT_TTL_SECONDS                   Default: 3600 (max 7200).
#   PRAGMA_CLI_VERSION                 Default: >=3.0.0. Version
#                                      specifier passed to `uv run
#                                      --with pragmatiks-cli<spec>`.
#
# To target a non-prod API, configure a CLI context with the desired
# api_url before invoking this script (e.g. `pragma config set-context
# staging --api-url https://api.staging.pragmatiks.io`) and pass
# `--context staging` through PRAGMA_CONTEXT.
#
# Usage:
#   publish_platform_provider.sh <provider_dir>
#
# Example:
#   publish_platform_provider.sh packages/gcp

set -euo pipefail

PROVIDER_DIR="${1:?provider_dir argument is required (e.g. packages/gcp)}"

MINT_TTL_SECONDS="${MINT_TTL_SECONDS:-3600}"
PRAGMA_CLI_VERSION="${PRAGMA_CLI_VERSION:->=3.0.0}"

if [ -z "${PRAGMA_CONSOLE_MACHINE_SECRET_KEY:-}" ]; then
  echo "publish_platform_provider: PRAGMA_CONSOLE_MACHINE_SECRET_KEY is empty or unset" >&2
  exit 1
fi

if [ ! -d "${PROVIDER_DIR}" ]; then
  echo "publish_platform_provider: provider directory not found at ${PROVIDER_DIR}" >&2
  exit 1
fi

if [ ! -f "${PROVIDER_DIR}/pyproject.toml" ]; then
  echo "publish_platform_provider: ${PROVIDER_DIR}/pyproject.toml not found" >&2
  exit 1
fi

echo "Minting ConsoleMachine M2M token (ttl=${MINT_TTL_SECONDS}s)..."

MINT_STDERR="$(mktemp)"
trap 'rm -f "${MINT_STDERR}"' EXIT

if ! TOKEN=$(
  CONSOLE_CLERK_MACHINE_SECRET_KEY="${PRAGMA_CONSOLE_MACHINE_SECRET_KEY}" \
  MINT_TTL_SECONDS="${MINT_TTL_SECONDS}" \
  uv run --isolated --with 'clerk-backend-api>=5.0.2' \
    python -c '
import os
import sys

from clerk_backend_api import Clerk

ttl = int(os.environ["MINT_TTL_SECONDS"])
secret = os.environ["CONSOLE_CLERK_MACHINE_SECRET_KEY"]

with Clerk(bearer_auth=secret) as clerk:
    created = clerk.m2m.create_token(
        seconds_until_expiration=float(ttl),
        claims=None,
    )

token = getattr(created, "token", None)
if not token:
    sys.stderr.write(f"mint returned no token: {created!r}\n")
    sys.exit(1)

sys.stdout.write(token)
' 2> "${MINT_STDERR}"
); then
  echo "publish_platform_provider: failed to mint ConsoleMachine token" >&2
  cat "${MINT_STDERR}" >&2
  exit 1
fi

if [ -z "${TOKEN}" ]; then
  echo "publish_platform_provider: mint returned an empty token" >&2
  cat "${MINT_STDERR}" >&2
  exit 1
fi

echo "::add-mask::${TOKEN}"
echo "Minted token (length=${#TOKEN}). Publishing ${PROVIDER_DIR} via pragma CLI..."

# `uv run --with` builds an ephemeral env that includes:
#   - pragmatiks-cli (the publishing CLI itself).
#   - keyrings.google-artifactregistry-auth (so `uv publish` inside the
#     CLI can resolve GAR credentials from Application Default
#     Credentials).
# The CLI inherits PRAGMA_AUTH_TOKEN from the parent shell.
PRAGMA_AUTH_TOKEN="${TOKEN}" \
uv run --isolated \
  --with "pragmatiks-cli${PRAGMA_CLI_VERSION}" \
  --with "keyrings.google-artifactregistry-auth" \
  pragma providers publish --directory "${PROVIDER_DIR}"
