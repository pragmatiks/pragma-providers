#!/usr/bin/env bash
#
# Publish a platform provider tarball to api.pragmatiks.io via the
# ConsoleMachine-authenticated /console/providers/{name}/publish endpoint.
#
# Mints a fresh short-lived Clerk M2M JWT (mt_...) from the long-lived
# Clerk Machine Secret (ak_...) for each call, then POSTs the tarball as
# multipart/form-data. No long-lived M2M token is kept in CI.
#
# Required env vars:
#   PRAGMA_CONSOLE_MACHINE_SECRET_KEY  Clerk Machine Secret (ak_...) for
#                                      the Pragmatiks Console "ci-release"
#                                      Machine.
#
# Optional env vars:
#   API_BASE_URL                       Default: https://api.pragmatiks.io
#   MINT_TTL_SECONDS                   Default: 3600 (max 7200)
#
# Usage:
#   publish_platform_provider.sh <provider_short_name> <tarball_path> <version>
#
# Example:
#   publish_platform_provider.sh gcp dist/pragmatiks_gcp_provider-0.173.0.tar.gz 0.173.0

set -euo pipefail

PROVIDER_NAME="${1:?provider_short_name argument is required (e.g. gcp)}"
TARBALL_PATH="${2:?tarball_path argument is required}"
VERSION="${3:?version argument is required}"

API_BASE_URL="${API_BASE_URL:-https://api.pragmatiks.io}"
MINT_TTL_SECONDS="${MINT_TTL_SECONDS:-3600}"

if [ -z "${PRAGMA_CONSOLE_MACHINE_SECRET_KEY:-}" ]; then
  echo "publish_platform_provider: PRAGMA_CONSOLE_MACHINE_SECRET_KEY is empty or unset" >&2
  exit 1
fi

if [ ! -f "${TARBALL_PATH}" ]; then
  echo "publish_platform_provider: tarball not found at ${TARBALL_PATH}" >&2
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

from clerk_backend_api import Clerk, TokenFormat

ttl = int(os.environ["MINT_TTL_SECONDS"])
secret = os.environ["CONSOLE_CLERK_MACHINE_SECRET_KEY"]

with Clerk(bearer_auth=secret) as clerk:
    created = clerk.m2m.create_token(
        token_format=TokenFormat.JWT,
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
echo "Minted token (length=${#TOKEN}). Posting ${TARBALL_PATH} to ${API_BASE_URL}..."

RESPONSE_BODY="$(mktemp)"
trap 'rm -f "${MINT_STDERR}" "${RESPONSE_BODY}"' EXIT

HTTP_CODE=$(
  curl -sS \
    -o "${RESPONSE_BODY}" \
    -w '%{http_code}' \
    -X POST "${API_BASE_URL}/console/providers/${PROVIDER_NAME}/publish" \
    -H "Authorization: Bearer ${TOKEN}" \
    -F "version=${VERSION}" \
    -F "code=@${TARBALL_PATH};type=application/gzip"
)

if [ "${HTTP_CODE}" -lt 200 ] || [ "${HTTP_CODE}" -ge 300 ]; then
  echo "publish_platform_provider: API returned HTTP ${HTTP_CODE}" >&2
  cat "${RESPONSE_BODY}" >&2
  echo >&2
  exit 1
fi

echo "Published platform/${PROVIDER_NAME} v${VERSION} (HTTP ${HTTP_CODE})"
cat "${RESPONSE_BODY}"
echo
