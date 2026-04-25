#!/usr/bin/env bash
#
# Publish a platform provider tarball to api.pragmatiks.io via the
# ConsoleMachine-authenticated /console/providers/{name}/publish endpoint.
#
# Mints a fresh short-lived Clerk M2M token (mt_...) from the long-lived
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
#   publish_platform_provider.sh <provider_short_name> <tarball_path> <version> <pyproject_path>
#
# The pyproject_path points at the provider's pyproject.toml. Provider
# metadata (display_name, description, icon_url, tags) is read from the
# [tool.pragma] table and forwarded to the API as form fields.
#
# Example:
#   publish_platform_provider.sh \
#     gcp \
#     dist/pragmatiks_gcp_provider-0.173.0.tar.gz \
#     0.173.0 \
#     packages/gcp/pyproject.toml

set -euo pipefail

PROVIDER_NAME="${1:?provider_short_name argument is required (e.g. gcp)}"
TARBALL_PATH="${2:?tarball_path argument is required}"
VERSION="${3:?version argument is required}"
PYPROJECT_PATH="${4:?pyproject_path argument is required (e.g. packages/gcp/pyproject.toml)}"

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

if [ ! -f "${PYPROJECT_PATH}" ]; then
  echo "publish_platform_provider: pyproject not found at ${PYPROJECT_PATH}" >&2
  exit 1
fi

echo "Reading provider metadata from ${PYPROJECT_PATH}..."

if ! METADATA=$(
  PYPROJECT_PATH="${PYPROJECT_PATH}" \
  uv run --isolated python -c '
import json
import os
import sys
import tomllib

path = os.environ["PYPROJECT_PATH"]
with open(path, "rb") as f:
    data = tomllib.load(f)

pragma = data.get("tool", {}).get("pragma", {})

display_name = pragma.get("display_name")
description = pragma.get("description")
if not display_name or not description:
    sys.stderr.write(
        f"{path}: [tool.pragma] must define both display_name and description\n"
    )
    sys.exit(1)

icon_url = pragma.get("icon_url") or ""
tags = pragma.get("tags") or []
if not isinstance(tags, list):
    sys.stderr.write(f"{path}: [tool.pragma].tags must be an array\n")
    sys.exit(1)
tags_json = json.dumps(tags) if tags else ""

# Emit shell-evaluable lines: KEY=<base64-encoded value>. Base64 keeps
# arbitrary characters (quotes, newlines, shell metachars) safe across
# the bash boundary.
import base64

def emit(key: str, value: str) -> None:
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    sys.stdout.write(f"{key}={encoded}\n")

emit("DISPLAY_NAME", display_name)
emit("DESCRIPTION", description)
emit("ICON_URL", icon_url)
emit("TAGS_JSON", tags_json)
'
); then
  echo "publish_platform_provider: failed to read metadata from ${PYPROJECT_PATH}" >&2
  exit 1
fi

decode_meta() {
  local key="$1"
  local line
  line=$(printf '%s\n' "${METADATA}" | grep "^${key}=" || true)
  if [ -z "${line}" ]; then
    echo ""
    return
  fi
  printf '%s' "${line#${key}=}" | base64 --decode
}

DISPLAY_NAME=$(decode_meta DISPLAY_NAME)
DESCRIPTION=$(decode_meta DESCRIPTION)
ICON_URL=$(decode_meta ICON_URL)
TAGS_JSON=$(decode_meta TAGS_JSON)

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
echo "Minted token (length=${#TOKEN}). Posting ${TARBALL_PATH} to ${API_BASE_URL}..."

RESPONSE_BODY="$(mktemp)"
trap 'rm -f "${MINT_STDERR}" "${RESPONSE_BODY}"' EXIT

CURL_FORM_ARGS=(
  -F "version=${VERSION}"
  -F "display_name=${DISPLAY_NAME}"
  -F "description=${DESCRIPTION}"
)

# Optional fields are omitted entirely when empty so the API form parser
# treats them as absent rather than empty strings.
if [ -n "${ICON_URL}" ]; then
  CURL_FORM_ARGS+=(-F "icon_url=${ICON_URL}")
fi
if [ -n "${TAGS_JSON}" ]; then
  CURL_FORM_ARGS+=(-F "tags=${TAGS_JSON}")
fi

CURL_FORM_ARGS+=(-F "code=@${TARBALL_PATH};type=application/gzip")

HTTP_CODE=$(
  curl -sS \
    -o "${RESPONSE_BODY}" \
    -w '%{http_code}' \
    -X POST "${API_BASE_URL}/console/providers/${PROVIDER_NAME}/publish" \
    -H "Authorization: Bearer ${TOKEN}" \
    "${CURL_FORM_ARGS[@]}"
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
