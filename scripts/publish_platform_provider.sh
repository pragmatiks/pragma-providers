#!/usr/bin/env bash
#
# Register a platform provider version on the Pragmatiks Console API
# using a wheel already uploaded to PyPI.
#
# Assumes the wheel for `<dist-name>==<version>` (where `<dist-name>` is
# the `[project].name` field of the provider's pyproject.toml) has just
# been published to PyPI. The script:
#   - Reads `[project].name` and `[tool.pragma]` metadata from the
#     provider's pyproject.toml (`provider`, `package`, `display_name`,
#     `description`, optional `icon_url`, `tags`, `entrypoint`).
#   - Polls https://pypi.org/pypi/<dist-name>/<version>/json until the
#     wheel ("bdist_wheel") entry appears, then captures its `url` and
#     `digests.sha256`.
#   - Mints a fresh short-lived Clerk M2M JWT (mt_...) from the
#     long-lived Clerk Machine Secret (ak_...).
#   - POSTs a `WheelProviderVersionCreate` payload to
#     `${PRAGMA_API_URL}/console/provider-versions` with the bearer
#     JWT. The endpoint is gated by the Console release-machine
#     allow-list and accepts only `platform/<name>` registrations.
#
# Resource schemas are intentionally omitted: the wheel-based register
# route accepts a null `schemas` field, and the runtime extracts schemas
# from the installed provider package at deploy time. This is why the
# script does NOT install the provider wheel locally.
#
# Required env vars:
#   PRAGMA_CONSOLE_MACHINE_SECRET_KEY  Clerk Machine Secret (ak_...) for
#                                      the Pragmatiks Console release
#                                      Machine. The Machine ID must be
#                                      in the API's
#                                      CONSOLE_RELEASE_MACHINE_IDS
#                                      allow-list.
#
# Optional env vars:
#   PRAGMA_API_URL                     Default: https://api.pragmatiks.io
#                                      Pragmatiks Console API base URL.
#   MINT_TTL_SECONDS                   Default: 3600 (max 7200).
#   PYPI_POLL_ATTEMPTS                 Default: 30. Times to poll PyPI for
#                                      wheel availability.
#   PYPI_POLL_INTERVAL                 Default: 10. Seconds between polls.
#
# Usage:
#   publish_platform_provider.sh <provider_dir> <version>
#
# Example:
#   publish_platform_provider.sh packages/gcp 0.186.0

set -euo pipefail

PROVIDER_DIR="${1:?provider_dir argument is required (e.g. packages/gcp)}"
VERSION="${2:?version argument is required}"

PYPROJECT_PATH="${PROVIDER_DIR}/pyproject.toml"

PRAGMA_API_URL="${PRAGMA_API_URL:-https://api.pragmatiks.io}"
MINT_TTL_SECONDS="${MINT_TTL_SECONDS:-3600}"
PYPI_POLL_ATTEMPTS="${PYPI_POLL_ATTEMPTS:-30}"
PYPI_POLL_INTERVAL="${PYPI_POLL_INTERVAL:-10}"

if [ -z "${PRAGMA_CONSOLE_MACHINE_SECRET_KEY:-}" ]; then
  echo "publish_platform_provider: PRAGMA_CONSOLE_MACHINE_SECRET_KEY is empty or unset" >&2
  exit 1
fi

if [ ! -f "${PYPROJECT_PATH}" ]; then
  echo "publish_platform_provider: pyproject not found at ${PYPROJECT_PATH}" >&2
  exit 1
fi

MINT_STDERR="$(mktemp)"
REGISTER_STDERR="$(mktemp)"
PAYLOAD_FILE="$(mktemp)"
cleanup() {
  rm -f "${MINT_STDERR}" "${REGISTER_STDERR}" "${PAYLOAD_FILE}"
}
trap cleanup EXIT

echo "Reading distribution name from ${PYPROJECT_PATH}..."

if ! DIST_NAME=$(
  PYPROJECT_PATH="${PYPROJECT_PATH}" \
  uv run --no-project python -c '
import os
import sys
import tomllib

path = os.environ["PYPROJECT_PATH"]
with open(path, "rb") as f:
    data = tomllib.load(f)

name = data.get("project", {}).get("name")
if not name:
    sys.stderr.write(f"{path}: [project].name is missing\n")
    sys.exit(1)

sys.stdout.write(name)
'
); then
  echo "publish_platform_provider: failed to read [project].name from ${PYPROJECT_PATH}" >&2
  exit 1
fi

echo "Resolving wheel URL for ${DIST_NAME} v${VERSION} on PyPI..."

PYPI_JSON_URL="https://pypi.org/pypi/${DIST_NAME}/${VERSION}/json"

# Resolves the wheel URL + sha256 from the PyPI JSON streamed on stdin.
# Exit codes:
#   0  exactly one bdist_wheel entry; emits "<url>\n<sha256>" on stdout.
#   1  zero bdist_wheel entries (wheel not yet propagated — retry).
#   2  malformed JSON or multiple/ambiguous wheels (fatal — do not retry).
read -r -d '' WHEEL_RESOLVER_PY <<'PY' || true
import json
import sys

try:
    data = json.load(sys.stdin)
except json.JSONDecodeError as exc:
    sys.stderr.write(f"invalid PyPI JSON: {exc}\n")
    sys.exit(2)

wheels = [e for e in (data.get("urls") or []) if e.get("packagetype") == "bdist_wheel"]
if not wheels:
    sys.exit(1)
if len(wheels) > 1:
    sys.stderr.write(f"expected exactly one bdist_wheel entry, found {len(wheels)}\n")
    sys.exit(2)

entry = wheels[0]
url = entry.get("url")
sha256 = (entry.get("digests") or {}).get("sha256")
if not url or not sha256:
    sys.stderr.write("wheel entry is missing url or digests.sha256\n")
    sys.exit(2)

sys.stdout.write(f"{url}\n{sha256}\n")
PY

resolve_wheel() {
  curl -sfL "${PYPI_JSON_URL}" \
    | uv run --no-project python -c "${WHEEL_RESOLVER_PY}"
}

WHEEL_META=""
for ((attempt = 1; attempt <= PYPI_POLL_ATTEMPTS; attempt++)); do
  if WHEEL_META=$(resolve_wheel); then
    break
  fi
  rc=$?
  if [ "${rc}" -eq 2 ]; then
    echo "publish_platform_provider: fatal error resolving wheel on PyPI" >&2
    exit 1
  fi
  echo "Attempt ${attempt}: wheel not yet visible on PyPI, waiting ${PYPI_POLL_INTERVAL}s..."
  sleep "${PYPI_POLL_INTERVAL}"
done

if [ -z "${WHEEL_META}" ]; then
  echo "publish_platform_provider: timed out waiting for ${DIST_NAME} v${VERSION} wheel on PyPI" >&2
  exit 1
fi

{ read -r WHEEL_URL; read -r WHEEL_SHA256; } <<<"${WHEEL_META}"

if [ -z "${WHEEL_URL}" ] || [ -z "${WHEEL_SHA256}" ]; then
  echo "publish_platform_provider: empty wheel URL or sha256" >&2
  exit 1
fi

echo "Wheel URL: ${WHEEL_URL}"
echo "Wheel sha256: ${WHEEL_SHA256}"

echo "Building Console registration payload from ${PYPROJECT_PATH}..."

if ! PROVIDER_NAME=$(
  PYPROJECT_PATH="${PYPROJECT_PATH}" \
  VERSION="${VERSION}" \
  WHEEL_URL="${WHEEL_URL}" \
  WHEEL_SHA256="${WHEEL_SHA256}" \
  PAYLOAD_FILE="${PAYLOAD_FILE}" \
  uv run --no-project python -c '
import json
import os
import sys
import tomllib

with open(os.environ["PYPROJECT_PATH"], "rb") as f:
    data = tomllib.load(f)

pragma = data.get("tool", {}).get("pragma")
if not pragma:
    sys.stderr.write("[tool.pragma] table is missing\n")
    sys.exit(1)

required = ["provider", "package", "display_name", "description"]
missing = [k for k in required if not pragma.get(k)]
if missing:
    sys.stderr.write(f"[tool.pragma] missing required keys: {missing}\n")
    sys.exit(1)

metadata = {
    "display_name": pragma["display_name"],
    "description": pragma["description"],
    "tags": pragma.get("tags") or [],
}
icon_url = pragma.get("icon_url")
if icon_url:
    metadata["icon_url"] = icon_url

provider_short = pragma["provider"]
payload = {
    "name": f"platform/{provider_short}",
    "version": os.environ["VERSION"],
    "wheel_url": os.environ["WHEEL_URL"],
    "sha256": os.environ["WHEEL_SHA256"],
    "package_name": pragma["package"],
    "metadata": metadata,
}

entrypoint = pragma.get("entrypoint")
if entrypoint:
    payload["entrypoint"] = list(entrypoint)

with open(os.environ["PAYLOAD_FILE"], "w") as out:
    json.dump(payload, out)

sys.stdout.write(payload["name"])
'
); then
  echo "publish_platform_provider: failed to build registration payload" >&2
  exit 1
fi

echo "Minting ConsoleMachine M2M token (ttl=${MINT_TTL_SECONDS}s)..."

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
echo "Minted token (length=${#TOKEN}). Registering ${PROVIDER_NAME} v${VERSION} at ${PRAGMA_API_URL}/console/provider-versions..."

REGISTER_URL="${PRAGMA_API_URL}/console/provider-versions"
HTTP_CODE=$(
  curl -sS -o "${REGISTER_STDERR}" -w '%{http_code}' \
    -X POST "${REGISTER_URL}" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    --data-binary "@${PAYLOAD_FILE}"
)

case "${HTTP_CODE}" in
  201)
    echo "Registered ${PROVIDER_NAME} v${VERSION} (HTTP ${HTTP_CODE})."
    ;;
  200)
    echo "Already registered: ${PROVIDER_NAME} v${VERSION} (HTTP ${HTTP_CODE})."
    ;;
  *)
    echo "publish_platform_provider: register failed with HTTP ${HTTP_CODE}" >&2
    cat "${REGISTER_STDERR}" >&2
    echo >&2
    exit 1
    ;;
esac
