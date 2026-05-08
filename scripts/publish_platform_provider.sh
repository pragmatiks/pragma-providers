#!/usr/bin/env bash
#
# Register a platform provider version on api.pragmatiks.io using a wheel
# already uploaded to PyPI.
#
# Assumes the wheel for `<dist-name>==<version>` (where `<dist-name>` is the
# `[project].name` field of the provider's pyproject.toml) has just been
# published to PyPI. The script:
#   - Reads `[project].name` from the provider's pyproject.toml.
#   - Polls https://pypi.org/pypi/<dist-name>/<version>/json until the
#     wheel ("bdist_wheel") entry appears, then captures its `url` and
#     `digests.sha256`.
#   - Mints a fresh short-lived Clerk M2M token (mt_...) from the long-lived
#     Clerk Machine Secret (ak_...), exports it as PRAGMA_AUTH_TOKEN, and
#     invokes `pragma providers register` with the wheel URL, sha256,
#     version, and pyproject path. The CLI reads the rest of the metadata
#     (display_name, description, icon_url, tags, package) out of the
#     [tool.pragma] table itself, then imports the provider package to
#     extract resource schemas — which is why this script installs the
#     provider's just-published wheel into the same `uv run` env via
#     `--with <dist-name>==<version>`. Without that, schema extraction
#     fails with ModuleNotFoundError for the provider's runtime deps
#     (e.g. google-cloud-storage, qdrant_client, agno, ...).
#
# Required env vars:
#   PRAGMA_CONSOLE_MACHINE_SECRET_KEY  Clerk Machine Secret (ak_...) for
#                                      the Pragmatiks Console "ci-release"
#                                      Machine.
#
# Optional env vars:
#   MINT_TTL_SECONDS                   Default: 3600 (max 7200).
#   PRAGMA_CLI_VERSION                 Default: >=3.0.0. Version specifier
#                                      passed to `uv run --with
#                                      pragmatiks-cli<spec>`.
#   PYPI_POLL_ATTEMPTS                 Default: 30. Times to poll PyPI for
#                                      wheel availability.
#   PYPI_POLL_INTERVAL                 Default: 10. Seconds between polls.
#
# To target a non-prod API, configure a CLI context with the desired
# api_url before invoking this script (e.g. `pragma config set-context
# staging --api-url https://api.staging.pragmatiks.io`) and pass
# `--context staging` through PRAGMA_CONTEXT.
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

PRAGMA_CLI_VERSION="${PRAGMA_CLI_VERSION:->=3.0.0}"
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
cleanup() {
  rm -f "${MINT_STDERR}"
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
echo "Minted token (length=${#TOKEN}). Registering ${DIST_NAME} v${VERSION} with Pragma catalog..."

CONTEXT_ARGS=()
if [ -n "${PRAGMA_CONTEXT:-}" ]; then
  CONTEXT_ARGS+=(--context "${PRAGMA_CONTEXT}")
fi

PRAGMA_AUTH_TOKEN="${TOKEN}" \
  uv run --isolated \
    --with "pragmatiks-cli${PRAGMA_CLI_VERSION}" \
    --with "${DIST_NAME}==${VERSION}" \
    pragma "${CONTEXT_ARGS[@]}" providers register \
    --wheel-url "${WHEEL_URL}" \
    --sha256 "${WHEEL_SHA256}" \
    --version "${VERSION}" \
    --pyproject "${PYPROJECT_PATH}"

echo "Registered ${DIST_NAME} v${VERSION} with Pragma catalog."
