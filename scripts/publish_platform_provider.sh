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
#     (display_name, description, icon_url, tags, package, entrypoint) out
#     of the [tool.pragma] table itself.
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

echo "Reading distribution name from ${PYPROJECT_PATH}..."

if ! DIST_NAME=$(
  PYPROJECT_PATH="${PYPROJECT_PATH}" \
  uv run --isolated python -c '
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
PYPI_RESPONSE="$(mktemp)"
trap 'rm -f "${PYPI_RESPONSE}"' EXIT

WHEEL_FOUND=0
for attempt in $(seq 1 "${PYPI_POLL_ATTEMPTS}"); do
  if curl -sfL -o "${PYPI_RESPONSE}" "${PYPI_JSON_URL}"; then
    HAS_WHEEL=$(
      PYPI_RESPONSE="${PYPI_RESPONSE}" uv run --isolated python -c '
import json
import os
import sys

with open(os.environ["PYPI_RESPONSE"], "rb") as f:
    data = json.load(f)

for entry in data.get("urls") or []:
    if entry.get("packagetype") == "bdist_wheel":
        sys.stdout.write("yes")
        sys.exit(0)

sys.stdout.write("no")
'
    )
    if [ "${HAS_WHEEL}" = "yes" ]; then
      WHEEL_FOUND=1
      break
    fi
  fi
  echo "Attempt ${attempt}: wheel not yet visible on PyPI, waiting ${PYPI_POLL_INTERVAL}s..."
  sleep "${PYPI_POLL_INTERVAL}"
done

if [ "${WHEEL_FOUND}" -ne 1 ]; then
  echo "publish_platform_provider: timed out waiting for ${DIST_NAME} v${VERSION} wheel on PyPI" >&2
  exit 1
fi

if ! WHEEL_META=$(
  PYPI_RESPONSE="${PYPI_RESPONSE}" uv run --isolated python -c '
import json
import os
import sys

with open(os.environ["PYPI_RESPONSE"], "rb") as f:
    data = json.load(f)

for entry in data.get("urls") or []:
    if entry.get("packagetype") != "bdist_wheel":
        continue
    url = entry.get("url")
    sha256 = (entry.get("digests") or {}).get("sha256")
    if not url or not sha256:
        sys.stderr.write("wheel entry is missing url or digests.sha256\n")
        sys.exit(1)
    sys.stdout.write(f"{url}\n{sha256}\n")
    sys.exit(0)

sys.stderr.write("no bdist_wheel entry found in PyPI response\n")
sys.exit(1)
'
); then
  echo "publish_platform_provider: failed to extract wheel metadata from PyPI response" >&2
  exit 1
fi

WHEEL_URL=$(printf '%s\n' "${WHEEL_META}" | sed -n '1p')
WHEEL_SHA256=$(printf '%s\n' "${WHEEL_META}" | sed -n '2p')

if [ -z "${WHEEL_URL}" ] || [ -z "${WHEEL_SHA256}" ]; then
  echo "publish_platform_provider: empty wheel URL or sha256" >&2
  exit 1
fi

echo "Wheel URL: ${WHEEL_URL}"
echo "Wheel sha256: ${WHEEL_SHA256}"

echo "Minting ConsoleMachine M2M token (ttl=${MINT_TTL_SECONDS}s)..."

MINT_STDERR="$(mktemp)"
trap 'rm -f "${PYPI_RESPONSE}" "${MINT_STDERR}"' EXIT

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
  uv run --isolated --with "pragmatiks-cli${PRAGMA_CLI_VERSION}" \
    pragma "${CONTEXT_ARGS[@]}" providers register \
    --wheel-url "${WHEEL_URL}" \
    --sha256 "${WHEEL_SHA256}" \
    --version "${VERSION}" \
    --pyproject "${PYPROJECT_PATH}"

echo "Registered ${DIST_NAME} v${VERSION} with Pragma catalog."
