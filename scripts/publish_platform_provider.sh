#!/usr/bin/env bash
#
# Publish a first-party provider version to the Pragmatiks catalog.
#
# Builds the wheel and uploads it via the pragma CLI
# (`pragma providers publish`), which extracts resource schemas in the
# provider's own uv environment and POSTs the wheel to
# `${PRAGMA_API_URL}/providers/publish`. Authentication is a bearer token
# read from PRAGMA_PUBLISH_API_KEY (an org-scoped Clerk API key for the
# reserved `pragmatiks` org).
#
# Required env vars:
#   PRAGMA_PUBLISH_API_KEY  Bearer token for the publishing org.
#
# Optional env vars:
#   PRAGMA_API_URL          Default: https://api.pragmatiks.io
#
# Usage:
#   publish_platform_provider.sh <provider_dir> <version>
#
# Example:
#   publish_platform_provider.sh packages/qdrant 7.0.1

set -euo pipefail

PROVIDER_DIRECTORY="${1:?provider_dir argument is required (e.g. packages/qdrant)}"
VERSION="${2:?version argument is required}"
PRAGMA_API_URL="${PRAGMA_API_URL:-https://api.pragmatiks.io}"
PROVIDER_NAMESPACE="pragmatiks"

if [ -z "${PRAGMA_PUBLISH_API_KEY:-}" ]; then
  echo "publish_platform_provider: PRAGMA_PUBLISH_API_KEY is empty or unset" >&2
  exit 1
fi

PYPROJECT_PATH="${PROVIDER_DIRECTORY}/pyproject.toml"
if [ ! -f "${PYPROJECT_PATH}" ]; then
  echo "publish_platform_provider: pyproject not found at ${PYPROJECT_PATH}" >&2
  exit 1
fi

read -r PYPROJECT_VERSION PROVIDER_SHORT_NAME < <(
  PYPROJECT_PATH="${PYPROJECT_PATH}" python3 -c '
import os, sys, tomllib

with open(os.environ["PYPROJECT_PATH"], "rb") as handle:
    data = tomllib.load(handle)

version = data.get("project", {}).get("version")
provider = data.get("tool", {}).get("pragma", {}).get("provider")
if not version or not provider:
    sys.stderr.write("pyproject missing [project].version or [tool.pragma].provider\n")
    sys.exit(1)

print(f"{version} {provider}")
'
)

if [ "${VERSION}" != "${PYPROJECT_VERSION}" ]; then
  echo "publish_platform_provider: version argument '${VERSION}' does not match ${PYPROJECT_PATH} version '${PYPROJECT_VERSION}'" >&2
  exit 1
fi

echo "Publishing ${PROVIDER_NAMESPACE}/${PROVIDER_SHORT_NAME} v${VERSION} to ${PRAGMA_API_URL}..."

if PRAGMA_AUTH_TOKEN="${PRAGMA_PUBLISH_API_KEY}" PRAGMA_API_URL="${PRAGMA_API_URL}" \
  uvx --from pragmatiks-cli pragma providers publish "${PROVIDER_DIRECTORY}" --version "${VERSION}"; then
  exit 0
fi

VERSIONS_URL="${PRAGMA_API_URL}/providers/${PROVIDER_NAMESPACE}/${PROVIDER_SHORT_NAME}/versions"
if curl -sf --max-time 30 "${VERSIONS_URL}" \
  | VERSION="${VERSION}" python3 -c 'import json, os, sys; sys.exit(0 if any(entry.get("version") == os.environ["VERSION"] for entry in json.load(sys.stdin)) else 1)'; then
  echo "Already published: ${PROVIDER_NAMESPACE}/${PROVIDER_SHORT_NAME} v${VERSION}"
  exit 0
fi

echo "publish_platform_provider: publish failed and ${PROVIDER_NAMESPACE}/${PROVIDER_SHORT_NAME} v${VERSION} is not in the catalog" >&2
exit 1
