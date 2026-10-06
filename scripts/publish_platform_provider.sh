#!/usr/bin/env bash
#
# Publish a first-party provider version to the Pragmatiks catalog.
#
# Runs `pragma providers publish`, which builds the wheel, uploads it to
# `${PRAGMA_API_URL}/providers/publish` and waits until the organization's
# provider host admits the version. The wheel identifies the version: its
# `pragma.provider` entry point names the provider and its metadata carries
# the version. Authentication is a bearer token read from
# PRAGMA_PUBLISH_API_KEY (an org-scoped Clerk API key for the reserved
# `pragmatiks` org).
#
# Preconditions, checked before publishing:
#   - <provider_dir>/pyproject.toml exists.
#   - Its [project].version equals the <version> argument.
#   - It declares exactly one [project.entry-points."pragma.provider"] entry;
#     that entry's name is the provider name in the catalog.
#
# The CLI is pinned to the first major that waits for admission, so the
# script fails when the platform refuses the version.
#
# When the publish fails, the script still succeeds if the catalog already
# lists the version as published, so a rerun of a released version is a
# no-op.
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

project = data.get("project", {})
version = project.get("version")
providers = list(project.get("entry-points", {}).get("pragma.provider", {}))
if not version or len(providers) != 1:
    sys.stderr.write("pyproject needs [project].version and exactly one [project.entry-points.\"pragma.provider\"] entry\n")
    sys.exit(1)

print(f"{version} {providers[0]}")
'
)

if [ "${VERSION}" != "${PYPROJECT_VERSION}" ]; then
  echo "publish_platform_provider: version argument '${VERSION}' does not match ${PYPROJECT_PATH} version '${PYPROJECT_VERSION}'" >&2
  exit 1
fi

echo "Publishing ${PROVIDER_NAMESPACE}/${PROVIDER_SHORT_NAME} v${VERSION} to ${PRAGMA_API_URL}..."

if PRAGMA_AUTH_TOKEN="${PRAGMA_PUBLISH_API_KEY}" PRAGMA_API_URL="${PRAGMA_API_URL}" \
  uvx --from 'pragmatiks-cli>=11' pragma providers publish "${PROVIDER_DIRECTORY}"; then
  exit 0
fi

version_is_published() {
  curl -sf --max-time 30 "${PRAGMA_API_URL}/providers/${PROVIDER_NAMESPACE}/${PROVIDER_SHORT_NAME}/versions" \
    | VERSION="${VERSION}" python3 -c '
import json, os, sys

versions = json.load(sys.stdin)
published = any(
    entry.get("version") == os.environ["VERSION"] and entry.get("status") == "published"
    for entry in versions
)
sys.exit(0 if published else 1)
'
}

if version_is_published; then
  echo "Already published: ${PROVIDER_NAMESPACE}/${PROVIDER_SHORT_NAME} v${VERSION}"
  exit 0
fi

echo "publish_platform_provider: publish failed and ${PROVIDER_NAMESPACE}/${PROVIDER_SHORT_NAME} v${VERSION} is not published in the catalog" >&2
exit 1
