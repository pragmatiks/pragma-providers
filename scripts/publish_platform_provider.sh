#!/usr/bin/env bash
#
# Publish a first-party provider version to the Pragmatiks registry and wait until the catalog records
# it as published.
#
# Usage:
#   publish_platform_provider.sh <provider_dir> <version>
#
# Arguments:
#   provider_dir  Provider package directory, such as packages/qdrant. Its pyproject.toml declares
#                 [project].version and exactly one [project.entry-points."pragma.provider"] entry,
#                 whose name is the provider's name in the catalog.
#   version       Version to publish; must equal the pyproject version.
#
# Environment:
#   PRAGMA_PUBLISH_API_KEY  Required. API key of the publishing organization; only the pinned pragma
#                           CLI receives it. The caller's own pragma contexts, stored credentials and
#                           uv tools are neither used nor changed.
#
# Requires: python3 >= 3.11, jq, and uv at the version the workspace pyproject.toml pins.
#
# Example:
#   publish_platform_provider.sh packages/qdrant 7.0.1
#
# Exits 0 without uploading when the catalog already records the version as published. Otherwise
# uploads the wheel and waits for its admission; the upload replaces any record of the version that
# failed or is still being admitted.
#
# Exits 0 once the catalog records the version as published.
#
# Exits non-zero when:
#   - an argument is missing, or PRAGMA_PUBLISH_API_KEY is empty or unset;
#   - the pyproject is missing or malformed, or its version differs from <version>;
#   - the build fails or does not yield exactly one wheel;
#   - the pinned pragma CLI cannot be installed;
#   - the catalog cannot be read to check whether the version is already published;
#   - the admission fails; the catalog's error message is printed;
#   - the upload is refused: for about a minute the catalog reads with no record of it, and the pragma
#     CLI error printed above says why;
#   - 20 minutes after the upload started, the version is still being admitted or the catalog still
#     cannot be read.

set -euo pipefail

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
  echo "publish_platform_provider:" \
    "usage: publish_platform_provider.sh <provider_dir> <version>" >&2
  exit 1
fi

if [ -z "${PRAGMA_PUBLISH_API_KEY:-}" ]; then
  echo "publish_platform_provider: PRAGMA_PUBLISH_API_KEY is empty or unset" >&2
  exit 1
fi

export -n PRAGMA_PUBLISH_API_KEY

PROVIDER_DIRECTORY="$1"
VERSION="$2"

PROVIDER_NAMESPACE="pragmatiks"
PRAGMA_CLI_REQUIREMENT="pragmatiks-cli>=11,<12"
PUBLISH_DEADLINE_SECONDS=1200
POLL_INTERVAL_SECONDS=15
UNRECORDED_READ_LIMIT=5

WORK_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "${WORK_DIRECTORY}"' EXIT
PRAGMA_EXECUTABLE="${WORK_DIRECTORY}/bin/pragma"
CLI_CONFIG_DIRECTORY="${WORK_DIRECTORY}/config"
WHEEL_DIRECTORY="${WORK_DIRECTORY}/wheels"

run_pragma() {
  XDG_CONFIG_HOME="${CLI_CONFIG_DIRECTORY}" PRAGMA_AUTH_TOKEN_DEFAULT="${PRAGMA_PUBLISH_API_KEY}" \
    "${PRAGMA_EXECUTABLE}" --context default "$@"
}

read_catalog_version() {
  local provider_name="$1"
  local version="$2"
  local versions

  if ! versions="$(run_pragma providers versions "${provider_name}" --output json)"; then
    if [[ "${versions}" == *"not found in the store."* ]]; then
      echo "null"
      return 0
    fi

    printf '%s\n' "${versions}" >&2
    return 1
  fi

  jq --arg version "${version}" 'map(select(.version == $version))[0]' <<< "${versions}"
}

PYPROJECT_PATH="${PROVIDER_DIRECTORY}/pyproject.toml"
if [ ! -f "${PYPROJECT_PATH}" ]; then
  echo "publish_platform_provider: pyproject not found at ${PYPROJECT_PATH}" >&2
  exit 1
fi

PYPROJECT_FIELDS="$(python3 - "${PYPROJECT_PATH}" <<'PYTHON'
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    project = tomllib.load(handle).get("project", {})

version = project.get("version")
providers = list(project.get("entry-points", {}).get("pragma.provider", {}))
if not version or len(providers) != 1:
    sys.exit('pyproject needs [project].version and exactly one [project.entry-points."pragma.provider"] entry')

print(version, providers[0])
PYTHON
)"
read -r PYPROJECT_VERSION PROVIDER_SHORT_NAME <<< "${PYPROJECT_FIELDS}"

if [ "${VERSION}" != "${PYPROJECT_VERSION}" ]; then
  echo "publish_platform_provider:" \
    "version '${VERSION}' differs from ${PYPROJECT_PATH}: '${PYPROJECT_VERSION}'" >&2
  exit 1
fi

PROVIDER_NAME="${PROVIDER_NAMESPACE}/${PROVIDER_SHORT_NAME}"
VERSION_LABEL="${PROVIDER_NAME} ${VERSION}"
CHECK_HINT="Check it with: pragma providers versions ${PROVIDER_NAME}"

uv build "${PROVIDER_DIRECTORY}" --wheel --no-sources --out-dir "${WHEEL_DIRECTORY}"

shopt -s nullglob
WHEELS=("${WHEEL_DIRECTORY}"/*.whl)
shopt -u nullglob
if [ "${#WHEELS[@]}" -ne 1 ]; then
  echo "publish_platform_provider: expected exactly one wheel, found ${#WHEELS[@]}" >&2
  exit 1
fi

UV_TOOL_DIR="${WORK_DIRECTORY}/tools" UV_TOOL_BIN_DIR="${WORK_DIRECTORY}/bin" \
  uv tool install --quiet "${PRAGMA_CLI_REQUIREMENT}"

if ! PREVIOUS_CATALOG_VERSION="$(read_catalog_version "${PROVIDER_NAME}" "${VERSION}")"; then
  echo "publish_platform_provider: ${VERSION_LABEL}:" \
    "the catalog could not be read to check whether the version is already published" >&2
  exit 1
fi

PREVIOUS_STATUS="$(jq --raw-output '.status // ""' <<< "${PREVIOUS_CATALOG_VERSION}")"
if [ "${PREVIOUS_STATUS}" = published ]; then
  echo "publish_platform_provider: ${VERSION_LABEL}: already published"
  exit 0
fi

PREVIOUS_UPDATED_AT="$(jq --raw-output '.updated_at // ""' <<< "${PREVIOUS_CATALOG_VERSION}")"
PUBLISH_STARTED_SECONDS="${SECONDS}"

if run_pragma providers publish --wheel "${WHEELS[0]}"; then
  exit 0
fi

UNRECORDED_READS=0
while true; do
  if CATALOG_VERSION="$(read_catalog_version "${PROVIDER_NAME}" "${VERSION}")"; then
    STATUS="$(jq --raw-output --arg previous_updated_at "${PREVIOUS_UPDATED_AT}" '
      if . == null or .updated_at == $previous_updated_at then "unrecorded" else .status end
    ' <<< "${CATALOG_VERSION}")"
  else
    STATUS="unreadable"
  fi

  case "${STATUS}" in
    published)
      echo "publish_platform_provider: ${VERSION_LABEL}: published"
      exit 0
      ;;
    failed)
      ERROR_MESSAGE="$(jq --raw-output '.error_message // ""' <<< "${CATALOG_VERSION}")"
      echo "publish_platform_provider: ${VERSION_LABEL}:" \
        "failed admission: ${ERROR_MESSAGE}" >&2
      exit 1
      ;;
    pending)
      UNRECORDED_READS=0
      echo "publish_platform_provider: ${VERSION_LABEL}: still being admitted"
      ;;
    unrecorded)
      UNRECORDED_READS=$((UNRECORDED_READS + 1))
      echo "publish_platform_provider: ${VERSION_LABEL}:" \
        "no record of this upload (read ${UNRECORDED_READS}/${UNRECORDED_READ_LIMIT})"
      if [ "${UNRECORDED_READS}" -ge "${UNRECORDED_READ_LIMIT}" ]; then
        echo "publish_platform_provider: ${VERSION_LABEL}:" \
          "upload refused, see the pragma CLI error above" >&2
        exit 1
      fi
      ;;
    unreadable)
      echo "publish_platform_provider: ${VERSION_LABEL}: the catalog could not be read"
      ;;
  esac

  if [ $((SECONDS - PUBLISH_STARTED_SECONDS)) -ge "${PUBLISH_DEADLINE_SECONDS}" ]; then
    echo "publish_platform_provider: ${VERSION_LABEL}:" \
      "not published after ${PUBLISH_DEADLINE_SECONDS}s. ${CHECK_HINT}" >&2
    exit 1
  fi

  sleep "${POLL_INTERVAL_SECONDS}"
done
