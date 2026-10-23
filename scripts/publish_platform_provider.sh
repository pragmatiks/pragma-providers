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
# Requires: python3 >= 3.11, jq, uv at the version the workspace pyproject.toml pins, and a Python
# >= 3.13 that uv can find or download for the pinned pragma CLI (pragmatiks-cli 13.1 or a later 13.x
# on pragmatiks-sdk 18.x).
#
# Example:
#   publish_platform_provider.sh packages/qdrant 7.0.1
#
# Exits 0 without building or uploading when the catalog already records the version as published.
# Otherwise builds the wheel and publishes it with pragma providers publish. A wheel other than the one
# the catalog records for the version replaces a record that is still being published or failed, and
# is refused while the record is promoting (pragma exit 1). The same wheel, which a rerun on the same
# commit builds, is watched while the record is still being published, and starts a new attempt when
# the record failed. pragma providers publish runs with --output json: its stderr flows to the log, and
# once its watch ends its stdout carries one document, {"operation_id": <this publish's operation>,
# "version": <the catalog record as last read>}, on exits 0, 1 and 3; earlier exits leave it empty. A
# record whose operation_id differs from the document's operation_id was replaced by another publish of
# this version. The pragma providers publish exit code decides what happens next:
#   - 0: the version is published; exits 0.
#   - 3: the wheel was accepted but the CLI stopped watching before the version settled, or could not
#     read it; polls the catalog, or exits 1 when the document has no operation_id.
#   - anything else: exits 1 at once without reading the catalog; the pragma CLI error, a replaced
#     publish included, is printed above.
#
# Polling ends when the catalog records the version as published (exit 0), failed (exit 1, with the
# catalog's error message), missing (exit 1), replaced by another publish of this version (exit 1), or
# 20 minutes after the publish started have passed (exit 1). Any other status (staged, queued, admitting
# or promoting) and a catalog that cannot be read are polled again until the deadline.
#
# Also exits 1 when an argument is missing, PRAGMA_PUBLISH_API_KEY is empty or unset, the pyproject is
# missing or malformed or its version differs from <version>, the catalog cannot be read to check
# whether the version is already published, or the build does not yield exactly one wheel. A failing
# uv tool install (the pinned pragma CLI cannot be installed) or uv build exits with uv's own non-zero
# exit code instead of 1; uv's error is printed above.
#
# Catalog reads go through read_catalog_version <provider_name> <version>, which maps the exit code of
# pragma providers versions --output json: 0 prints the version's catalog record as JSON, or null when
# the provider has no such version; 4 (the provider is not found or not visible) prints null; anything
# else, including 1 for a status the pragma CLI does not recognize and 2, returns 1 after printing the
# pragma CLI error, and the catalog counts as unreadable. run_pragma runs the pinned pragma CLI in an
# isolated context.

set -euo pipefail

PROVIDER_NAMESPACE="pragmatiks"
PRAGMA_CLI_REQUIREMENT="pragmatiks-cli>=13.1,<14"
PRAGMA_SDK_REQUIREMENT="pragmatiks-sdk>=18,<19"
PUBLISH_DEADLINE_SECONDS=1200
POLL_INTERVAL_SECONDS=15

FAILURE_EXIT_CODE=1
ADMISSION_UNCONFIRMED_EXIT_CODE=3
NOT_FOUND_EXIT_CODE=4

PUBLISHED_STATUS="published"
FAILED_STATUS="failed"
MISSING_RECORD_STATUS="missing"
REPLACED_RECORD_STATUS="replaced"
UNREADABLE_CATALOG_STATUS="unreadable"

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
  echo "publish_platform_provider:" \
    "usage: publish_platform_provider.sh <provider_dir> <version>" >&2
  exit "${FAILURE_EXIT_CODE}"
fi

if [ -z "${PRAGMA_PUBLISH_API_KEY:-}" ]; then
  echo "publish_platform_provider: PRAGMA_PUBLISH_API_KEY is empty or unset" >&2
  exit "${FAILURE_EXIT_CODE}"
fi

export -n PRAGMA_PUBLISH_API_KEY

PROVIDER_DIRECTORY="$1"
VERSION="$2"

WORK_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "${WORK_DIRECTORY}"' EXIT
PRAGMA_EXECUTABLE="${WORK_DIRECTORY}/bin/pragma"
CLI_CONFIG_DIRECTORY="${WORK_DIRECTORY}/config"
WHEEL_DIRECTORY="${WORK_DIRECTORY}/wheels"
VERSIONS_STDERR_PATH="${WORK_DIRECTORY}/versions.stderr"

run_pragma() {
  XDG_CONFIG_HOME="${CLI_CONFIG_DIRECTORY}" PRAGMA_AUTH_TOKEN_DEFAULT="${PRAGMA_PUBLISH_API_KEY}" \
    "${PRAGMA_EXECUTABLE}" --context default "$@"
}

read_catalog_version() {
  local provider_name="$1"
  local version="$2"
  local versions
  local versions_exit_code=0

  versions="$(run_pragma providers versions "${provider_name}" --output json \
    2> "${VERSIONS_STDERR_PATH}")" || versions_exit_code=$?

  case "${versions_exit_code}" in
    0)
      jq --arg version "${version}" 'map(select(.version == $version))[0]' <<< "${versions}"
      ;;
    "${NOT_FOUND_EXIT_CODE}")
      echo "null"
      ;;
    *)
      cat "${VERSIONS_STDERR_PATH}" >&2
      return 1
      ;;
  esac
}

compute_record_status() {
  jq --raw-output --arg missing "${MISSING_RECORD_STATUS}" '.status // $missing' <<< "$1"
}

compute_operation_id() {
  jq --raw-output '.operation_id // ""' <<< "$1"
}

PYPROJECT_PATH="${PROVIDER_DIRECTORY}/pyproject.toml"
if [ ! -f "${PYPROJECT_PATH}" ]; then
  echo "publish_platform_provider: pyproject not found at ${PYPROJECT_PATH}" >&2
  exit "${FAILURE_EXIT_CODE}"
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
  exit "${FAILURE_EXIT_CODE}"
fi

PROVIDER_NAME="${PROVIDER_NAMESPACE}/${PROVIDER_SHORT_NAME}"
VERSION_LABEL="${PROVIDER_NAME} ${VERSION}"
CHECK_HINT="Check it with: pragma providers versions ${PROVIDER_NAME}"

UV_TOOL_DIR="${WORK_DIRECTORY}/tools" UV_TOOL_BIN_DIR="${WORK_DIRECTORY}/bin" \
  uv tool install --quiet --with "${PRAGMA_SDK_REQUIREMENT}" "${PRAGMA_CLI_REQUIREMENT}"

if ! PREVIOUS_CATALOG_VERSION="$(read_catalog_version "${PROVIDER_NAME}" "${VERSION}")"; then
  echo "publish_platform_provider: ${VERSION_LABEL}:" \
    "the catalog could not be read to check whether the version is already published" >&2
  exit "${FAILURE_EXIT_CODE}"
fi

PREVIOUS_STATUS="$(compute_record_status "${PREVIOUS_CATALOG_VERSION}")"
if [ "${PREVIOUS_STATUS}" = "${PUBLISHED_STATUS}" ]; then
  echo "publish_platform_provider: ${VERSION_LABEL}: already published"
  exit 0
fi

uv build "${PROVIDER_DIRECTORY}" --wheel --no-sources --out-dir "${WHEEL_DIRECTORY}"

shopt -s nullglob
WHEELS=("${WHEEL_DIRECTORY}"/*.whl)
shopt -u nullglob
if [ "${#WHEELS[@]}" -ne 1 ]; then
  echo "publish_platform_provider: expected exactly one wheel, found ${#WHEELS[@]}" >&2
  exit "${FAILURE_EXIT_CODE}"
fi

PUBLISH_STARTED_SECONDS="${SECONDS}"
PUBLISH_EXIT_CODE=0
PUBLISH_DOCUMENT="$(run_pragma providers publish --wheel "${WHEELS[0]}" --output json)" \
  || PUBLISH_EXIT_CODE=$?

case "${PUBLISH_EXIT_CODE}" in
  0)
    exit 0
    ;;
  "${ADMISSION_UNCONFIRMED_EXIT_CODE}")
    OPERATION_ID="$(compute_operation_id "${PUBLISH_DOCUMENT}")"
    if [ -z "${OPERATION_ID}" ]; then
      echo "publish_platform_provider: ${VERSION_LABEL}:" \
        "publish exited ${PUBLISH_EXIT_CODE} without an operation id. ${CHECK_HINT}" >&2
      exit "${FAILURE_EXIT_CODE}"
    fi
    echo "publish_platform_provider: ${VERSION_LABEL}:" \
      "wheel accepted, the version has not settled yet; polling the catalog"
    ;;
  *)
    echo "publish_platform_provider: ${VERSION_LABEL}:" \
      "publish failed (pragma exit ${PUBLISH_EXIT_CODE}), see the pragma CLI error above" >&2
    exit "${FAILURE_EXIT_CODE}"
    ;;
esac

while true; do
  if CATALOG_VERSION="$(read_catalog_version "${PROVIDER_NAME}" "${VERSION}")"; then
    STATUS="$(compute_record_status "${CATALOG_VERSION}")"
  else
    STATUS="${UNREADABLE_CATALOG_STATUS}"
  fi

  case "${STATUS}" in
    "${UNREADABLE_CATALOG_STATUS}" | "${MISSING_RECORD_STATUS}")
      ;;
    *)
      if [ "$(compute_operation_id "${CATALOG_VERSION}")" != "${OPERATION_ID}" ]; then
        STATUS="${REPLACED_RECORD_STATUS}"
      fi
      ;;
  esac

  case "${STATUS}" in
    "${MISSING_RECORD_STATUS}")
      echo "publish_platform_provider: ${VERSION_LABEL}:" \
        "the catalog has no record of the accepted wheel. ${CHECK_HINT}" >&2
      exit "${FAILURE_EXIT_CODE}"
      ;;
    "${REPLACED_RECORD_STATUS}")
      echo "publish_platform_provider: ${VERSION_LABEL}:" \
        "replaced by another publish of this version. ${CHECK_HINT}" >&2
      exit "${FAILURE_EXIT_CODE}"
      ;;
    "${PUBLISHED_STATUS}")
      echo "publish_platform_provider: ${VERSION_LABEL}: published"
      exit 0
      ;;
    "${FAILED_STATUS}")
      ERROR_MESSAGE="$(jq --raw-output '.error_message // ""' <<< "${CATALOG_VERSION}")"
      echo "publish_platform_provider: ${VERSION_LABEL}:" \
        "failed admission: ${ERROR_MESSAGE}" >&2
      exit "${FAILURE_EXIT_CODE}"
      ;;
    "${UNREADABLE_CATALOG_STATUS}")
      echo "publish_platform_provider: ${VERSION_LABEL}: the catalog could not be read"
      ;;
    *)
      echo "publish_platform_provider: ${VERSION_LABEL}: still being published: ${STATUS}"
      ;;
  esac

  if [ $((SECONDS - PUBLISH_STARTED_SECONDS)) -ge "${PUBLISH_DEADLINE_SECONDS}" ]; then
    echo "publish_platform_provider: ${VERSION_LABEL}:" \
      "not published after ${PUBLISH_DEADLINE_SECONDS}s. ${CHECK_HINT}" >&2
    exit "${FAILURE_EXIT_CODE}"
  fi

  sleep "${POLL_INTERVAL_SECONDS}"
done
