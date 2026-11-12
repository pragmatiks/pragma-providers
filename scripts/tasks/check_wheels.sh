#!/usr/bin/env bash
#
# Build every workspace provider's wheel once, the way publishing builds it, then check each
# provider in packages/*/ against that release set with scripts/check_provider_wheel.sh.
#
# A provider floor above a sibling's in-tree version is not released yet: CI bumps the sibling at
# release, before this check runs on the bumped tree. scripts/provider_wheel.py unreleased-floors warns
# for each such floor and writes a uv override that lets the sibling's in-tree wheel satisfy every
# requirement on it; any other unsatisfied requirement fails, and on a bumped tree no override is written.
#
# Usage:
#   check_wheels.sh
#
# Exit codes: 0 when every provider passes; otherwise the exit code of the first failing build or
# provider check, which stops the run.

set -euo pipefail

WORKSPACE_DIRECTORY="$(cd "$(dirname "$0")/../.." && pwd)"
SCRATCH_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "${SCRATCH_DIRECTORY}"' EXIT
WHEEL_DIRECTORY="${SCRATCH_DIRECTORY}/wheels"
OVERRIDE_PATH="${SCRATCH_DIRECTORY}/overrides.txt"

for provider_directory in "${WORKSPACE_DIRECTORY}"/packages/*/; do
  uv build "${provider_directory}" --wheel --no-sources --quiet --out-dir "${WHEEL_DIRECTORY}"
done

uv run --isolated --no-project --quiet --with packaging python "${WORKSPACE_DIRECTORY}/scripts/provider_wheel.py" \
  unreleased-floors "${WHEEL_DIRECTORY}" > "${OVERRIDE_PATH}"

for provider_directory in "${WORKSPACE_DIRECTORY}"/packages/*/; do
  "${WORKSPACE_DIRECTORY}/scripts/check_provider_wheel.sh" "${provider_directory%/}" "${WHEEL_DIRECTORY}" "${OVERRIDE_PATH}"
done
