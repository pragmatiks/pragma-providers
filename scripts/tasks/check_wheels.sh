#!/usr/bin/env bash
#
# Build every workspace provider's wheel once as publishing builds it, with
# scripts/build_provider_wheel.sh, then check each provider in packages/*/ against that release set with
# scripts/check_provider_wheel.sh.
#
# A provider floor above a sibling's in-tree version is not released yet: CI bumps the sibling at
# release, before this check runs on the bumped tree. scripts/provider_wheel.py unreleased-floors warns
# for each such floor and writes a uv override that lets the sibling's in-tree wheel satisfy every
# requirement on it; any other unsatisfied requirement fails, and on a bumped tree no override is written.
#
# Usage:
#   check_wheels.sh
#
# Requires: uv and yq at the versions mise.toml pins.
#
# Exit codes: 0 when every provider passes; otherwise the exit code of the first failing build or
# provider check, which stops the run.

set -euo pipefail

WORKSPACE_DIRECTORY="$(cd "$(dirname "$0")/../.." && pwd)"
SCRATCH_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "${SCRATCH_DIRECTORY}"' EXIT
BUILD_DIRECTORY="${SCRATCH_DIRECTORY}/builds"
WHEEL_DIRECTORY="${SCRATCH_DIRECTORY}/wheels"
OVERRIDE_PATH="${SCRATCH_DIRECTORY}/overrides.txt"
mkdir -p "${WHEEL_DIRECTORY}"

for provider_directory in "${WORKSPACE_DIRECTORY}"/packages/*/; do
  provider_directory="${provider_directory%/}"
  provider_build_directory="${BUILD_DIRECTORY}/$(basename "${provider_directory}")"
  version="$(yq --input-format toml '.project.version' "${provider_directory}/pyproject.toml")"
  "${WORKSPACE_DIRECTORY}/scripts/build_provider_wheel.sh" \
    "${provider_directory}" "${version}" "${provider_build_directory}"
  mv "${provider_build_directory}"/*.whl "${WHEEL_DIRECTORY}/"
done

uv run --isolated --no-project --quiet --with packaging python "${WORKSPACE_DIRECTORY}/scripts/provider_wheel.py" \
  unreleased-floors "${WHEEL_DIRECTORY}" > "${OVERRIDE_PATH}"

for provider_directory in "${WORKSPACE_DIRECTORY}"/packages/*/; do
  "${WORKSPACE_DIRECTORY}/scripts/check_provider_wheel.sh" "${provider_directory%/}" "${WHEEL_DIRECTORY}" "${OVERRIDE_PATH}"
done
