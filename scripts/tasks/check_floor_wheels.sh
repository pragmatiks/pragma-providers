#!/usr/bin/env bash
#
# Check every provider in packages/*/ against the lowest released sibling versions its requirements
# allow, so a floor below the sibling release that introduced a name the provider imports fails.
#
# Usage:
#   check_floor_wheels.sh
#
# For each provider, scripts/provider_wheel.py floor-tags resolves every sibling it reaches to the
# lowest <provider>-v<version> tag satisfying the requirement, read again at that tag for the
# sibling's own floors. Each floor tag's package is built from the tag's tree with
# uv build --wheel --no-sources, the command publishing uses, beside the provider's own wheel built
# from the working tree, and scripts/check_provider_wheel.sh installs that set (third-party
# requirements from the package index) and runs pragma_sdk.introspection. A floor no tag satisfies
# yet warns, and the sibling's in-tree wheel stands in.
#
# Requires: git with every <provider>-v<version> tag fetched, and uv at the pinned version.
#
# Exit codes: 0 when every provider passes; otherwise the exit code of the first failing build or
# provider check, which stops the run.

set -euo pipefail

WORKSPACE_DIRECTORY="$(cd "$(dirname "$0")/../.." && pwd)"
HELPER="${WORKSPACE_DIRECTORY}/scripts/provider_wheel.py"
SCRATCH_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "${SCRATCH_DIRECTORY}"' EXIT

run_helper() {
  uv run --isolated --no-project --quiet --with packaging python "${HELPER}" "$@"
}

build_tag_wheel() {
  local tag="$1"
  local out_directory="$2"
  local tree_directory="${SCRATCH_DIRECTORY}/trees/${tag}"
  local wheel_cache="${SCRATCH_DIRECTORY}/tags/${tag}"

  if [ ! -d "${wheel_cache}" ]; then
    mkdir -p "${tree_directory}" "${wheel_cache}"
    git -C "${WORKSPACE_DIRECTORY}" archive "${tag}" "packages/${tag%-v*}" | tar -x -C "${tree_directory}"
    uv build "${tree_directory}/packages/${tag%-v*}" --wheel --no-sources --quiet --out-dir "${wheel_cache}"
  fi
  cp "${wheel_cache}"/*.whl "${out_directory}/"
}

for provider_directory in "${WORKSPACE_DIRECTORY}"/packages/*/; do
  provider_directory="${provider_directory%/}"
  distribution="$(run_helper name "${provider_directory}/pyproject.toml")"
  mapfile -t floor_tags < <(run_helper floor-tags "${WORKSPACE_DIRECTORY}" "${distribution}")
  if [ "${#floor_tags[@]}" -eq 0 ]; then
    continue
  fi

  wheel_directory="${SCRATCH_DIRECTORY}/sets/${distribution}"
  mkdir -p "${wheel_directory}"
  uv build "${provider_directory}" --wheel --no-sources --quiet --out-dir "${wheel_directory}"
  for tag in "${floor_tags[@]}"; do
    if [ "${tag}" = "${tag%-v*}" ]; then
      uv build "${WORKSPACE_DIRECTORY}/packages/${tag}" --wheel --no-sources --quiet --out-dir "${wheel_directory}"
    else
      build_tag_wheel "${tag}" "${wheel_directory}"
    fi
  done
  echo "check_floors: ${distribution} against ${floor_tags[*]}"
  "${WORKSPACE_DIRECTORY}/scripts/check_provider_wheel.sh" "${provider_directory}" "${wheel_directory}"
done
