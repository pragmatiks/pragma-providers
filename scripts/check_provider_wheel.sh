#!/usr/bin/env bash
#
# Check that a provider's publishable wheel loads and declares its embeds the way the Pragmatiks
# registry admits it, against a directory of wheels built for one release.
#
# Usage:
#   check_provider_wheel.sh <provider_dir> <wheel_dir> [override_file]
#
# Arguments:
#   provider_dir  Provider package directory, such as packages/qdrant.
#   wheel_dir     Directory holding every workspace provider's wheel built with
#                 uv build --wheel --no-sources (scripts/tasks/check_wheels.sh builds it).
#   override_file Optional uv overrides file of `<distribution> @ file://<wheel>` lines; each such wheel
#                 satisfies every requirement on its distribution, whatever the version
#                 (scripts/tasks/check_wheels.sh and scripts/tasks/check_floor_wheels.sh write one for
#                 an in-tree sibling standing in for a floor not released yet).
#
# The provider's wheel is installed in an isolated environment with only the wheels in wheel_dir its
# Requires-Dist reaches without an extra, directly or through the chain, through uv tool run, the uv
# command whose install takes an overrides file; every other requirement resolves from the package
# index. The check fails when pragma_sdk.introspection reports a failure,
# or when the pragma.embeds keys differ from the providers among those reached wheels, or name the
# provider itself.
#
# Requires: uv at the version the workspace pyproject.toml pins.
#
# Exit codes: 0 when both checks pass; 1 on a check failure; 2 on a usage error; a failing uv
# install or run exits with uv's own non-zero exit code.

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "check_provider_wheel: usage: check_provider_wheel.sh <provider_dir> <wheel_dir> [override_file]" >&2
  exit 2
fi

PROVIDER_DIRECTORY="$1"
WHEEL_DIRECTORY="$2"
HELPER="$(cd "$(dirname "$0")" && pwd)/provider_wheel.py"

REPORT_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "${REPORT_DIRECTORY}"' EXIT
REPORT_PATH="${REPORT_DIRECTORY}/report.json"

run_helper() {
  uv run --isolated --no-project --quiet --with packaging python "${HELPER}" "$@"
}

DISTRIBUTION="$(run_helper name "${PROVIDER_DIRECTORY}/pyproject.toml")"
mapfile -t RELEASE_SET_WHEELS < <(run_helper release-set "${WHEEL_DIRECTORY}" "${DISTRIBUTION}")

INSTALL_ARGUMENTS=(--from "${RELEASE_SET_WHEELS[0]}")
for wheel_path in "${RELEASE_SET_WHEELS[@]:1}"; do
  INSTALL_ARGUMENTS+=(--with "${wheel_path}")
done
if [ "$#" -eq 3 ]; then
  INSTALL_ARGUMENTS+=(--overrides "$3")
fi

run_in_release_set() {
  uv tool run --isolated --quiet "${INSTALL_ARGUMENTS[@]}" "$@"
}

PRAGMA_PROVIDER_DISTRIBUTION="${DISTRIBUTION}" \
  run_in_release_set python -m pragma_sdk.introspection --output "${REPORT_PATH}"
run_helper report "${DISTRIBUTION}" "${REPORT_PATH}"
run_helper embeds "${WHEEL_DIRECTORY}" "${DISTRIBUTION}"
