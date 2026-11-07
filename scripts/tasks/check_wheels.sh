#!/usr/bin/env bash
#
# Build every workspace provider's wheel once, the way publishing builds it, then check each
# provider in packages/*/ against that release set with scripts/check_provider_wheel.sh.
#
# Usage:
#   check_wheels.sh
#
# Exit codes: 0 when every provider passes; otherwise the exit code of the first failing build or
# provider check, which stops the run.

set -euo pipefail

WORKSPACE_DIRECTORY="$(cd "$(dirname "$0")/../.." && pwd)"
WHEEL_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "${WHEEL_DIRECTORY}"' EXIT

for provider_directory in "${WORKSPACE_DIRECTORY}"/packages/*/; do
  uv build "${provider_directory}" --wheel --no-sources --quiet --out-dir "${WHEEL_DIRECTORY}"
done

for provider_directory in "${WORKSPACE_DIRECTORY}"/packages/*/; do
  "${WORKSPACE_DIRECTORY}/scripts/check_provider_wheel.sh" "${provider_directory%/}" "${WHEEL_DIRECTORY}"
done
