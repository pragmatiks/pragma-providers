#!/usr/bin/env bash
#
# Check that every provider's requirements on its workspace siblings resolve without workspace
# sources, the way a published wheel's requirements resolve.
#
# Usage:
#   check_floors.sh
#
# Runs uv lock --dry-run --no-sources, which keeps workspace members satisfying each other but applies
# each requirement's specifier against the member's in-tree version, which is its last released one.
# A failure whose resolver message names only requirements of one provider on another
# (pragmatiks-<dependent> depends on pragmatiks-<sibling>) with a floor above the sibling's in-tree
# version is printed as a GitHub Actions warning: the sibling releases first in the same publish run,
# so the dependent publishes after it. The check does not read the Pragmatiks registry, so a floor
# below the sibling version that introduced the names a provider imports is not caught.
#
# Exit codes: 0 when the workspace resolves or every failure is an unpublished sibling floor; 1 for any
# other resolve failure, such as a third-party conflict or a network error.

set -euo pipefail

FAILURE_EXIT_CODE=1
WORKSPACE_DIRECTORY="$(cd "$(dirname "$0")/../.." && pwd)"
WORKSPACE_REQUIREMENT_PATTERN='pragmatiks-[a-z]+ depends on pragmatiks-[a-z]+'

read_member_version() {
  sed -n 's/^version = "\(.*\)"$/\1/p' "${WORKSPACE_DIRECTORY}/packages/$1/pyproject.toml" | head -n 1
}

read_requirement_floor() {
  sed -n "s/^ *\"$2>=\([^\",;]*\).*/\1/p" "${WORKSPACE_DIRECTORY}/packages/$1/pyproject.toml" | head -n 1
}

is_version_above() {
  [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n 1)" = "$1" ]
}

warn_unpublished_floor() {
  local dependent="$1"
  local sibling="$2"
  local floor
  local version

  floor="$(read_requirement_floor "${dependent#pragmatiks-}" "${sibling}")"
  version="$(read_member_version "${sibling#pragmatiks-}")"
  if [ -z "${floor}" ] || [ -z "${version}" ] || ! is_version_above "${floor}" "${version}"; then
    return 1
  fi

  echo "::warning::${dependent} requires ${sibling}>=${floor}; in-tree ${version} is not published yet"
}

if resolve_output="$(cd "${WORKSPACE_DIRECTORY}" && uv lock --dry-run --no-sources 2>&1)"; then
  echo "check_floors: every provider's sibling requirements resolve without workspace sources"
  exit 0
fi

echo "${resolve_output}" >&2
requirements="$(grep -oE "${WORKSPACE_REQUIREMENT_PATTERN}" <<< "${resolve_output}" | sort -u || true)"
if [ -z "${requirements}" ]; then
  echo "check_floors: the resolve failed for a reason other than a sibling floor" >&2
  exit "${FAILURE_EXIT_CODE}"
fi

while read -r dependent _ _ sibling; do
  if ! warn_unpublished_floor "${dependent}" "${sibling}"; then
    echo "check_floors: ${dependent} on ${sibling} does not resolve, and its floor is not above the in-tree version" >&2
    exit "${FAILURE_EXIT_CODE}"
  fi
done <<< "${requirements}"
