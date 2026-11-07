#!/usr/bin/env bash
#
# Build one first-party provider wheel the way the Pragmatiks registry receives it: isolated, with
# workspace sources ignored, with the provider commit's own uv configuration ignored, and with the
# uv_build backend constrained to the hash-locked version in scripts/publish_tooling/uv.lock.
#
# Usage:
#   build_provider_wheel.sh <provider_dir> <version> <wheel_dir>
#
# Arguments:
#   provider_dir  Provider package directory, such as packages/qdrant, from any commit. Its
#                 pyproject.toml declares [project].version and a [build-system] whose only requirement
#                 is uv_build, with a range that admits the locked uv_build version.
#   version       Version to build; must equal the pyproject version.
#   wheel_dir     Directory that receives the wheel; it must hold exactly one wheel afterwards.
#
# The build runs with uv --no-config, so the uv version mise.toml pins builds a provider directory whose
# commit requires another uv version or carries other [tool.uv] settings.
#
# Requires: uv at the version mise.toml pins, and scripts/publish_tooling next to this script; the
# working directory does not matter. Bumping the backend pin: edit scripts/publish_tooling/pyproject.toml,
# then mise run lock.
#
# Example:
#   build_provider_wheel.sh release-tag/packages/qdrant 7.0.1 "${RUNNER_TEMP}/wheels"
#
# Exits 1 when an argument is missing, the pyproject is missing or malformed, its version differs from
# <version>, its [build-system] cannot be satisfied by the locked uv_build, or <wheel_dir> does not hold
# exactly one wheel after the build. A failed constraint export, a backend hash mismatch or a failed
# build exits with uv's own non-zero exit code; uv's error is printed above.

set -euo pipefail

FAILURE_EXIT_CODE=1
TOOLING_PROJECT="$(cd "$(dirname "$0")/publish_tooling" && pwd)"

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
  echo "build_provider_wheel:" \
    "usage: build_provider_wheel.sh <provider_dir> <version> <wheel_dir>" >&2
  exit "${FAILURE_EXIT_CODE}"
fi

PROVIDER_DIRECTORY="$1"
VERSION="$2"
WHEEL_DIRECTORY="$3"

PYPROJECT_PATH="${PROVIDER_DIRECTORY}/pyproject.toml"
if [ ! -f "${PYPROJECT_PATH}" ]; then
  echo "build_provider_wheel: pyproject not found at ${PYPROJECT_PATH}" >&2
  exit "${FAILURE_EXIT_CODE}"
fi

CONSTRAINTS_PATH="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_PATH}"' EXIT

uv export \
  --project "${TOOLING_PROJECT}" \
  --frozen \
  --only-group build \
  --no-emit-project \
  --format requirements.txt \
  --quiet \
  --output-file "${CONSTRAINTS_PATH}"

if ! uv run \
  --project "${TOOLING_PROJECT}" \
  --locked \
  --no-build \
  --isolated \
  --quiet \
  python -I - "${PYPROJECT_PATH}" "${VERSION}" "${CONSTRAINTS_PATH}" <<'PYTHON'
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

pyproject_path, expected_version, constraints_path = sys.argv[1:]

with open(pyproject_path, "rb") as handle:
    pyproject = tomllib.load(handle)

version = pyproject.get("project", {}).get("version")
if not version:
    sys.exit(f"{pyproject_path} needs [project].version")
if version != expected_version:
    sys.exit(f"version '{expected_version}' differs from {pyproject_path}: '{version}'")

with open(constraints_path) as handle:
    pin_lines = [line.split(";")[0].strip().rstrip("\\").strip() for line in handle if line[:1].isalnum()]
pins = [Requirement(pin_line) for pin_line in pin_lines]
uv_build_pins = [pin for pin in pins if canonicalize_name(pin.name) == "uv-build"]
backend_versions = [next(iter(pin.specifier)).version for pin in uv_build_pins]
if len(backend_versions) != 1:
    sys.exit("scripts/publish_tooling/uv.lock pins no single uv_build version")
backend_version = backend_versions[0]

build_system = pyproject.get("build-system", {})
requirements = [Requirement(requirement) for requirement in build_system.get("requires", [])]
admitted = (
    build_system.get("build-backend") == "uv_build"
    and len(requirements) == 1
    and canonicalize_name(requirements[0].name) == "uv-build"
    and requirements[0].specifier.contains(backend_version, prereleases=True)
)
if not admitted:
    sys.exit(
        f"{pyproject_path} [build-system] must be build-backend uv_build with exactly one requirement "
        f"on uv_build admitting the locked uv_build {backend_version}; it has {build_system}"
    )
PYTHON
then
  echo "build_provider_wheel: ${PYPROJECT_PATH} cannot be built as published" >&2
  exit "${FAILURE_EXIT_CODE}"
fi

uv build "${PROVIDER_DIRECTORY}" \
  --no-config \
  --wheel \
  --no-sources \
  --build-constraints "${CONSTRAINTS_PATH}" \
  --require-hashes \
  --quiet \
  --out-dir "${WHEEL_DIRECTORY}"

shopt -s nullglob
WHEELS=("${WHEEL_DIRECTORY}"/*.whl)
shopt -u nullglob
if [ "${#WHEELS[@]}" -ne 1 ]; then
  echo "build_provider_wheel: expected exactly one wheel in ${WHEEL_DIRECTORY}, found ${#WHEELS[@]}" >&2
  exit "${FAILURE_EXIT_CODE}"
fi
