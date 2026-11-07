#!/usr/bin/env bash
set -euo pipefail

distribution=$(yq -p toml -oy '.project.name' pyproject.toml)

env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$MISE_CONFIG_ROOT/.venvs/$distribution" \
  uv run --package "$distribution" --exact --locked --group dev "$@"
