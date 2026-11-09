# Agno runner container image.
#
# Serves one or more Agno agents, teams, and workflows as an HTTP API using
# a single AgentOS instance. Entities are reconstructed at startup from the
# AGNO_SPECS_JSON environment variable.
#
# This image is tagged with a major version (v2, v3, ...) that matches the
# transport contract expected by the agno provider's Runner resource, and with
# the agno provider version it was released with. The wire format is otherwise
# unversioned -- bump the transport tag whenever the env-var contract changes
# and update the Runner default image in lockstep.
#
# uv is installed at the version the workspace pyproject.toml pins in
# [tool.uv] required-version, so the image and the workspace cannot disagree.
# Third-party dependencies are installed from the workspace manifests and lock
# alone, so a change to provider source reuses that layer.
#
# Build from the repository root, against the workspace lock:
#   docker build -f packages/agno/runner/Dockerfile -t agno-runner:dev .
#
# Optional: pass the provider version so it is recorded as an image label:
#   docker build -f packages/agno/runner/Dockerfile \
#     --build-arg PROVIDER_VERSION="$(cd packages/agno && cz version --project)" \
#     -t agno-runner:dev .

FROM python:3.13-slim@sha256:bf44cdfcb76cd3b41e879bc058fc37ec5872002ccfde7fcb765e218cde0cd79c AS manifests

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY packages ./packages

RUN find packages -mindepth 2 -maxdepth 2 ! -name pyproject.toml -exec rm -rf {} +

FROM python:3.13-slim@sha256:bf44cdfcb76cd3b41e879bc058fc37ec5872002ccfde7fcb765e218cde0cd79c AS builder

ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN --mount=type=bind,source=pyproject.toml,target=/tmp/workspace-pyproject.toml \
    UV_SPECIFIER="$(sed -n 's/^required-version = "\(.*\)"$/\1/p' /tmp/workspace-pyproject.toml)" && \
    test -n "${UV_SPECIFIER}" && \
    pip install --no-cache-dir --root-user-action=ignore "uv${UV_SPECIFIER}"

COPY --from=manifests /app ./

RUN uv sync --frozen --package pragmatiks-agno --no-dev --no-editable --no-install-workspace

COPY packages ./packages

RUN uv sync --frozen --package pragmatiks-agno --no-dev --no-editable

# --- Final image ---
FROM python:3.13-slim@sha256:bf44cdfcb76cd3b41e879bc058fc37ec5872002ccfde7fcb765e218cde0cd79c

ARG PROVIDER_VERSION=dev

LABEL org.opencontainers.image.title="pragma-agno-runner"
LABEL org.opencontainers.image.description="Runtime container for Pragma agno agents and teams."
LABEL org.opencontainers.image.source="https://github.com/pragmatiks/pragma-providers"
LABEL org.opencontainers.image.version="${PROVIDER_VERSION}"
LABEL ai.pragmatiks.agno-runner.features="output_schema,multi_entity"
LABEL ai.pragmatiks.agno-runner.transport="v2"

# Node.js is needed for MCP tools that use npx
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://deb.nodesource.com/setup_26.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY packages/agno/runner/server.py /app/runner/server.py

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["uvicorn", "runner.server:app", "--host", "0.0.0.0", "--port", "8000"]
