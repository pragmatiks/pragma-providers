# Agno runner container image.
#
# Serves one or more Agno agents, teams, and workflows as an HTTP API using
# a single AgentOS instance. Entities are reconstructed at startup from the
# AGNO_SPECS_JSON environment variable.
#
# This image is tagged with a major version (v2, v3, ...) that matches the
# transport contract expected by the agno provider's Runner resource. The
# wire format is otherwise unversioned -- bump the tag whenever the env-var
# contract changes and update the Runner default image in lockstep.
#
# Build from the repository root, against the workspace lock:
#   docker build -f packages/agno/runner/Dockerfile \
#     -t ghcr.io/pragmatiks/agno-runner:v2 \
#     -t ghcr.io/pragmatiks/agno-runner:latest .
#
# Optional: pass the current provider version so it is recorded as an image label:
#   docker build -f packages/agno/runner/Dockerfile \
#     --build-arg PROVIDER_VERSION="$(cd packages/agno && cz version --project)" \
#     -t ghcr.io/pragmatiks/agno-runner:v2 \
#     -t ghcr.io/pragmatiks/agno-runner:latest .

ARG PYTHON_BASE_IMAGE=python:3.13-slim@sha256:bf44cdfcb76cd3b41e879bc058fc37ec5872002ccfde7fcb765e218cde0cd79c

FROM ${PYTHON_BASE_IMAGE} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.20@sha256:100047e74f30778ab704942321a09750d6158739573ff58bf3924085cc6cd2d8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY packages ./packages

RUN uv sync --frozen --package pragmatiks-agno --no-dev --no-editable

# --- Final image ---
FROM ${PYTHON_BASE_IMAGE}

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
