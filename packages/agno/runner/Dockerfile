# Agno runner container image.
#
# Serves an Agno agent or team as an HTTP API using AgentOS.
# The agent/team is reconstructed from spec JSON passed via environment variables.
#
# Build from agno package root:
#   docker build -f runner/Dockerfile -t ghcr.io/pragmatiks/agno-runner:latest .

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --no-sources --no-dev --no-install-project --frozen

COPY src ./src

RUN uv sync --no-sources --no-dev --no-editable --frozen

# --- Final image ---
FROM python:3.13-slim

# Node.js is needed for MCP tools that use npx
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY runner/server.py /app/runner/server.py

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["uvicorn", "runner.server:app", "--host", "0.0.0.0", "--port", "8000"]
