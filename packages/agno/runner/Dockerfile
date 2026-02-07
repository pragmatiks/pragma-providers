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

# Copy package metadata and lock for dependency resolution
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY README.md ./

# Install all dependencies (excluding dev group) into a venv.
# Remove [tool.uv.sources] to avoid workspace-relative paths,
# then install deps from lock, then install the package itself without deps.
RUN uv export --no-sources --no-group dev --no-editable --no-emit-project > requirements.txt && \
    uv venv && \
    uv pip install -r requirements.txt && \
    python -c "import re; f='pyproject.toml'; open(f,'w').write(re.sub(r'\[tool\.uv\.sources\][^\[]*','',open(f).read()))" && \
    uv pip install --no-deps .

# Copy runner server code
COPY runner/server.py ./runner/server.py

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
COPY --from=builder /app/runner /app/runner

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["uvicorn", "runner.server:app", "--host", "0.0.0.0", "--port", "8000"]
