"""Agno runner server - reconstructs agent/team from spec and serves via AgentOS.

Reads AGNO_SPEC_TYPE and AGNO_SPEC_JSON environment variables to reconstruct
an Agno agent or team, then serves it as an HTTP API using AgentOS on port 8000.

Environment variables:
    AGNO_SPEC_TYPE: "agent" or "team"
    AGNO_SPEC_JSON: JSON-serialized AgentSpec or TeamSpec
"""

from __future__ import annotations

import json
import os
import sys

from agno_provider.resources.agent import Agent, AgentSpec
from agno_provider.resources.team import Team, TeamSpec


def build_app():
    """Build the FastAPI app from environment variables.

    Reads AGNO_SPEC_TYPE and AGNO_SPEC_JSON, reconstructs the agent or team,
    and wraps it in an AgentOS instance.

    Returns:
        FastAPI application ready to serve.
    """
    from agno.os.app import AgentOS  # noqa: PLC0415

    spec_type = os.environ.get("AGNO_SPEC_TYPE")
    spec_json = os.environ.get("AGNO_SPEC_JSON")

    if not spec_type:
        print("AGNO_SPEC_TYPE environment variable is required", file=sys.stderr)
        sys.exit(1)

    if not spec_json:
        print("AGNO_SPEC_JSON environment variable is required", file=sys.stderr)
        sys.exit(1)

    authorization = bool(os.environ.get("JWT_VERIFICATION_KEY"))

    spec_data = json.loads(spec_json)

    if spec_type == "agent":
        spec = AgentSpec.model_validate(spec_data)
        agent = Agent.from_spec(spec)

        agent_os = AgentOS(
            name=spec.name,
            agents=[agent],
            authorization=authorization,
            telemetry=False,
        )

    elif spec_type == "team":
        spec = TeamSpec.model_validate(spec_data)
        team = Team.from_spec(spec)

        agent_os = AgentOS(
            name=spec.name,
            teams=[team],
            authorization=authorization,
            telemetry=False,
        )

    else:
        print(f"AGNO_SPEC_TYPE must be 'agent' or 'team', got: {spec_type}", file=sys.stderr)
        sys.exit(1)

    return agent_os.get_app()


app = build_app()
