"""Agno runner server - reconstructs agents, teams, and workflows from specs.

Reads the AGNO_SPECS_JSON environment variable to reconstruct one or more
Agno agents and teams, then serves them as an HTTP API using a single
AgentOS instance on port 8000.

Environment variables:
    AGNO_SPECS_JSON: JSON payload with shape
        {"agents": [<AgentSpec>...], "teams": [<TeamSpec>...], "workflows": []}
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from agno_provider.resources.agent import Agent, AgentSpec
from agno_provider.resources.team import Team, TeamSpec


def _exit_with_error(message: str) -> None:
    """Log an error to stderr and exit with status 1.

    Args:
        message: Human-readable error message for stderr.
    """
    print(message, file=sys.stderr)
    sys.exit(1)


def _parse_specs_payload(raw: str) -> tuple[list[AgentSpec], list[TeamSpec]]:
    """Parse the AGNO_SPECS_JSON payload into typed spec lists.

    Args:
        raw: Raw JSON string from the AGNO_SPECS_JSON env var.

    Returns:
        Tuple of (agent_specs, team_specs) parsed from the payload.
    """
    payload: dict[str, Any] = json.loads(raw)

    agent_dicts: list[dict[str, Any]] = payload.get("agents", []) or []
    team_dicts: list[dict[str, Any]] = payload.get("teams", []) or []
    workflow_dicts: list[dict[str, Any]] = payload.get("workflows", []) or []

    if workflow_dicts:
        _exit_with_error("AGNO_SPECS_JSON.workflows is reserved and must be empty in this runner image")

    agent_specs = [AgentSpec.model_validate(item) for item in agent_dicts]
    team_specs = [TeamSpec.model_validate(item) for item in team_dicts]

    if not agent_specs and not team_specs:
        _exit_with_error("AGNO_SPECS_JSON must include at least one agent or team spec")

    return agent_specs, team_specs


def _build_agent_os_name(agent_specs: list[AgentSpec], team_specs: list[TeamSpec]) -> str:
    """Derive a stable AgentOS instance name from the first provided entity.

    Args:
        agent_specs: Parsed agent specs.
        team_specs: Parsed team specs.

    Returns:
        Name string used by AgentOS for identification.
    """
    if agent_specs:
        return agent_specs[0].name

    return team_specs[0].name


def build_app():  # noqa: ANN201
    """Build the FastAPI app from the AGNO_SPECS_JSON environment variable.

    Reconstructs every agent and team from the payload and registers all of
    them with a single AgentOS instance.

    Returns:
        FastAPI application ready to serve.
    """
    from agno.os.app import AgentOS  # noqa: PLC0415

    specs_json = os.environ.get("AGNO_SPECS_JSON")

    if not specs_json:
        _exit_with_error("AGNO_SPECS_JSON environment variable is required")
        return None

    authorization = bool(os.environ.get("JWT_VERIFICATION_KEY"))

    agent_specs, team_specs = _parse_specs_payload(specs_json)

    agents = [Agent.from_spec(spec) for spec in agent_specs]
    teams = [Team.from_spec(spec) for spec in team_specs]

    agent_os = AgentOS(
        name=_build_agent_os_name(agent_specs, team_specs),
        agents=agents,
        teams=teams,
        workflows=[],
        authorization=authorization,
        telemetry=False,
    )

    return agent_os.get_app()


app = build_app()
