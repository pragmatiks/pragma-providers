"""Agno runner server - reconstructs agents and teams from specs.

Reads the AGNO_SPECS_JSON environment variable to reconstruct one or more
Agno agents and teams, then serves them as an HTTP API using a single
AgentOS instance on port 8000.

Environment variables:
    AGNO_SPECS_JSON: JSON payload with shape
        {"agents": [<AgentSpec>...], "teams": [<TeamSpec>...], "workflows": []}
        The ``workflows`` key is reserved and must be empty or absent.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, NoReturn

from agno_provider.resources.agent import Agent, AgentSpec
from agno_provider.resources.team import Team, TeamSpec


logger = logging.getLogger("agno_runner")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def _exit_with_error(message: str) -> NoReturn:
    """Log a structured error and exit with status 1.

    Args:
        message: Human-readable error message describing what failed.
    """
    logger.error("runner_startup_failed: %s", message)
    print(f"runner_startup_failed: {message}", file=sys.stderr)
    sys.exit(1)


def _load_specs_json(raw: str) -> dict[str, Any]:
    """Decode the AGNO_SPECS_JSON string into a dict.

    Args:
        raw: Raw JSON payload from the AGNO_SPECS_JSON env var.

    Returns:
        Top-level dict with ``agents``/``teams``/``workflows`` keys.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _exit_with_error(f"AGNO_SPECS_JSON is not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}")

    if not isinstance(payload, dict):
        _exit_with_error(f"AGNO_SPECS_JSON must be a JSON object, got {type(payload).__name__}")

    return payload


def _extract_entity_dicts(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Read a list of spec dicts from the payload, validating the shape.

    Args:
        payload: Parsed AGNO_SPECS_JSON payload.
        key: Entity list key (e.g. ``"agents"``, ``"teams"``, ``"workflows"``).

    Returns:
        List of spec dicts, empty if the key is missing or null.
    """
    value = payload.get(key, []) or []

    if not isinstance(value, list):
        _exit_with_error(f"AGNO_SPECS_JSON.{key} must be a list, got {type(value).__name__}")

    for index, item in enumerate(value):
        if not isinstance(item, dict):
            _exit_with_error(f"AGNO_SPECS_JSON.{key}[{index}] must be an object, got {type(item).__name__}")

    return value


def _parse_specs_payload(raw: str) -> tuple[list[AgentSpec], list[TeamSpec]]:
    """Parse the AGNO_SPECS_JSON payload into typed spec lists.

    Args:
        raw: Raw JSON string from the AGNO_SPECS_JSON env var.

    Returns:
        Tuple of (agent_specs, team_specs) parsed from the payload.
    """
    payload = _load_specs_json(raw)

    agent_dicts = _extract_entity_dicts(payload, "agents")
    team_dicts = _extract_entity_dicts(payload, "teams")
    workflow_dicts = _extract_entity_dicts(payload, "workflows")

    if workflow_dicts:
        _exit_with_error("AGNO_SPECS_JSON.workflows is reserved and must be empty in this runner image")

    agent_specs: list[AgentSpec] = []
    for index, item in enumerate(agent_dicts):
        try:
            agent_specs.append(AgentSpec.model_validate(item))
        except Exception as exc:
            name = item.get("name") if isinstance(item, dict) else None
            label = f"name={name!r}" if name else f"index={index}"
            _exit_with_error(f"AGNO_SPECS_JSON.agents[{index}] is not a valid AgentSpec ({label}): {exc}")

    team_specs: list[TeamSpec] = []
    for index, item in enumerate(team_dicts):
        try:
            team_specs.append(TeamSpec.model_validate(item))
        except Exception as exc:
            name = item.get("name") if isinstance(item, dict) else None
            label = f"name={name!r}" if name else f"index={index}"
            _exit_with_error(f"AGNO_SPECS_JSON.teams[{index}] is not a valid TeamSpec ({label}): {exc}")

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


def _build_agent(spec: AgentSpec) -> Agent:
    """Reconstruct a single Agent from its spec with a clear error surface.

    Args:
        spec: Parsed AgentSpec to hydrate.

    Returns:
        Fully-hydrated Agent instance.
    """
    try:
        return Agent.from_spec(spec)
    except Exception as exc:
        _exit_with_error(f"failed to reconstruct agent name={spec.name!r}: {type(exc).__name__}: {exc}")


def _build_team(spec: TeamSpec) -> Team:
    """Reconstruct a single Team from its spec with a clear error surface.

    Args:
        spec: Parsed TeamSpec to hydrate.

    Returns:
        Fully-hydrated Team instance.
    """
    try:
        return Team.from_spec(spec)
    except Exception as exc:
        _exit_with_error(f"failed to reconstruct team name={spec.name!r}: {type(exc).__name__}: {exc}")


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

    authorization = bool(os.environ.get("JWT_VERIFICATION_KEY"))

    agent_specs, team_specs = _parse_specs_payload(specs_json)

    agents = [_build_agent(spec) for spec in agent_specs]
    teams = [_build_team(spec) for spec in team_specs]

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
