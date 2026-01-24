"""Tests for Agno provider."""

from __future__ import annotations

from agno_provider import (
    Agent,
    AgentConfig,
    AgentOutputs,
    agno,
)


def test_provider_name() -> None:
    """Provider has correct name."""
    assert agno.name == "agno"


def test_agent_registered() -> None:
    """Agent resource is registered with provider."""
    # The resource decorator registers the class
    assert Agent.provider == "agno"
    assert Agent.resource == "agent"


def test_agent_config_model() -> None:
    """AgentConfig can be exported."""
    assert AgentConfig is not None


def test_agent_outputs_model() -> None:
    """AgentOutputs can be exported."""
    assert AgentOutputs is not None
