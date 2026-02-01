"""Tests for Agno provider."""

from __future__ import annotations

from agno_provider import (
    Agent,
    AgentConfig,
    AgentOutputs,
    ModelOutputs,
    OpenAIModel,
    OpenAIModelConfig,
    agno,
)


def test_provider_name() -> None:
    """Provider has correct name."""
    assert agno.name == "agno"


def test_agent_registered() -> None:
    """Agent resource is registered with provider."""
    assert Agent.provider == "agno"
    assert Agent.resource == "agent"


def test_agent_config_model() -> None:
    """AgentConfig can be exported."""
    assert AgentConfig is not None


def test_agent_outputs_model() -> None:
    """AgentOutputs can be exported."""
    assert AgentOutputs is not None


def test_openai_model_registered() -> None:
    """OpenAI model resource is registered with provider."""
    assert OpenAIModel.provider == "agno"
    assert OpenAIModel.resource == "models/openai"


def test_openai_model_config_model() -> None:
    """OpenAIModelConfig can be exported."""
    assert OpenAIModelConfig is not None


def test_openai_model_outputs_model() -> None:
    """ModelOutputs can be exported."""
    assert ModelOutputs is not None
