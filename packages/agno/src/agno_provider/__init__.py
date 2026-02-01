"""Agno agent provider for Pragmatiks.

Deploys Agno AI agents to Kubernetes clusters with reactive
dependency management for models, embeddings, and vector stores.
"""

from pragma_sdk import Provider

from agno_provider.resources import (
    Agent,
    AgentConfig,
    AgentOutputs,
    AnthropicModel,
    AnthropicModelConfig,
    AnthropicModelOutputs,
    DbPostgres,
    DbPostgresConfig,
    DbPostgresOutputs,
    OpenAIModel,
    OpenAIModelConfig,
    OpenAIModelOutputs,
)


agno = Provider(name="agno")

agno.resource("agent")(Agent)
agno.resource("db/postgres")(DbPostgres)
agno.resource("models/anthropic")(AnthropicModel)
agno.resource("models/openai")(OpenAIModel)

__all__ = [
    "agno",
    "Agent",
    "AgentConfig",
    "AgentOutputs",
    "AnthropicModel",
    "AnthropicModelConfig",
    "AnthropicModelOutputs",
    "DbPostgres",
    "DbPostgresConfig",
    "DbPostgresOutputs",
    "OpenAIModel",
    "OpenAIModelConfig",
    "OpenAIModelOutputs",
]
