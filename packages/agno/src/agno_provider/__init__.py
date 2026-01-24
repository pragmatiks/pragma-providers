"""Agno agent provider for Pragmatiks.

Deploys Agno AI agents to Kubernetes clusters with reactive
dependency management for models, embeddings, and vector stores.
"""

from pragma_sdk import Provider

from agno_provider.resources import (
    Agent,
    AgentConfig,
    AgentOutputs,
)

agno = Provider(name="agno")

# Register resources
agno.resource("agent")(Agent)

__all__ = [
    "agno",
    "Agent",
    "AgentConfig",
    "AgentOutputs",
]
