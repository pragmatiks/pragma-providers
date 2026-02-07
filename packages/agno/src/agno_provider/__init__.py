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
    DbPostgres,
    DbPostgresConfig,
    DbPostgresOutputs,
    EmbedderOpenAI,
    EmbedderOpenAIConfig,
    EmbedderOpenAIOutputs,
    MemoryManager,
    MemoryManagerConfig,
    MemoryManagerOutputs,
    Model,
    ModelConfig,
    ModelOutputs,
    OpenAIModel,
    OpenAIModelConfig,
    Prompt,
    PromptConfig,
    PromptOutputs,
    Runner,
    RunnerConfig,
    RunnerOutputs,
    Team,
    TeamConfig,
    TeamOutputs,
    ToolsMCP,
    ToolsMCPConfig,
    ToolsMCPOutputs,
    ToolsWebSearch,
    ToolsWebSearchConfig,
    ToolsWebSearchOutputs,
    VectordbQdrant,
    VectordbQdrantConfig,
    VectordbQdrantOutputs,
)


agno = Provider(name="agno")

agno.resource("agent")(Agent)
agno.resource("db/postgres")(DbPostgres)
agno.resource("runner")(Runner)
agno.resource("knowledge/embedder/openai")(EmbedderOpenAI)
agno.resource("memory/manager")(MemoryManager)
agno.resource("models/anthropic")(AnthropicModel)
agno.resource("models/openai")(OpenAIModel)
agno.resource("prompt")(Prompt)
agno.resource("team")(Team)
agno.resource("tools/mcp")(ToolsMCP)
agno.resource("tools/websearch")(ToolsWebSearch)
agno.resource("vectordb/qdrant")(VectordbQdrant)

__all__ = [
    "agno",
    "Agent",
    "AgentConfig",
    "AgentOutputs",
    "AnthropicModel",
    "AnthropicModelConfig",
    "DbPostgres",
    "DbPostgresConfig",
    "DbPostgresOutputs",
    "Runner",
    "RunnerConfig",
    "RunnerOutputs",
    "EmbedderOpenAI",
    "EmbedderOpenAIConfig",
    "EmbedderOpenAIOutputs",
    "MemoryManager",
    "MemoryManagerConfig",
    "MemoryManagerOutputs",
    "Model",
    "ModelConfig",
    "ModelOutputs",
    "OpenAIModel",
    "OpenAIModelConfig",
    "Prompt",
    "PromptConfig",
    "PromptOutputs",
    "ToolsMCP",
    "ToolsMCPConfig",
    "ToolsMCPOutputs",
    "Team",
    "TeamConfig",
    "TeamOutputs",
    "ToolsWebSearch",
    "ToolsWebSearchConfig",
    "ToolsWebSearchOutputs",
    "VectordbQdrant",
    "VectordbQdrantConfig",
    "VectordbQdrantOutputs",
]
