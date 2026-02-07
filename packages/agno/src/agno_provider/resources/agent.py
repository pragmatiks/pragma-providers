"""Agno Agent resource - pure definition for AI agent configuration."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar

from agno.agent import Agent as AgnoAgent
from pragma_sdk import Config, Dependency, Outputs
from pydantic import Field

from agno_provider.resources.base import AgnoResource, AgnoSpec
from agno_provider.resources.db.postgres import DbPostgres, DbPostgresSpec
from agno_provider.resources.knowledge.knowledge import Knowledge, KnowledgeOutputs, KnowledgeSpec
from agno_provider.resources.memory.manager import MemoryManager, MemoryManagerSpec
from agno_provider.resources.models import model_from_spec
from agno_provider.resources.models.anthropic import (
    AnthropicModel,
    AnthropicModelOutputs,
    AnthropicModelSpec,
)
from agno_provider.resources.models.openai import (
    OpenAIModel,
    OpenAIModelOutputs,
    OpenAIModelSpec,
)
from agno_provider.resources.prompt import Prompt, PromptSpec
from agno_provider.resources.tools.mcp import ToolsMCP, ToolsMCPOutputs, ToolsMCPSpec
from agno_provider.resources.tools.websearch import ToolsWebSearch, ToolsWebSearchOutputs, ToolsWebSearchSpec


class AgentSpec(AgnoSpec):
    """Specification for reconstructing an Agno Agent at runtime.

    Contains all necessary information to create an Agent instance
    with all nested dependencies. Used for deployment to containers
    where the agent needs to be reconstructed from serialized config.
    """

    name: str
    description: str | None = None
    role: str | None = None
    instructions: list[str] | None = None
    model_spec: Annotated[OpenAIModelSpec | AnthropicModelSpec, Field(discriminator="type")]
    tools_specs: list[ToolsMCPSpec | ToolsWebSearchSpec] = []
    knowledge_spec: KnowledgeSpec | None = None
    memory_spec: MemoryManagerSpec | None = None
    db_spec: DbPostgresSpec | None = None
    prompt_spec: PromptSpec | None = None
    markdown: bool = False
    add_datetime_to_context: bool = False
    read_chat_history: bool | None = None
    add_history_to_context: bool | None = None
    num_history_runs: int | None = None
    enable_agentic_memory: bool = False
    update_memory_on_run: bool = False
    add_memories_to_context: bool | None = None
    enable_session_summaries: bool = False


class AgentConfig(Config):
    """Configuration for an Agno agent definition."""

    name: str | None = None
    description: str | None = None
    role: str | None = None

    model: Dependency[AnthropicModel] | Dependency[OpenAIModel]

    instructions: list[str] | None = None
    prompt: Dependency[Prompt] | None = None

    tools: list[Dependency[ToolsMCP] | Dependency[ToolsWebSearch]] = []

    knowledge: Dependency[Knowledge] | None = None

    db: Dependency[DbPostgres] | None = None

    memory: Dependency[MemoryManager] | None = None

    markdown: bool = False
    add_datetime_to_context: bool = False
    read_chat_history: bool | None = None
    add_history_to_context: bool | None = None
    num_history_runs: int | None = None
    enable_agentic_memory: bool = False
    update_memory_on_run: bool = False
    add_memories_to_context: bool | None = None
    enable_session_summaries: bool = False


class AgentOutputs(Outputs):
    """Outputs from Agno agent definition.

    Attributes:
        spec: Specification for reconstructing the agent at runtime.
        pip_dependencies: Python packages required by this agent.
    """

    spec: AgentSpec
    pip_dependencies: list[str]


class Agent(AgnoResource[AgentConfig, AgentOutputs, AgentSpec]):
    """Agno AI agent definition resource.

    A pure configuration wrapper that produces a serializable AgentSpec.
    No deployment logic - the spec is used by other resources (e.g., deployment)
    to reconstruct the agent at runtime.

    The outputs include an AgentSpec that can be used to reconstruct
    the complete agent via Agent.from_spec().

    Example YAML:
        provider: agno
        resource: agent
        name: my-agent
        config:
          model:
            provider: agno
            resource: models/anthropic
            name: claude
          instructions:
            - "You are a helpful assistant."
          tools:
            - provider: agno
              resource: tools/mcp
              name: search
          markdown: true

    Runtime reconstruction via spec:
        agent = Agent.from_spec(spec)

    Lifecycle:
        - on_create: Resolve dependencies, return outputs with spec
        - on_update: Re-resolve dependencies, return updated outputs
        - on_delete: No-op (stateless wrapper)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "agent"

    @staticmethod
    def from_spec(spec: AgentSpec) -> AgnoAgent:
        """Factory: construct Agno Agent from spec.

        Builds all nested dependencies from their specs and constructs
        the Agent with all configured components.

        Args:
            spec: The agent specification.

        Returns:
            Configured Agno Agent instance.
        """
        model = model_from_spec(spec.model_spec)

        knowledge = None
        if spec.knowledge_spec:
            knowledge = Knowledge.from_spec(spec.knowledge_spec)

        memory_manager = None
        if spec.memory_spec:
            memory_manager = MemoryManager.from_spec(spec.memory_spec)

        db = None
        if spec.db_spec:
            db = DbPostgres.from_spec(spec.db_spec)

        tools: list[Any] = []
        for tool_spec in spec.tools_specs:
            if isinstance(tool_spec, ToolsMCPSpec):
                tools.append(ToolsMCP.from_spec(tool_spec))
            else:
                tools.append(ToolsWebSearch.from_spec(tool_spec))

        instructions: str | list[str] | None = spec.instructions
        if spec.prompt_spec:
            instructions = Prompt.from_spec(spec.prompt_spec)

        read_chat_history = spec.read_chat_history
        if read_chat_history is None:
            read_chat_history = db is not None

        add_history_to_context = spec.add_history_to_context
        if add_history_to_context is None:
            add_history_to_context = db is not None

        return AgnoAgent(
            name=spec.name,
            description=spec.description,
            role=spec.role,
            model=model,
            knowledge=knowledge,
            memory_manager=memory_manager,
            db=db,
            tools=tools if tools else None,
            instructions=instructions,
            markdown=spec.markdown,
            add_datetime_to_context=spec.add_datetime_to_context,
            read_chat_history=read_chat_history,
            add_history_to_context=add_history_to_context,
            num_history_runs=spec.num_history_runs,
            enable_agentic_memory=spec.enable_agentic_memory,
            update_memory_on_run=spec.update_memory_on_run,
            add_memories_to_context=spec.add_memories_to_context,
            enable_session_summaries=spec.enable_session_summaries,
        )

    async def _build_spec(self) -> AgentSpec:
        """Build spec from resolved dependencies.

        Creates a specification that can be serialized and used to
        reconstruct the agent at runtime. Extracts nested specs from
        all resolved dependency outputs.

        Returns:
            AgentSpec with all nested specs from dependencies.

        Raises:
            RuntimeError: If model dependency is not resolved or has no spec.
        """
        model = await self.config.model.resolve()
        model_outputs = model.outputs

        if model_outputs is None:
            msg = "Model dependency not resolved"
            raise RuntimeError(msg)

        model_spec: Annotated[OpenAIModelSpec | AnthropicModelSpec, Field(discriminator="type")]
        if isinstance(model_outputs, OpenAIModelOutputs):
            model_spec = model_outputs.spec
        elif isinstance(model_outputs, AnthropicModelOutputs):
            model_spec = model_outputs.spec
        else:
            msg = f"Unsupported model outputs type: {type(model_outputs)}"
            raise RuntimeError(msg)

        knowledge_spec: KnowledgeSpec | None = None
        if self.config.knowledge is not None:
            kb = await self.config.knowledge.resolve()
            if kb.outputs is not None:
                knowledge_spec = kb.outputs.spec

        memory_spec: MemoryManagerSpec | None = None
        if self.config.memory is not None:
            memory = await self.config.memory.resolve()
            if memory.outputs is not None:
                memory_spec = memory.outputs.spec

        db_spec: DbPostgresSpec | None = None
        if self.config.db is not None:
            db = await self.config.db.resolve()
            if db.outputs is not None:
                db_spec = db.outputs.spec

        tools_specs: list[ToolsMCPSpec | ToolsWebSearchSpec] = []
        for tool_dep in self.config.tools:
            tool = await tool_dep.resolve()
            if tool.outputs is not None:
                if isinstance(tool.outputs, ToolsMCPOutputs):
                    tools_specs.append(tool.outputs.spec)
                elif isinstance(tool.outputs, ToolsWebSearchOutputs):
                    tools_specs.append(tool.outputs.spec)

        prompt_spec: PromptSpec | None = None
        if self.config.prompt is not None:
            prompt = await self.config.prompt.resolve()
            if prompt.outputs is not None:
                prompt_spec = prompt.outputs.spec

        agent_name = self.config.name if self.config.name else self.name

        return AgentSpec(
            name=agent_name,
            description=self.config.description,
            role=self.config.role,
            instructions=self.config.instructions,
            model_spec=model_spec,
            tools_specs=tools_specs,
            knowledge_spec=knowledge_spec,
            memory_spec=memory_spec,
            db_spec=db_spec,
            prompt_spec=prompt_spec,
            markdown=self.config.markdown,
            add_datetime_to_context=self.config.add_datetime_to_context,
            read_chat_history=self.config.read_chat_history,
            add_history_to_context=self.config.add_history_to_context,
            num_history_runs=self.config.num_history_runs,
            enable_agentic_memory=self.config.enable_agentic_memory,
            update_memory_on_run=self.config.update_memory_on_run,
            add_memories_to_context=self.config.add_memories_to_context,
            enable_session_summaries=self.config.enable_session_summaries,
        )

    def _get_pip_dependencies(self) -> list[str]:
        """Aggregate pip dependencies from all tool and knowledge dependencies.

        Note: This method accesses dependency._resolved directly instead of using
        await resolve(). This is intentional because _build_spec() is always called
        first (in _build_outputs), which resolves all dependencies. This method is
        synchronous and simply reads the already-resolved values.

        Returns:
            Deduplicated list of pip packages required.
        """
        deps: set[str] = set()

        for tool_dep in self.config.tools:
            tool = tool_dep._resolved
            if tool is not None and tool.outputs is not None:
                if isinstance(tool.outputs, ToolsWebSearchOutputs):
                    deps.update(tool.outputs.pip_dependencies)

        if self.config.knowledge is not None:
            kb = self.config.knowledge._resolved
            if kb is not None and kb.outputs is not None:
                assert isinstance(kb.outputs, KnowledgeOutputs)
                deps.update(kb.outputs.pip_dependencies)

        return sorted(deps)

    async def _build_outputs(self) -> AgentOutputs:
        """Build outputs with spec and pip dependencies.

        Returns:
            AgentOutputs with the spec and pip_dependencies.
        """
        spec = await self._build_spec()

        return AgentOutputs(
            spec=spec,
            pip_dependencies=self._get_pip_dependencies(),
        )

    async def on_create(self) -> AgentOutputs:
        """Create agent definition and return serializable outputs.

        Idempotent: Simply resolves dependencies and builds the spec.

        Returns:
            AgentOutputs with spec and pip_dependencies.
        """
        return await self._build_outputs()

    async def on_update(self, previous_config: AgentConfig) -> AgentOutputs:  # noqa: ARG002
        """Update agent definition and return serializable outputs.

        Args:
            previous_config: The previous configuration (unused for stateless resource).

        Returns:
            AgentOutputs with updated spec and pip_dependencies.
        """
        return await self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
