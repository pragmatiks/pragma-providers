"""Agno Team resource - pure definition for AI team configuration."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar

from agno.team import Team as AgnoTeam
from pragma_sdk import Config, Dependency, Outputs
from pydantic import Field, field_validator

from agno_provider.resources.agent import Agent, AgentOutputs, AgentSpec
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


class TeamSpec(AgnoSpec):
    """Specification for reconstructing an Agno Team at runtime.

    Contains all necessary information to create a Team instance
    with all member agents and dependencies. Used for deployment to containers
    where the team needs to be reconstructed from serialized config.
    """

    name: str
    description: str | None = None
    role: str | None = None
    instructions: list[str] | None = None

    member_specs: list[AgentSpec]

    model_spec: Annotated[OpenAIModelSpec | AnthropicModelSpec, Field(discriminator="type")] | None = None

    tools_specs: list[ToolsMCPSpec | ToolsWebSearchSpec] = []
    knowledge_spec: KnowledgeSpec | None = None
    memory_spec: MemoryManagerSpec | None = None
    db_spec: DbPostgresSpec | None = None
    prompt_spec: PromptSpec | None = None

    respond_directly: bool = False
    delegate_to_all_members: bool = False
    markdown: bool = False
    add_datetime_to_context: bool = False
    read_chat_history: bool | None = None
    add_history_to_context: bool | None = None
    num_history_runs: int | None = None
    enable_agentic_memory: bool = False
    update_memory_on_run: bool = False
    add_memories_to_context: bool | None = None
    enable_session_summaries: bool = False


class TeamConfig(Config):
    """Configuration for an Agno team definition."""

    name: str | None = None
    description: str | None = None
    role: str | None = None

    members: list[Dependency[Agent]]

    @field_validator("members")
    @classmethod
    def validate_members_not_empty(cls, v: list[Dependency[Agent]]) -> list[Dependency[Agent]]:
        """Validate that at least one member is provided.

        Returns:
            The validated members list.

        Raises:
            ValueError: If members list is empty.
        """
        if not v:
            msg = "Team must have at least one member"
            raise ValueError(msg)

        return v

    model: Dependency[AnthropicModel] | Dependency[OpenAIModel] | None = None

    instructions: list[str] | None = None
    prompt: Dependency[Prompt] | None = None

    tools: list[Dependency[ToolsMCP] | Dependency[ToolsWebSearch]] = []

    knowledge: Dependency[Knowledge] | None = None

    db: Dependency[DbPostgres] | None = None

    memory: Dependency[MemoryManager] | None = None

    respond_directly: bool = False
    delegate_to_all_members: bool = False
    markdown: bool = False
    add_datetime_to_context: bool = False
    read_chat_history: bool | None = None
    add_history_to_context: bool | None = None
    num_history_runs: int | None = None
    enable_agentic_memory: bool = False
    update_memory_on_run: bool = False
    add_memories_to_context: bool | None = None
    enable_session_summaries: bool = False


class TeamOutputs(Outputs):
    """Outputs from Agno team definition.

    Attributes:
        spec: Specification for reconstructing the team at runtime.
        member_count: Number of member agents in the team.
        pip_dependencies: Python packages required by this team and its members.
    """

    spec: TeamSpec
    member_count: int
    pip_dependencies: list[str]


class Team(AgnoResource[TeamConfig, TeamOutputs, TeamSpec]):
    """Agno AI team definition resource.

    A pure configuration wrapper that produces a serializable TeamSpec.
    No deployment logic - the spec is used by other resources (e.g., deployment)
    to reconstruct the team at runtime.

    The outputs include a TeamSpec that can be used to reconstruct
    the complete team via Team.from_spec().

    Example YAML:
        provider: agno
        resource: team
        name: my-team
        config:
          members:
            - provider: agno
              resource: agent
              name: researcher
            - provider: agno
              resource: agent
              name: writer
          model:
            provider: agno
            resource: models/anthropic
            name: claude
          delegate_to_all_members: true
          markdown: true

    Runtime reconstruction via spec:
        team = Team.from_spec(spec)

    Lifecycle:
        - on_create: Resolve dependencies, return outputs with spec
        - on_update: Re-resolve dependencies, return updated outputs
        - on_delete: No-op (stateless wrapper)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "team"

    @staticmethod
    def from_spec(spec: TeamSpec) -> AgnoTeam:
        """Factory: construct Agno Team from spec.

        Builds all nested dependencies from their specs and constructs
        the Team with all configured components. Member agents are
        reconstructed first, then the team is built with those members.

        Args:
            spec: The team specification.

        Returns:
            Configured Agno Team instance.
        """
        members = [Agent.from_spec(member_spec) for member_spec in spec.member_specs]

        model = None
        if spec.model_spec:
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

        return AgnoTeam(
            name=spec.name,
            description=spec.description,
            role=spec.role,
            members=members,
            model=model,
            knowledge=knowledge,
            memory_manager=memory_manager,
            db=db,
            tools=tools if tools else None,
            instructions=instructions,
            respond_directly=spec.respond_directly,
            delegate_to_all_members=spec.delegate_to_all_members,
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

    async def _build_spec(self) -> TeamSpec:
        """Build spec from resolved dependencies.

        Creates a specification that can be serialized and used to
        reconstruct the team at runtime. Extracts nested specs from
        all resolved dependency outputs.

        Returns:
            TeamSpec with all nested specs from dependencies.

        Raises:
            RuntimeError: If a member or model dependency is not resolved or has no spec.
        """
        member_specs: list[AgentSpec] = []
        for member_dep in self.config.members:
            member = await member_dep.resolve()

            if member.outputs is None:
                msg = f"Member agent dependency not resolved: {member.name}"
                raise RuntimeError(msg)

            assert isinstance(member.outputs, AgentOutputs)
            member_specs.append(member.outputs.spec)

        model_spec: Annotated[OpenAIModelSpec | AnthropicModelSpec, Field(discriminator="type")] | None = None
        if self.config.model is not None:
            model = await self.config.model.resolve()
            model_outputs = model.outputs

            if model_outputs is None:
                msg = "Model dependency not resolved"
                raise RuntimeError(msg)

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

        team_name = self.config.name if self.config.name else self.name

        return TeamSpec(
            name=team_name,
            description=self.config.description,
            role=self.config.role,
            instructions=self.config.instructions,
            member_specs=member_specs,
            model_spec=model_spec,
            tools_specs=tools_specs,
            knowledge_spec=knowledge_spec,
            memory_spec=memory_spec,
            db_spec=db_spec,
            prompt_spec=prompt_spec,
            respond_directly=self.config.respond_directly,
            delegate_to_all_members=self.config.delegate_to_all_members,
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
        """Aggregate pip dependencies from all members and team dependencies.

        Note: This method accesses dependency._resolved directly instead of using
        await resolve(). This is intentional because _build_spec() is always called
        first (in _build_outputs), which resolves all dependencies. This method is
        synchronous and simply reads the already-resolved values.

        Returns:
            Deduplicated list of pip packages required.
        """
        deps: set[str] = set()

        for member_dep in self.config.members:
            member = member_dep._resolved
            if member is not None and member.outputs is not None:
                assert isinstance(member.outputs, AgentOutputs)
                deps.update(member.outputs.pip_dependencies)

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

    async def _build_outputs(self) -> TeamOutputs:
        """Build outputs with spec and pip dependencies.

        Returns:
            TeamOutputs with the spec, member_count, and pip_dependencies.
        """
        spec = await self._build_spec()

        return TeamOutputs(
            spec=spec,
            member_count=len(spec.member_specs),
            pip_dependencies=self._get_pip_dependencies(),
        )

    async def on_create(self) -> TeamOutputs:
        """Create team definition and return serializable outputs.

        Idempotent: Simply resolves dependencies and builds the spec.

        Returns:
            TeamOutputs with spec, member_count, and pip_dependencies.
        """
        return await self._build_outputs()

    async def on_update(self, previous_config: TeamConfig) -> TeamOutputs:  # noqa: ARG002
        """Update team definition and return serializable outputs.

        Args:
            previous_config: The previous configuration (unused for stateless resource).

        Returns:
            TeamOutputs with updated spec, member_count, and pip_dependencies.
        """
        return await self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
