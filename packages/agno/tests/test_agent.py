"""Tests for Agno Agent resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pragma_sdk import Dependency, LifecycleState
from pragma_sdk.provider import ProviderHarness

from agno_provider import (
    Agent,
    AgentConfig,
    AgentOutputs,
    AnthropicModel,
    AnthropicModelConfig,
    DbPostgres,
    DbPostgresConfig,
    OpenAIModel,
    OpenAIModelConfig,
    Prompt,
    PromptConfig,
    ToolsMCP,
    ToolsMCPConfig,
    ToolsWebSearch,
    ToolsWebSearchConfig,
)
from agno_provider.resources.agent import AgentSpec
from agno_provider.resources.db.postgres import DbPostgresOutputs, DbPostgresSpec
from agno_provider.resources.knowledge.knowledge import (
    Knowledge,
    KnowledgeConfig,
    KnowledgeOutputs,
    KnowledgeSpec,
)
from agno_provider.resources.memory import MemoryManager, MemoryManagerConfig, MemoryManagerOutputs
from agno_provider.resources.memory.manager import MemoryManagerSpec
from agno_provider.resources.models.anthropic import AnthropicModelOutputs, AnthropicModelSpec
from agno_provider.resources.models.openai import OpenAIModelOutputs, OpenAIModelSpec
from agno_provider.resources.prompt import PromptOutputs, PromptSpec
from agno_provider.resources.tools.mcp import ToolsMCPOutputs, ToolsMCPSpec
from agno_provider.resources.tools.websearch import ToolsWebSearchOutputs, ToolsWebSearchSpec
from agno_provider.resources.vectordb.qdrant import (
    VectordbQdrant,
    VectordbQdrantConfig,
    VectordbQdrantOutputs,
    VectordbQdrantSpec,
)


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_anthropic_model_resource(name: str = "claude") -> AnthropicModel:
    """Create an AnthropicModel resource for testing."""
    config = AnthropicModelConfig(
        id="claude-sonnet-4-20250514",
        api_key="sk-test-key",
    )
    resource = AnthropicModel(name=name, config=config, lifecycle_state=LifecycleState.READY)
    resource.outputs = AnthropicModelOutputs(
        spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test-key"),
    )
    return resource


def create_openai_model_resource(name: str = "gpt4") -> OpenAIModel:
    """Create an OpenAIModel resource for testing."""
    config = OpenAIModelConfig(
        id="gpt-4o",
        api_key="sk-test-key",
    )
    resource = OpenAIModel(name=name, config=config, lifecycle_state=LifecycleState.READY)
    resource.outputs = OpenAIModelOutputs(
        spec=OpenAIModelSpec(id="gpt-4o", api_key="sk-test-key"),
    )
    return resource


def create_anthropic_model_dependency(name: str = "claude") -> Dependency[AnthropicModel]:
    """Create an AnthropicModel dependency with resolved resource."""
    dep = Dependency[AnthropicModel](
        provider="agno",
        resource="models/anthropic",
        name=name,
    )
    dep._resolved = create_anthropic_model_resource(name)
    return dep


def create_openai_model_dependency(name: str = "gpt4") -> Dependency[OpenAIModel]:
    """Create an OpenAIModel dependency with resolved resource."""
    dep = Dependency[OpenAIModel](
        provider="agno",
        resource="models/openai",
        name=name,
    )
    dep._resolved = create_openai_model_resource(name)
    return dep


def create_tools_mcp_resource(name: str = "mcp-tools") -> ToolsMCP:
    """Create a ToolsMCP resource for testing."""
    config = ToolsMCPConfig(url="http://localhost:8080/sse")
    resource = ToolsMCP(name=name, config=config, lifecycle_state=LifecycleState.READY)
    resource.outputs = ToolsMCPOutputs(
        spec=ToolsMCPSpec(url="http://localhost:8080/sse", transport="sse"),
    )
    return resource


def create_tools_mcp_dependency(name: str = "mcp-tools") -> Dependency[ToolsMCP]:
    """Create a ToolsMCP dependency with resolved resource."""
    dep = Dependency[ToolsMCP](
        provider="agno",
        resource="tools/mcp",
        name=name,
    )
    dep._resolved = create_tools_mcp_resource(name)
    return dep


def create_tools_websearch_resource(name: str = "websearch") -> ToolsWebSearch:
    """Create a ToolsWebSearch resource for testing."""
    config = ToolsWebSearchConfig()
    resource = ToolsWebSearch(name=name, config=config, lifecycle_state=LifecycleState.READY)
    resource.outputs = ToolsWebSearchOutputs(
        pip_dependencies=["ddgs>=8.0.0"],
        spec=ToolsWebSearchSpec(enable_search=True, enable_news=True),
    )
    return resource


def create_tools_websearch_dependency(name: str = "websearch") -> Dependency[ToolsWebSearch]:
    """Create a ToolsWebSearch dependency with resolved resource."""
    dep = Dependency[ToolsWebSearch](
        provider="agno",
        resource="tools/websearch",
        name=name,
    )
    dep._resolved = create_tools_websearch_resource(name)
    return dep


def create_knowledge_resource(name: str = "knowledge") -> Knowledge:
    """Create a Knowledge resource for testing."""
    vectordb_dep = Dependency[VectordbQdrant](
        provider="agno",
        resource="vectordb/qdrant",
        name="test-vectordb",
    )
    vectordb = VectordbQdrant(
        name="test-vectordb",
        config=VectordbQdrantConfig(url="http://localhost:6333", collection="test"),
        lifecycle_state=LifecycleState.READY,
    )
    vectordb.outputs = VectordbQdrantOutputs(
        spec=VectordbQdrantSpec(url="http://localhost:6333", collection="test"),
        pip_dependencies=["qdrant-client"],
    )
    vectordb_dep._resolved = vectordb

    config = KnowledgeConfig(vector_db=vectordb_dep, max_results=5)
    resource = Knowledge(name=name, config=config, lifecycle_state=LifecycleState.READY)
    resource.outputs = KnowledgeOutputs(
        pip_dependencies=["qdrant-client"],
        spec=KnowledgeSpec(
            name=name,
            max_results=5,
            vector_db_spec=VectordbQdrantSpec(url="http://localhost:6333", collection="test"),
        ),
    )
    return resource


def create_knowledge_dependency(name: str = "knowledge") -> Dependency[Knowledge]:
    """Create a Knowledge dependency with resolved resource."""
    dep = Dependency[Knowledge](
        provider="agno",
        resource="knowledge",
        name=name,
    )
    dep._resolved = create_knowledge_resource(name)
    return dep


def create_storage_resource(name: str = "storage") -> DbPostgres:
    """Create a DbPostgres resource for testing."""
    config = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/db",
        db_schema="ai",
    )
    resource = DbPostgres(name=name, config=config, lifecycle_state=LifecycleState.READY)
    resource.outputs = DbPostgresOutputs(
        spec=DbPostgresSpec(
            connection_url="postgresql://user:pass@localhost:5432/db",
            db_schema="ai",
        ),
    )
    return resource


def create_storage_dependency(name: str = "storage") -> Dependency[DbPostgres]:
    """Create a DbPostgres dependency with resolved resource."""
    dep = Dependency[DbPostgres](
        provider="agno",
        resource="db/postgres",
        name=name,
    )
    dep._resolved = create_storage_resource(name)
    return dep


def create_memory_resource(name: str = "memory") -> MemoryManager:
    """Create a MemoryManager resource for testing."""
    storage_dep = create_storage_dependency()
    config = MemoryManagerConfig(db=storage_dep)
    resource = MemoryManager(name=name, config=config, lifecycle_state=LifecycleState.READY)
    resource.outputs = MemoryManagerOutputs(
        spec=MemoryManagerSpec(
            db_spec=DbPostgresSpec(
                connection_url="postgresql://user:pass@localhost:5432/db",
                db_schema="ai",
            ),
        ),
    )
    return resource


def create_memory_dependency(name: str = "memory") -> Dependency[MemoryManager]:
    """Create a MemoryManager dependency with resolved resource."""
    dep = Dependency[MemoryManager](
        provider="agno",
        resource="memory/manager",
        name=name,
    )
    dep._resolved = create_memory_resource(name)
    return dep


def create_prompt_resource(name: str = "prompt") -> Prompt:
    """Create a Prompt resource for testing."""
    config = PromptConfig(instructions=["You are a helpful assistant."])
    resource = Prompt(name=name, config=config, lifecycle_state=LifecycleState.READY)
    resource.outputs = PromptOutputs(
        spec=PromptSpec(
            instructions=["You are a helpful assistant."],
            rendered="You are a helpful assistant.",
        ),
    )
    return resource


def create_prompt_dependency(name: str = "prompt") -> Dependency[Prompt]:
    """Create a Prompt dependency with resolved resource."""
    dep = Dependency[Prompt](
        provider="agno",
        resource="prompt",
        name=name,
    )
    dep._resolved = create_prompt_resource(name)
    return dep


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert Agent.provider == "agno"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Agent.resource == "agent"


def test_config_with_anthropic_model() -> None:
    """Config accepts Anthropic model dependency."""
    model_dep = create_anthropic_model_dependency()

    config = AgentConfig(model=model_dep)

    assert config.model is not None
    assert config.model.name == "claude"
    assert config.instructions is None
    assert config.tools == []


def test_config_with_openai_model() -> None:
    """Config accepts OpenAI model dependency."""
    model_dep = create_openai_model_dependency()

    config = AgentConfig(model=model_dep)

    assert config.model is not None
    assert config.model.name == "gpt4"


def test_config_with_instructions() -> None:
    """Config accepts inline instructions."""
    model_dep = create_anthropic_model_dependency()

    config = AgentConfig(
        model=model_dep,
        instructions=["Be helpful.", "Be concise."],
    )

    assert config.instructions == ["Be helpful.", "Be concise."]


def test_config_with_tools() -> None:
    """Config accepts tools list."""
    model_dep = create_anthropic_model_dependency()
    mcp_dep = create_tools_mcp_dependency()
    websearch_dep = create_tools_websearch_dependency()

    config = AgentConfig(
        model=model_dep,
        tools=[mcp_dep, websearch_dep],
    )

    assert len(config.tools) == 2


def test_config_with_all_options() -> None:
    """Config accepts all optional fields."""
    model_dep = create_anthropic_model_dependency()
    knowledge_dep = create_knowledge_dependency()
    storage_dep = create_storage_dependency()
    memory_dep = create_memory_dependency()
    prompt_dep = create_prompt_dependency()

    config = AgentConfig(
        name="my-agent",
        description="A test agent",
        role="assistant",
        model=model_dep,
        instructions=["Be helpful."],
        prompt=prompt_dep,
        knowledge=knowledge_dep,
        db=storage_dep,
        memory=memory_dep,
        markdown=True,
        add_datetime_to_context=True,
    )

    assert config.name == "my-agent"
    assert config.description == "A test agent"
    assert config.role == "assistant"
    assert config.prompt is not None
    assert config.knowledge is not None
    assert config.db is not None
    assert config.memory is not None
    assert config.markdown is True
    assert config.add_datetime_to_context is True


def test_config_defaults() -> None:
    """Config has correct default values."""
    model_dep = create_anthropic_model_dependency()
    config = AgentConfig(model=model_dep)

    assert config.name is None
    assert config.description is None
    assert config.role is None
    assert config.instructions is None
    assert config.prompt is None
    assert config.tools == []
    assert config.knowledge is None
    assert config.db is None
    assert config.memory is None
    assert config.markdown is False
    assert config.add_datetime_to_context is False


def test_outputs_are_serializable() -> None:
    """Outputs contain only serializable data."""
    outputs = AgentOutputs(
        spec=AgentSpec(
            name="test-agent",
            model_spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test"),
        ),
        pip_dependencies=[],
    )

    assert outputs.spec.name == "test-agent"

    serialized = outputs.model_dump_json()
    assert "test-agent" in serialized
    assert "claude-sonnet-4-20250514" in serialized


def test_outputs_with_nested_specs() -> None:
    """Outputs serialize nested specs correctly."""
    outputs = AgentOutputs(
        spec=AgentSpec(
            name="test-agent",
            description="Test description",
            role="assistant",
            instructions=["Be helpful."],
            model_spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test"),
            tools_specs=[
                ToolsMCPSpec(url="http://localhost:8080", transport="sse"),
                ToolsWebSearchSpec(enable_search=True, enable_news=False),
            ],
            knowledge_spec=KnowledgeSpec(
                name="kb",
                vector_db_spec=VectordbQdrantSpec(url="http://localhost:6333", collection="test"),
            ),
            db_spec=DbPostgresSpec(connection_url="postgresql://localhost/db", db_schema="ai"),
            memory_spec=MemoryManagerSpec(
                db_spec=DbPostgresSpec(connection_url="postgresql://localhost/db", db_schema="ai"),
            ),
            prompt_spec=PromptSpec(instructions=["Base instruction."], rendered="Base instruction."),
            markdown=True,
            add_datetime_to_context=True,
        ),
        pip_dependencies=["ddgs>=8.0.0", "qdrant-client"],
    )

    serialized = outputs.model_dump_json()

    assert "test-agent" in serialized
    assert "claude-sonnet-4-20250514" in serialized
    assert "Be helpful." in serialized
    assert "sse" in serialized
    assert "qdrant-client" in serialized


async def test_create_with_model_only(harness: ProviderHarness) -> None:
    """on_create with only model returns outputs with spec."""
    model_dep = create_anthropic_model_dependency()
    config = AgentConfig(model=model_dep)

    result = await harness.invoke_create(Agent, name="test-agent", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.name == "test-agent"
    assert result.outputs.spec.model_spec.id == "claude-sonnet-4-20250514"
    assert result.outputs.pip_dependencies == []


async def test_create_with_openai_model(harness: ProviderHarness) -> None:
    """on_create with OpenAI model returns outputs with correct model spec."""
    model_dep = create_openai_model_dependency()
    config = AgentConfig(model=model_dep)

    result = await harness.invoke_create(Agent, name="test-agent", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.model_spec.id == "gpt-4o"


async def test_create_with_instructions(harness: ProviderHarness) -> None:
    """on_create with instructions includes them in spec."""
    model_dep = create_anthropic_model_dependency()
    config = AgentConfig(
        model=model_dep,
        instructions=["Be helpful.", "Be concise."],
    )

    result = await harness.invoke_create(Agent, name="test-agent", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.instructions == ["Be helpful.", "Be concise."]


async def test_create_with_custom_name(harness: ProviderHarness) -> None:
    """on_create with custom name uses it in spec."""
    model_dep = create_anthropic_model_dependency()
    config = AgentConfig(
        name="custom-agent-name",
        model=model_dep,
    )

    result = await harness.invoke_create(Agent, name="resource-name", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.name == "custom-agent-name"


async def test_create_with_tools(harness: ProviderHarness) -> None:
    """on_create with tools includes tool specs."""
    model_dep = create_anthropic_model_dependency()
    mcp_dep = create_tools_mcp_dependency()
    websearch_dep = create_tools_websearch_dependency()

    config = AgentConfig(
        model=model_dep,
        tools=[mcp_dep, websearch_dep],
    )

    result = await harness.invoke_create(Agent, name="test-agent", config=config)

    assert result.success
    assert result.outputs is not None
    assert len(result.outputs.spec.tools_specs) == 2
    assert result.outputs.pip_dependencies == ["ddgs>=8.0.0"]


async def test_create_with_knowledge(harness: ProviderHarness) -> None:
    """on_create with knowledge includes knowledge spec and pip dependencies."""
    model_dep = create_anthropic_model_dependency()
    knowledge_dep = create_knowledge_dependency()

    config = AgentConfig(
        model=model_dep,
        knowledge=knowledge_dep,
    )

    result = await harness.invoke_create(Agent, name="test-agent", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.knowledge_spec is not None
    assert result.outputs.spec.knowledge_spec.name == "knowledge"
    assert "qdrant-client" in result.outputs.pip_dependencies


async def test_create_with_db(harness: ProviderHarness) -> None:
    """on_create with db includes db spec."""
    model_dep = create_anthropic_model_dependency()
    storage_dep = create_storage_dependency()

    config = AgentConfig(
        model=model_dep,
        db=storage_dep,
    )

    result = await harness.invoke_create(Agent, name="test-agent", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.db_spec is not None
    assert result.outputs.spec.db_spec.db_schema == "ai"


async def test_create_with_memory(harness: ProviderHarness) -> None:
    """on_create with memory includes memory spec."""
    model_dep = create_anthropic_model_dependency()
    memory_dep = create_memory_dependency()

    config = AgentConfig(
        model=model_dep,
        memory=memory_dep,
    )

    result = await harness.invoke_create(Agent, name="test-agent", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.memory_spec is not None
    assert result.outputs.spec.memory_spec.db_spec is not None


async def test_create_with_prompt(harness: ProviderHarness) -> None:
    """on_create with prompt includes prompt spec."""
    model_dep = create_anthropic_model_dependency()
    prompt_dep = create_prompt_dependency()

    config = AgentConfig(
        model=model_dep,
        prompt=prompt_dep,
    )

    result = await harness.invoke_create(Agent, name="test-agent", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.prompt_spec is not None
    assert result.outputs.spec.prompt_spec.rendered == "You are a helpful assistant."


async def test_create_with_all_options(harness: ProviderHarness) -> None:
    """on_create with all options returns complete spec."""
    model_dep = create_anthropic_model_dependency()
    mcp_dep = create_tools_mcp_dependency()
    knowledge_dep = create_knowledge_dependency()
    storage_dep = create_storage_dependency()
    memory_dep = create_memory_dependency()

    config = AgentConfig(
        name="full-agent",
        description="A fully configured agent",
        role="expert assistant",
        model=model_dep,
        instructions=["Be thorough."],
        tools=[mcp_dep],
        knowledge=knowledge_dep,
        db=storage_dep,
        memory=memory_dep,
        markdown=True,
        add_datetime_to_context=True,
    )

    result = await harness.invoke_create(Agent, name="test-agent", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.name == "full-agent"
    assert result.outputs.spec.description == "A fully configured agent"
    assert result.outputs.spec.role == "expert assistant"
    assert result.outputs.spec.instructions == ["Be thorough."]
    assert len(result.outputs.spec.tools_specs) == 1
    assert result.outputs.spec.knowledge_spec is not None
    assert result.outputs.spec.db_spec is not None
    assert result.outputs.spec.memory_spec is not None
    assert result.outputs.spec.markdown is True
    assert result.outputs.spec.add_datetime_to_context is True


async def test_update_returns_outputs(harness: ProviderHarness) -> None:
    """on_update returns updated outputs."""
    model_dep = create_anthropic_model_dependency()
    previous = AgentConfig(model=model_dep)
    current = AgentConfig(
        model=model_dep,
        instructions=["New instructions."],
        markdown=True,
    )
    previous_outputs = AgentOutputs(
        spec=AgentSpec(
            name="test-agent",
            model_spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test"),
        ),
        pip_dependencies=[],
    )

    result = await harness.invoke_update(
        Agent,
        name="test-agent",
        config=current,
        previous_config=previous,
        current_outputs=previous_outputs,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.instructions == ["New instructions."]
    assert result.outputs.spec.markdown is True


async def test_delete_success(harness: ProviderHarness) -> None:
    """on_delete completes without error (stateless resource)."""
    model_dep = create_anthropic_model_dependency()
    config = AgentConfig(model=model_dep)

    result = await harness.invoke_delete(Agent, name="test-agent", config=config)

    assert result.success


def test_from_spec_returns_agno_agent(mocker: MockerFixture) -> None:
    """from_spec() returns configured Agno Agent instance."""
    mock_agent_init = mocker.patch(
        "agno.agent.Agent.__init__",
        return_value=None,
    )

    spec = AgentSpec(
        name="my-agent",
        description="Test agent",
        role="assistant",
        instructions=["Be helpful."],
        model_spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test"),
        markdown=True,
        add_datetime_to_context=True,
    )

    Agent.from_spec(spec)

    mock_agent_init.assert_called_once()
    call_kwargs = mock_agent_init.call_args.kwargs
    assert call_kwargs["name"] == "my-agent"
    assert call_kwargs["description"] == "Test agent"
    assert call_kwargs["role"] == "assistant"
    assert call_kwargs["instructions"] == ["Be helpful."]
    assert call_kwargs["markdown"] is True
    assert call_kwargs["add_datetime_to_context"] is True
    assert call_kwargs["model"] is not None


def test_from_spec_with_tools(mocker: MockerFixture) -> None:
    """from_spec() constructs tools from nested specs."""
    mock_agent_init = mocker.patch(
        "agno.agent.Agent.__init__",
        return_value=None,
    )
    mocker.patch("agno.tools.mcp.MCPTools.__init__", return_value=None)
    mocker.patch("agno.tools.websearch.WebSearchTools.__init__", return_value=None)

    spec = AgentSpec(
        name="my-agent",
        model_spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test"),
        tools_specs=[
            ToolsMCPSpec(url="http://localhost:8080", transport="sse"),
            ToolsWebSearchSpec(enable_search=True),
        ],
    )

    Agent.from_spec(spec)

    mock_agent_init.assert_called_once()
    call_kwargs = mock_agent_init.call_args.kwargs
    assert call_kwargs["tools"] is not None
    assert len(call_kwargs["tools"]) == 2


def test_from_spec_with_knowledge(mocker: MockerFixture) -> None:
    """from_spec() constructs knowledge from nested spec."""
    mock_agent_init = mocker.patch(
        "agno.agent.Agent.__init__",
        return_value=None,
    )
    mocker.patch("agno.vectordb.qdrant.Qdrant.__init__", return_value=None)
    mocker.patch("agno.knowledge.knowledge.Knowledge.__init__", return_value=None)

    spec = AgentSpec(
        name="my-agent",
        model_spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test"),
        knowledge_spec=KnowledgeSpec(
            name="kb",
            vector_db_spec=VectordbQdrantSpec(url="http://localhost:6333", collection="test"),
        ),
    )

    Agent.from_spec(spec)

    mock_agent_init.assert_called_once()
    call_kwargs = mock_agent_init.call_args.kwargs
    assert call_kwargs["knowledge"] is not None


def test_from_spec_with_memory(mocker: MockerFixture) -> None:
    """from_spec() constructs memory from nested spec."""
    mock_agent_init = mocker.patch(
        "agno.agent.Agent.__init__",
        return_value=None,
    )
    mocker.patch("agno.memory.manager.MemoryManager.__init__", return_value=None)

    spec = AgentSpec(
        name="my-agent",
        model_spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test"),
        memory_spec=MemoryManagerSpec(
            db_spec=DbPostgresSpec(connection_url="postgresql://localhost/db", db_schema="ai"),
        ),
    )

    Agent.from_spec(spec)

    mock_agent_init.assert_called_once()
    call_kwargs = mock_agent_init.call_args.kwargs
    assert call_kwargs["memory_manager"] is not None


def test_from_spec_with_db(mocker: MockerFixture) -> None:
    """from_spec() constructs db from nested spec."""
    mock_agent_init = mocker.patch(
        "agno.agent.Agent.__init__",
        return_value=None,
    )

    spec = AgentSpec(
        name="my-agent",
        model_spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test"),
        db_spec=DbPostgresSpec(connection_url="postgresql://localhost/db", db_schema="ai"),
    )

    Agent.from_spec(spec)

    mock_agent_init.assert_called_once()
    call_kwargs = mock_agent_init.call_args.kwargs
    assert call_kwargs["db"] is not None
    assert call_kwargs["read_chat_history"] is True


def test_from_spec_with_prompt(mocker: MockerFixture) -> None:
    """from_spec() uses prompt as instructions when provided."""
    mock_agent_init = mocker.patch(
        "agno.agent.Agent.__init__",
        return_value=None,
    )

    spec = AgentSpec(
        name="my-agent",
        model_spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test"),
        prompt_spec=PromptSpec(
            instructions=["From prompt."],
            rendered="From prompt.",
        ),
    )

    Agent.from_spec(spec)

    mock_agent_init.assert_called_once()
    call_kwargs = mock_agent_init.call_args.kwargs
    assert call_kwargs["instructions"] == "From prompt."


def test_from_spec_uses_inline_instructions_without_prompt(mocker: MockerFixture) -> None:
    """from_spec() uses inline instructions when no prompt provided."""
    mock_agent_init = mocker.patch(
        "agno.agent.Agent.__init__",
        return_value=None,
    )

    spec = AgentSpec(
        name="my-agent",
        model_spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test"),
        instructions=["Inline instruction."],
    )

    Agent.from_spec(spec)

    mock_agent_init.assert_called_once()
    call_kwargs = mock_agent_init.call_args.kwargs
    assert call_kwargs["instructions"] == ["Inline instruction."]
