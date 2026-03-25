"""Tests for Agno tools/mcp resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pragma_sdk.provider import ProviderHarness
from pydantic import ValidationError

from agno_provider import (
    ToolsMCP,
    ToolsMCPConfig,
    ToolsMCPOutputs,
)
from agno_provider.resources.tools.mcp import ToolsMCPSpec


if TYPE_CHECKING:
    from pytest_mock import MockerFixture, MockType


@pytest.mark.usefixtures("mock_mcp_tools")
async def test_create_with_command(harness: ProviderHarness) -> None:
    """on_create with stdio command returns spec with transport config."""
    config = ToolsMCPConfig(command="npx -y @modelcontextprotocol/server-github")

    result = await harness.invoke_create(ToolsMCP, name="github", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.transport == "stdio"
    assert result.outputs.spec.command == "npx"
    assert result.outputs.spec.args == ["-y", "@modelcontextprotocol/server-github"]


@pytest.mark.usefixtures("mock_mcp_tools")
async def test_create_with_url(harness: ProviderHarness) -> None:
    """on_create with SSE/HTTP URL returns spec with URL config."""
    config = ToolsMCPConfig(url="https://mcp.example.com/api", transport="sse")

    result = await harness.invoke_create(ToolsMCP, name="remote", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.transport == "sse"
    assert result.outputs.spec.url == "https://mcp.example.com/api"


@pytest.mark.usefixtures("mock_mcp_tools")
async def test_create_with_env(harness: ProviderHarness) -> None:
    """on_create with env variables includes them in spec."""
    config = ToolsMCPConfig(
        command="npx -y @modelcontextprotocol/server-github",
        env={"GITHUB_TOKEN": "test-token"},
    )

    result = await harness.invoke_create(ToolsMCP, name="github", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.env == {"GITHUB_TOKEN": "test-token"}


async def test_toolkit_returns_mcp_tools(harness: ProviderHarness, mock_mcp_tools: MockType) -> None:
    """toolkit() method returns configured MCPTools instance."""
    config = ToolsMCPConfig(command="npx -y @modelcontextprotocol/server-github")

    result = await harness.invoke_create(ToolsMCP, name="github", config=config)

    assert result.success
    assert result.resource is not None

    mock_mcp_tools.reset_mock()

    result.resource.toolkit()

    mock_mcp_tools.assert_called_once()
    call_kwargs = mock_mcp_tools.call_args.kwargs
    assert call_kwargs["command"] == "npx -y @modelcontextprotocol/server-github"


async def test_toolkit_with_all_options(harness: ProviderHarness, mock_mcp_tools: MockType) -> None:
    """toolkit() passes all configuration options."""
    config = ToolsMCPConfig(
        command="uvx mcp-server-git",
        timeout_seconds=30,
        include_tools=["tool_a", "tool_b"],
        exclude_tools=["tool_c"],
        tool_name_prefix="git",
    )

    result = await harness.invoke_create(ToolsMCP, name="git", config=config)

    assert result.success
    assert result.resource is not None

    mock_mcp_tools.reset_mock()

    result.resource.toolkit()

    call_kwargs = mock_mcp_tools.call_args.kwargs
    assert call_kwargs["command"] == "uvx mcp-server-git"
    assert call_kwargs["timeout_seconds"] == 30
    assert call_kwargs["include_tools"] == ["tool_a", "tool_b"]
    assert call_kwargs["exclude_tools"] == ["tool_c"]
    assert call_kwargs["tool_name_prefix"] == "git"


async def test_toolkit_with_env(harness: ProviderHarness, mock_mcp_tools: MockType) -> None:
    """toolkit() passes environment variables."""
    config = ToolsMCPConfig(
        command="npx -y @modelcontextprotocol/server-github",
        env={"GITHUB_TOKEN": "test-token-123"},
    )

    result = await harness.invoke_create(ToolsMCP, name="github", config=config)

    assert result.success
    assert result.resource is not None

    mock_mcp_tools.reset_mock()

    result.resource.toolkit()

    call_kwargs = mock_mcp_tools.call_args.kwargs
    assert call_kwargs["env"] == {"GITHUB_TOKEN": "test-token-123"}


@pytest.mark.usefixtures("mock_mcp_tools")
async def test_update_returns_new_spec(harness: ProviderHarness) -> None:
    """on_update returns spec with updated config."""
    previous = ToolsMCPConfig(command="npx old-server")
    current = ToolsMCPConfig(command="npx new-server")
    previous_outputs = ToolsMCPOutputs(
        spec=ToolsMCPSpec(command="npx", args=["old-server"], transport="stdio"),
    )

    result = await harness.invoke_update(
        ToolsMCP,
        name="server",
        config=current,
        previous_config=previous,
        current_outputs=previous_outputs,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.command == "npx"
    assert result.outputs.spec.args == ["new-server"]


@pytest.mark.usefixtures("mock_mcp_tools")
async def test_delete_success(harness: ProviderHarness) -> None:
    """on_delete completes without error (stateless resource)."""
    config = ToolsMCPConfig(command="npx -y @modelcontextprotocol/server-github")

    result = await harness.invoke_delete(ToolsMCP, name="github", config=config)

    assert result.success


def test_config_requires_command_or_url() -> None:
    """Config validation fails without command or url."""
    with pytest.raises(ValidationError) as exc_info:
        ToolsMCPConfig()

    errors = exc_info.value.errors()
    assert any("command" in str(e) or "url" in str(e) for e in errors)


def test_config_rejects_both_command_and_url() -> None:
    """Config validation fails with both command and url."""
    with pytest.raises(ValidationError) as exc_info:
        ToolsMCPConfig(command="npx server", url="https://example.com")

    errors = exc_info.value.errors()
    assert any("Cannot specify both" in str(e) for e in errors)


def test_config_stdio_requires_command() -> None:
    """Config validation fails for stdio transport without command."""
    with pytest.raises(ValidationError) as exc_info:
        ToolsMCPConfig(url="https://example.com", transport="stdio")

    errors = exc_info.value.errors()
    assert any("stdio" in str(e) and "command" in str(e) for e in errors)


def test_config_sse_requires_url() -> None:
    """Config validation fails for sse transport without url."""
    with pytest.raises(ValidationError) as exc_info:
        ToolsMCPConfig(command="npx server", transport="sse")

    errors = exc_info.value.errors()
    assert any("sse" in str(e) and "url" in str(e) for e in errors)


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert ToolsMCP.resource == "tools/mcp"


def test_outputs_serializable() -> None:
    """Outputs contain only serializable data."""
    outputs = ToolsMCPOutputs(
        spec=ToolsMCPSpec(command="npx", args=["-y", "@modelcontextprotocol/server-github"], transport="stdio"),
    )

    serialized = outputs.model_dump_json()

    assert "npx" in serialized
    assert "server-github" in serialized
    assert "stdio" in serialized


def test_config_defaults() -> None:
    """Config has expected defaults for optional fields."""
    config = ToolsMCPConfig(command="npx server")

    assert config.transport is None
    assert config.env is None
    assert config.timeout_seconds == 10
    assert config.include_tools is None
    assert config.exclude_tools is None
    assert config.tool_name_prefix is None


def test_static_headers_passed_to_mcp_tools(mock_mcp_tools: MockType) -> None:
    """Static headers are included in header_provider."""
    config = ToolsMCPConfig(
        url="https://mcp.example.com/api",
        headers={"Authorization": "Bearer secret-token"},
    )
    resource = ToolsMCP(name="test", config=config, outputs=None)

    resource.toolkit()

    call_kwargs = mock_mcp_tools.call_args.kwargs
    assert "header_provider" in call_kwargs
    header_fn = call_kwargs["header_provider"]
    headers = header_fn()
    assert headers["Authorization"] == "Bearer secret-token"


def test_run_context_headers_when_enabled(mock_mcp_tools: MockType, mocker: MockerFixture) -> None:
    """RunContext headers are included when include_run_context_headers is True."""
    config = ToolsMCPConfig(
        url="https://mcp.example.com/api",
        include_run_context_headers=True,
    )
    resource = ToolsMCP(name="test", config=config, outputs=None)

    resource.toolkit()

    call_kwargs = mock_mcp_tools.call_args.kwargs
    assert "header_provider" in call_kwargs
    header_fn = call_kwargs["header_provider"]

    mock_run_context = mocker.MagicMock()
    mock_run_context.user_id = "user-123"
    mock_run_context.session_id = "session-456"
    mock_run_context.run_id = "run-789"

    headers = header_fn(run_context=mock_run_context)
    assert headers["X-User-ID"] == "user-123"
    assert headers["X-Session-ID"] == "session-456"
    assert headers["X-Run-ID"] == "run-789"


def test_combined_static_and_run_context_headers(mock_mcp_tools: MockType, mocker: MockerFixture) -> None:
    """Static headers and RunContext headers can be combined."""
    config = ToolsMCPConfig(
        url="https://mcp.example.com/api",
        headers={"Authorization": "Bearer token"},
        include_run_context_headers=True,
    )
    resource = ToolsMCP(name="test", config=config, outputs=None)

    resource.toolkit()

    call_kwargs = mock_mcp_tools.call_args.kwargs
    header_fn = call_kwargs["header_provider"]

    mock_run_context = mocker.MagicMock()
    mock_run_context.user_id = "user-123"
    mock_run_context.session_id = None
    mock_run_context.run_id = "run-789"

    headers = header_fn(run_context=mock_run_context)
    assert headers["Authorization"] == "Bearer token"
    assert headers["X-User-ID"] == "user-123"
    assert headers["X-Run-ID"] == "run-789"
    assert "X-Session-ID" not in headers


def test_no_header_provider_when_not_configured(mock_mcp_tools: MockType) -> None:
    """No header_provider passed when headers not configured."""
    config = ToolsMCPConfig(command="npx server")
    resource = ToolsMCP(name="test", config=config, outputs=None)

    resource.toolkit()

    call_kwargs = mock_mcp_tools.call_args.kwargs
    assert "header_provider" not in call_kwargs


def test_agent_and_team_names_in_headers(mock_mcp_tools: MockType, mocker: MockerFixture) -> None:
    """Agent and team names included in headers when provided."""
    config = ToolsMCPConfig(
        url="https://mcp.example.com/api",
        include_run_context_headers=True,
    )
    resource = ToolsMCP(name="test", config=config, outputs=None)

    resource.toolkit()

    call_kwargs = mock_mcp_tools.call_args.kwargs
    header_fn = call_kwargs["header_provider"]

    mock_agent = mocker.MagicMock()
    mock_agent.name = "ResearchAgent"
    mock_team = mocker.MagicMock()
    mock_team.name = "ContentTeam"

    headers = header_fn(agent=mock_agent, team=mock_team)
    assert headers["X-Agent-Name"] == "ResearchAgent"
    assert headers["X-Team-Name"] == "ContentTeam"


def test_config_defaults_for_headers() -> None:
    """Config has expected defaults for header fields."""
    config = ToolsMCPConfig(command="npx server")

    assert config.headers is None
    assert config.include_run_context_headers is False
