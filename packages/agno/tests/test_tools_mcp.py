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


if TYPE_CHECKING:
    from pytest_mock import MockerFixture, MockType


@pytest.mark.usefixtures("mock_mcp_tools")
async def test_create_with_command(harness: ProviderHarness) -> None:
    """on_create with stdio command discovers tools from MCP server."""
    config = ToolsMCPConfig(command="npx -y @modelcontextprotocol/server-github")

    result = await harness.invoke_create(ToolsMCP, name="github", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.ready is True
    assert result.outputs.tools == ["create_issue", "list_repos", "search_code"]
    assert "github" in result.outputs.name


@pytest.mark.usefixtures("mock_mcp_tools")
async def test_create_with_url(harness: ProviderHarness) -> None:
    """on_create with SSE/HTTP URL discovers tools from MCP server."""
    config = ToolsMCPConfig(url="https://mcp.example.com/api", transport="sse")

    result = await harness.invoke_create(ToolsMCP, name="remote", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.ready is True
    assert len(result.outputs.tools) == 3


async def test_create_connection_failure(harness: ProviderHarness, mock_mcp_tools: MockType) -> None:
    """on_create returns ready=False when MCP server connection fails."""
    mock_mcp_tools.return_value.connect.side_effect = ConnectionError("Server unreachable")
    config = ToolsMCPConfig(command="npx -y @modelcontextprotocol/server-github")

    result = await harness.invoke_create(ToolsMCP, name="github", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.ready is False
    assert result.outputs.tools == []


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


async def test_update_rediscovers_tools(
    harness: ProviderHarness, mock_mcp_tools: MockType, mocker: MockerFixture
) -> None:
    """on_update reconnects and refreshes tool list."""
    mock_mcp_tools.return_value.get_functions.return_value = {
        "new_tool": mocker.MagicMock(),
    }
    previous = ToolsMCPConfig(command="npx old-server")
    current = ToolsMCPConfig(command="npx new-server")
    previous_outputs = ToolsMCPOutputs(name="old", tools=["old_tool"], ready=True)

    result = await harness.invoke_update(
        ToolsMCP,
        name="server",
        config=current,
        previous_config=previous,
        current_outputs=previous_outputs,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.tools == ["new_tool"]


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


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert ToolsMCP.provider == "agno"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert ToolsMCP.resource == "tools/mcp"


def test_outputs_serializable() -> None:
    """Outputs contain only serializable data."""
    outputs = ToolsMCPOutputs(
        name="github",
        tools=["create_issue", "list_repos"],
        ready=True,
    )

    serialized = outputs.model_dump_json()

    assert "github" in serialized
    assert "create_issue" in serialized
    assert "true" in serialized.lower()


def test_config_defaults() -> None:
    """Config has expected defaults for optional fields."""
    config = ToolsMCPConfig(command="npx server")

    assert config.transport is None
    assert config.env is None
    assert config.timeout_seconds == 10
    assert config.include_tools is None
    assert config.exclude_tools is None
    assert config.tool_name_prefix is None


def test_server_name_from_npm_package() -> None:
    """Server name extracted from npm package in command."""
    config = ToolsMCPConfig(command="npx -y @modelcontextprotocol/server-github")
    resource = ToolsMCP(name="test", config=config, outputs=None)

    assert resource._server_name() == "server-github"


def test_server_name_from_uvx_command() -> None:
    """Server name extracted from uvx command."""
    config = ToolsMCPConfig(command="uvx mcp-server-git")
    resource = ToolsMCP(name="test", config=config, outputs=None)

    assert resource._server_name() == "mcp-server-git"


def test_server_name_from_url() -> None:
    """Server name extracted from URL hostname."""
    config = ToolsMCPConfig(url="https://mcp.example.com/api")
    resource = ToolsMCP(name="test", config=config, outputs=None)

    assert resource._server_name() == "mcp.example.com"


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
