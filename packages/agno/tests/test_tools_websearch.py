"""Tests for Agno tools/websearch resource."""

from __future__ import annotations

from agno.tools.websearch import WebSearchTools
from pragma_sdk.provider import ProviderHarness

from agno_provider import (
    ToolsWebSearch,
    ToolsWebSearchConfig,
    ToolsWebSearchOutputs,
)


async def test_create_default_config(harness: ProviderHarness) -> None:
    """on_create with default config returns both tools enabled."""
    config = ToolsWebSearchConfig()

    result = await harness.invoke_create(ToolsWebSearch, name="search", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.tools == ["web_search", "search_news"]
    assert result.outputs.pip_dependencies == ["ddgs>=8.0.0"]


async def test_create_search_only(harness: ProviderHarness) -> None:
    """on_create with news disabled returns only web_search."""
    config = ToolsWebSearchConfig(enable_news=False)

    result = await harness.invoke_create(ToolsWebSearch, name="search", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.tools == ["web_search"]


async def test_create_news_only(harness: ProviderHarness) -> None:
    """on_create with search disabled returns only search_news."""
    config = ToolsWebSearchConfig(enable_search=False)

    result = await harness.invoke_create(ToolsWebSearch, name="news", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.tools == ["search_news"]


async def test_create_neither_enabled(harness: ProviderHarness) -> None:
    """on_create with both disabled returns empty tools list."""
    config = ToolsWebSearchConfig(enable_search=False, enable_news=False)

    result = await harness.invoke_create(ToolsWebSearch, name="empty", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.tools == []


async def test_toolkit_returns_websearch_tools(harness: ProviderHarness) -> None:
    """toolkit() method returns configured WebSearchTools instance."""
    config = ToolsWebSearchConfig()

    result = await harness.invoke_create(ToolsWebSearch, name="search", config=config)

    assert result.success
    assert result.resource is not None

    toolkit = result.resource.toolkit()

    assert isinstance(toolkit, WebSearchTools)


async def test_toolkit_passes_config_options(harness: ProviderHarness) -> None:
    """toolkit() passes through configuration options."""
    config = ToolsWebSearchConfig(
        enable_search=True,
        enable_news=False,
        backend="duckduckgo",
        modifier="site:example.com",
        fixed_max_results=5,
        timeout=30,
        verify_ssl=False,
    )

    result = await harness.invoke_create(ToolsWebSearch, name="custom", config=config)

    assert result.success
    assert result.resource is not None

    toolkit = result.resource.toolkit()

    assert toolkit.backend == "duckduckgo"
    assert toolkit.modifier == "site:example.com"
    assert toolkit.fixed_max_results == 5
    assert toolkit.timeout == 30
    assert toolkit.verify_ssl is False


async def test_toolkit_with_proxy(harness: ProviderHarness) -> None:
    """toolkit() passes proxy configuration."""
    config = ToolsWebSearchConfig(proxy="http://proxy.example.com:8080")

    result = await harness.invoke_create(ToolsWebSearch, name="proxied", config=config)

    assert result.success
    assert result.resource is not None

    toolkit = result.resource.toolkit()

    assert toolkit.proxy == "http://proxy.example.com:8080"


async def test_toolkit_with_backend(harness: ProviderHarness) -> None:
    """toolkit() passes backend configuration."""
    config = ToolsWebSearchConfig(backend="google")

    result = await harness.invoke_create(ToolsWebSearch, name="google", config=config)

    assert result.success
    assert result.resource is not None

    toolkit = result.resource.toolkit()

    assert toolkit.backend == "google"


async def test_update_changes_outputs(harness: ProviderHarness) -> None:
    """on_update returns updated metadata."""
    previous = ToolsWebSearchConfig(enable_news=True)
    current = ToolsWebSearchConfig(enable_news=False)
    previous_outputs = ToolsWebSearchOutputs(
        tools=["web_search", "search_news"],
        pip_dependencies=["ddgs>=8.0.0"],
    )

    result = await harness.invoke_update(
        ToolsWebSearch,
        name="search",
        config=current,
        previous_config=previous,
        current_outputs=previous_outputs,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.tools == ["web_search"]


async def test_delete_success(harness: ProviderHarness) -> None:
    """on_delete completes without error (stateless resource)."""
    config = ToolsWebSearchConfig()

    result = await harness.invoke_delete(ToolsWebSearch, name="search", config=config)

    assert result.success


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert ToolsWebSearch.provider == "agno"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert ToolsWebSearch.resource == "tools/websearch"


def test_outputs_serializable() -> None:
    """Outputs contain only serializable data."""
    outputs = ToolsWebSearchOutputs(
        tools=["web_search", "search_news"],
        pip_dependencies=["ddgs>=8.0.0"],
    )

    serialized = outputs.model_dump_json()

    assert "web_search" in serialized
    assert "ddgs>=8.0.0" in serialized


def test_config_defaults() -> None:
    """Config has expected defaults."""
    config = ToolsWebSearchConfig()

    assert config.enable_search is True
    assert config.enable_news is True
    assert config.backend == "auto"
    assert config.modifier is None
    assert config.fixed_max_results is None
    assert config.proxy is None
    assert config.timeout == 10
    assert config.verify_ssl is True
