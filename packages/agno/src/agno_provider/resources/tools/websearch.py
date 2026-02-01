"""Agno tools/websearch resource wrapping WebSearchTools."""

from __future__ import annotations

from typing import ClassVar, Literal

from agno.tools.websearch import WebSearchTools
from pragma_sdk import Config, Outputs, Resource


BackendType = Literal["auto", "duckduckgo", "google", "bing", "brave", "yandex", "yahoo"]


class ToolsWebSearchConfig(Config):
    """Configuration for web search toolkit.

    Attributes:
        enable_search: Enable web search functionality.
        enable_news: Enable news search functionality.
        backend: Search backend (auto, duckduckgo, google, bing, brave, yandex, yahoo).
        modifier: Text prepended to all queries (e.g., "site:example.com").
        fixed_max_results: Override default max results limit.
        proxy: Proxy URL for requests.
        timeout: Request timeout in seconds.
        verify_ssl: Whether to verify SSL certificates.
    """

    enable_search: bool = True
    enable_news: bool = True
    backend: BackendType = "auto"
    modifier: str | None = None
    fixed_max_results: int | None = None
    proxy: str | None = None
    timeout: int = 10
    verify_ssl: bool = True


class ToolsWebSearchOutputs(Outputs):
    """Outputs from tools/websearch resource.

    Attributes:
        tools: List of tool names provided by this toolkit.
        pip_dependencies: Python packages required by this toolkit.
    """

    tools: list[str]
    pip_dependencies: list[str]


class ToolsWebSearch(Resource[ToolsWebSearchConfig, ToolsWebSearchOutputs]):
    """Web search toolkit resource wrapping Agno's WebSearchTools.

    Stateless resource that wraps WebSearchTools configuration. Dependent
    resources (e.g., agno/agent) can call toolkit() to get the configured
    toolkit instance.

    Lifecycle:
        - on_create: Validate and return metadata
        - on_update: Re-validate with new config
        - on_delete: No-op (stateless)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "tools/websearch"

    def _enabled_tools(self) -> list[str]:
        """Get list of enabled tool names based on config.

        Returns:
            List of tool names that are enabled.
        """
        tools = []

        if self.config.enable_search:
            tools.append("web_search")

        if self.config.enable_news:
            tools.append("search_news")

        return tools

    def _build_outputs(self) -> ToolsWebSearchOutputs:
        """Build outputs from current config.

        Returns:
            ToolsWebSearchOutputs with tool names and dependencies.
        """
        return ToolsWebSearchOutputs(
            tools=self._enabled_tools(),
            pip_dependencies=["ddgs>=8.0.0"],
        )

    def toolkit(self) -> WebSearchTools:
        """Returns the configured WebSearchTools instance.

        Called by dependent resources (e.g., agno/agent) that need
        the actual toolkit object for their tools list.
        """
        return WebSearchTools(
            enable_search=self.config.enable_search,
            enable_news=self.config.enable_news,
            backend=self.config.backend,
            modifier=self.config.modifier,
            fixed_max_results=self.config.fixed_max_results,
            proxy=self.config.proxy,
            timeout=self.config.timeout,
            verify_ssl=self.config.verify_ssl,
        )

    async def on_create(self) -> ToolsWebSearchOutputs:
        """Create resource and return metadata.

        Returns:
            ToolsWebSearchOutputs with tool names and dependencies.
        """
        return self._build_outputs()

    async def on_update(self, previous_config: ToolsWebSearchConfig) -> ToolsWebSearchOutputs:  # noqa: ARG002
        """Update resource and return metadata.

        Args:
            previous_config: The previous configuration (unused for stateless resource).

        Returns:
            ToolsWebSearchOutputs with updated tool names and dependencies.
        """
        return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
