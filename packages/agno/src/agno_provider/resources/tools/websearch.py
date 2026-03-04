"""Agno tools/websearch resource wrapping WebSearchTools."""

from __future__ import annotations

from typing import ClassVar, Literal

from agno.tools.websearch import WebSearchTools
from pragma_sdk import Config, Field, Outputs

from agno_provider.resources.base import AgnoResource, AgnoSpec


BackendType = Literal["auto", "duckduckgo", "google", "bing", "brave", "yandex", "yahoo"]


class ToolsWebSearchSpec(AgnoSpec):
    """Specification for reconstructing WebSearchTools at runtime.

    Contains all configuration needed to create a WebSearchTools instance.
    Used by dependent resources (e.g., agno/agent) to reconstruct the toolkit.

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
    enable_news: bool = False
    backend: BackendType = "auto"
    modifier: str | None = None
    fixed_max_results: int | None = None
    proxy: str | None = None
    timeout: int = 10
    verify_ssl: bool = True


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

    enable_search: Field[bool] = True
    enable_news: Field[bool] = True
    backend: Field[BackendType] = "auto"
    modifier: Field[str] | None = None
    fixed_max_results: Field[int] | None = None
    proxy: Field[str] | None = None
    timeout: Field[int] = 10
    verify_ssl: Field[bool] = True


class ToolsWebSearchOutputs(Outputs):
    """Outputs from tools/websearch resource.

    Attributes:
        pip_dependencies: Python packages required by this toolkit.
        spec: Specification for reconstructing the toolkit at runtime.
    """

    pip_dependencies: list[str]
    spec: ToolsWebSearchSpec


class ToolsWebSearch(AgnoResource[ToolsWebSearchConfig, ToolsWebSearchOutputs, ToolsWebSearchSpec]):
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

    @staticmethod
    def from_spec(spec: ToolsWebSearchSpec) -> WebSearchTools:
        """Factory: construct WebSearchTools from spec.

        Args:
            spec: The toolkit specification.

        Returns:
            Configured WebSearchTools instance.
        """
        return WebSearchTools(**spec.model_dump(exclude_none=True))

    def _build_spec(self) -> ToolsWebSearchSpec:
        """Build spec from current config.

        Creates a specification that can be serialized and used to
        reconstruct the toolkit at runtime.

        Returns:
            ToolsWebSearchSpec with all configuration fields.
        """
        return ToolsWebSearchSpec(
            enable_search=self.config.enable_search,
            enable_news=self.config.enable_news,
            backend=self.config.backend,
            modifier=self.config.modifier,
            fixed_max_results=self.config.fixed_max_results,
            proxy=self.config.proxy,
            timeout=self.config.timeout,
            verify_ssl=self.config.verify_ssl,
        )

    def _build_outputs(self) -> ToolsWebSearchOutputs:
        """Build outputs from current config.

        Returns:
            ToolsWebSearchOutputs with pip dependencies and spec.
        """
        return ToolsWebSearchOutputs(
            pip_dependencies=["ddgs>=8.0.0"],
            spec=self._build_spec(),
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
