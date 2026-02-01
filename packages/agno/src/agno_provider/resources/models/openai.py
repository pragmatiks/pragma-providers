"""OpenAI model resource wrapping Agno's OpenAIChat.

Provides a model() method that returns an OpenAIChat instance for use
by dependent resources (e.g., agno/agent).
"""

from __future__ import annotations

from typing import ClassVar

from agno.models.openai import OpenAIChat
from pragma_sdk import Field

from agno_provider.resources.models.base import Model, ModelConfig


class OpenAIModelConfig(ModelConfig):
    """Configuration for OpenAI model.

    Maps directly to Agno's OpenAIChat constructor parameters.

    Attributes:
        api_key: OpenAI API key. Use a FieldReference to inject from pragma/secret.
        max_tokens: Optional maximum tokens in the response.
        temperature: Optional sampling temperature (0.0-2.0).
        top_p: Optional nucleus sampling parameter.
        frequency_penalty: Optional frequency penalty (-2.0 to 2.0).
        presence_penalty: Optional presence penalty (-2.0 to 2.0).
        seed: Optional random seed for deterministic outputs.
        stop: Optional stop sequences.
        timeout: Optional request timeout in seconds.
        max_retries: Optional maximum number of retries.
        organization: Optional OpenAI organization ID.
        base_url: Optional custom base URL for OpenAI-compatible APIs.
    """

    api_key: Field[str]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    timeout: float | None = None
    max_retries: int | None = None
    organization: str | None = None
    base_url: str | None = None


class OpenAIModel(Model[OpenAIModelConfig, OpenAIChat]):
    """OpenAI model resource wrapping Agno's OpenAIChat.

    This resource is stateless - it just wraps configuration. The actual
    OpenAIChat instance is created on-demand via the model() method.

    Example usage with Agno Agent:
        ```python
        from agno.agent import Agent

        model_resource = await model_dependency.resolve()
        agent = Agent(
            model=model_resource.model(),  # Call method to get instance
            instructions="You are a helpful assistant.",
        )
        ```
    """

    resource: ClassVar[str] = "models/openai"

    def model(self) -> OpenAIChat:
        """Returns the configured OpenAIChat instance.

        Called by dependent resources (e.g., agno/agent) that need
        the actual SDK object.
        """
        return OpenAIChat(**self.config.model_dump(exclude_none=True))
