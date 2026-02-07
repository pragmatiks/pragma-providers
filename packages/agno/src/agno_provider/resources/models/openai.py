"""OpenAI model resource wrapping Agno's OpenAIChat.

Provides a from_spec() method that returns an OpenAIChat instance for use
by dependent resources at runtime.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from agno.models.openai import OpenAIChat
from pragma_sdk import Field

from agno_provider.resources.base import AgnoSpec
from agno_provider.resources.models.base import Model, ModelConfig, ModelOutputs


__all__ = ["OpenAIModel", "OpenAIModelConfig", "OpenAIModelOutputs", "OpenAIModelSpec"]


class OpenAIModelSpec(AgnoSpec):
    """Specification for reconstructing OpenAI model at runtime.

    Contains all necessary information to create an OpenAIChat instance.
    Use OpenAIModel.from_spec() to construct the model instance.

    Attributes:
        type: Discriminator field, always "openai".
        id: OpenAI model identifier (e.g., "gpt-4o").
        api_key: OpenAI API key.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling parameter.
        frequency_penalty: Frequency penalty (-2.0 to 2.0).
        presence_penalty: Presence penalty (-2.0 to 2.0).
        seed: Random seed for deterministic outputs.
        stop: Stop sequences.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries.
        organization: OpenAI organization ID.
        base_url: Custom base URL for OpenAI-compatible APIs.
    """

    type: Literal["openai"] = "openai"
    id: str
    api_key: str
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


class OpenAIModelOutputs(ModelOutputs):
    """Outputs from OpenAI model resource.

    Attributes:
        spec: Specification for reconstructing the model at runtime.
    """

    spec: OpenAIModelSpec


class OpenAIModel(Model[OpenAIModelConfig, OpenAIModelOutputs, OpenAIModelSpec, OpenAIChat]):
    """OpenAI model resource wrapping Agno's OpenAIChat.

    This resource is stateless - it just wraps configuration. The actual
    OpenAIChat instance is created via from_spec() at runtime.

    Runtime reconstruction from spec:
        ```python
        spec = outputs.spec
        model = OpenAIModel.from_spec(spec)
        ```
    """

    resource: ClassVar[str] = "models/openai"

    @staticmethod
    def from_spec(spec: OpenAIModelSpec) -> OpenAIChat:
        """Factory: construct OpenAIChat from spec.

        Args:
            spec: The model specification.

        Returns:
            Configured OpenAIChat instance.
        """
        return OpenAIChat(**spec.model_dump(exclude={"type"}, exclude_none=True))

    def _build_spec(self) -> OpenAIModelSpec:
        """Build spec from current config.

        Creates a specification that can be serialized and used to
        reconstruct the model at runtime.

        Returns:
            OpenAIModelSpec with all configuration fields.
        """
        return OpenAIModelSpec(
            id=self.config.id,
            api_key=str(self.config.api_key),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            seed=self.config.seed,
            stop=self.config.stop,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            organization=self.config.organization,
            base_url=self.config.base_url,
        )

    def _build_outputs(self) -> OpenAIModelOutputs:
        """Build outputs from current config.

        Returns:
            OpenAIModelOutputs with spec.
        """
        return OpenAIModelOutputs(spec=self._build_spec())

    async def on_create(self) -> OpenAIModelOutputs:
        """Create returns serializable outputs with spec.

        Returns:
            OpenAIModelOutputs with spec.
        """
        return self._build_outputs()

    async def on_update(self, previous_config: OpenAIModelConfig) -> OpenAIModelOutputs:  # noqa: ARG002
        """Update returns serializable outputs with spec.

        Args:
            previous_config: The previous configuration (unused for stateless resource).

        Returns:
            OpenAIModelOutputs with spec.
        """
        return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
