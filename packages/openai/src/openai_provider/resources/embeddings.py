"""OpenAI Embeddings API resource."""

from __future__ import annotations

from typing import Any, ClassVar

from openai import AsyncOpenAI
from pydantic import BaseModel
from pragma_sdk import Config, Field, Outputs, Resource


class EmbeddingsConfig(Config):
    """Configuration for OpenAI Embeddings API.

    Attributes:
        api_key: OpenAI API key. Use a FieldReference to inject from pragma/secret.
        model: Model identifier (e.g., "text-embedding-3-small", "text-embedding-3-large").
        dimensions: Optional dimension override for models that support it.
    """

    api_key: Field[str]
    model: str = "text-embedding-3-small"
    dimensions: int | None = None


class EmbeddingsOutputs(Outputs):
    """Outputs from OpenAI Embeddings resource validation.

    Attributes:
        model: Model used for embeddings.
        dimensions: Embedding dimensions.
        ready: Whether the resource is ready to generate embeddings.
    """

    model: str
    dimensions: int
    ready: bool


class EmbedInput(BaseModel):
    """Input for the embed action.

    Attributes:
        text: Text or list of texts to generate embeddings for.
    """

    text: str | list[str]


class EmbedOutput(BaseModel):
    """Output from the embed action.

    Attributes:
        embeddings: List of embedding vectors.
        model: Model used to generate embeddings.
        usage: Token usage information.
    """

    embeddings: list[list[float]]
    model: str
    usage: dict[str, Any]


class Embeddings(Resource[EmbeddingsConfig, EmbeddingsOutputs]):
    """OpenAI Embeddings API resource.

    Wraps the OpenAI Embeddings API for generating text embeddings.
    API keys can be injected via FieldReference from pragma/secret resources.

    Lifecycle:
        - on_create: Validate API key and model, return model info
        - on_update: Re-validate if config changed
        - on_delete: No-op (stateless)

    Actions:
        - embed: Generate embeddings for text
    """

    provider: ClassVar[str] = "openai"
    resource: ClassVar[str] = "embeddings"

    def _get_client(self) -> AsyncOpenAI:
        """Get OpenAI async client with configured API key."""
        return AsyncOpenAI(api_key=self.config.api_key)

    async def _validate_model(self) -> EmbeddingsOutputs:
        """Validate the API key and model by making a test embedding request."""
        async with self._get_client() as client:
            # Make a minimal embedding request to validate credentials and model
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "input": "test",
            }

            if self.config.dimensions is not None:
                kwargs["dimensions"] = self.config.dimensions

            response = await client.embeddings.create(**kwargs)

            # Get the actual dimensions from the response
            dimensions = len(response.data[0].embedding)

            return EmbeddingsOutputs(
                model=response.model,
                dimensions=dimensions,
                ready=True,
            )

    async def on_create(self) -> EmbeddingsOutputs:
        """Create by validating API key and model."""
        return await self._validate_model()

    async def on_update(self, previous_config: EmbeddingsConfig) -> EmbeddingsOutputs:
        """Update by re-validating if config changed."""
        if previous_config == self.config and self.outputs is not None:
            return self.outputs

        return await self._validate_model()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""

    async def embed(self, input: EmbedInput) -> EmbedOutput:
        """Generate embeddings for text.

        Args:
            input: Text or list of texts to generate embeddings for.

        Returns:
            Embedding vectors with model and usage information.
        """
        async with self._get_client() as client:
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "input": input.text,
            }

            if self.config.dimensions is not None:
                kwargs["dimensions"] = self.config.dimensions

            response = await client.embeddings.create(**kwargs)

            return EmbedOutput(
                embeddings=[e.embedding for e in response.data],
                model=response.model,
                usage=response.usage.model_dump(),
            )
