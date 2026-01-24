"""Tests for OpenAI Embeddings resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import APIError
from pragma_sdk.provider import ProviderHarness

from openai_provider import (
    EmbedInput,
    Embeddings,
    EmbeddingsConfig,
    EmbeddingsOutputs,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture, MockType


async def test_create_embeddings_success(
    harness: ProviderHarness,
    mock_embeddings_client: "MockType",
) -> None:
    """on_create validates API key and returns model info."""
    config = EmbeddingsConfig(
        api_key="sk-test-key",
        model="text-embedding-3-small",
    )

    result = await harness.invoke_create(Embeddings, name="test-embeddings", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.model == "text-embedding-3-small"
    assert result.outputs.dimensions == 1536
    assert result.outputs.ready is True

    mock_embeddings_client.embeddings.create.assert_called_once()


async def test_create_embeddings_with_dimensions(
    harness: ProviderHarness,
    mock_embeddings_client: "MockType",
) -> None:
    """on_create passes dimensions to API when specified."""
    config = EmbeddingsConfig(
        api_key="sk-test-key",
        model="text-embedding-3-small",
        dimensions=512,
    )

    result = await harness.invoke_create(Embeddings, name="test-embeddings", config=config)

    assert result.success
    call_kwargs = mock_embeddings_client.embeddings.create.call_args.kwargs
    assert call_kwargs["dimensions"] == 512


async def test_create_embeddings_without_dimensions(
    harness: ProviderHarness,
    mock_embeddings_client: "MockType",
) -> None:
    """on_create omits dimensions when not specified."""
    config = EmbeddingsConfig(
        api_key="sk-test-key",
        model="text-embedding-3-small",
    )

    result = await harness.invoke_create(Embeddings, name="test-embeddings", config=config)

    assert result.success
    call_kwargs = mock_embeddings_client.embeddings.create.call_args.kwargs
    assert "dimensions" not in call_kwargs


async def test_update_revalidates_on_model_change(
    harness: ProviderHarness,
    mock_embeddings_client: "MockType",
) -> None:
    """on_update revalidates when model changes."""
    previous = EmbeddingsConfig(
        api_key="sk-test-key",
        model="text-embedding-3-small",
    )
    current = EmbeddingsConfig(
        api_key="sk-test-key",
        model="text-embedding-3-large",
    )

    result = await harness.invoke_update(
        Embeddings,
        name="test-embeddings",
        config=current,
        previous_config=previous,
        current_outputs=EmbeddingsOutputs(
            model="text-embedding-3-small",
            dimensions=1536,
            ready=True,
        ),
    )

    assert result.success
    mock_embeddings_client.embeddings.create.assert_called_once()


async def test_update_returns_existing_when_unchanged(
    harness: ProviderHarness,
    mock_embeddings_client: "MockType",
) -> None:
    """on_update returns existing outputs when config unchanged."""
    config = EmbeddingsConfig(
        api_key="sk-test-key",
        model="text-embedding-3-small",
    )
    existing_outputs = EmbeddingsOutputs(
        model="text-embedding-3-small",
        dimensions=1536,
        ready=True,
    )

    result = await harness.invoke_update(
        Embeddings,
        name="test-embeddings",
        config=config,
        previous_config=config,
        current_outputs=existing_outputs,
    )

    assert result.success
    assert result.outputs == existing_outputs
    mock_embeddings_client.embeddings.create.assert_not_called()


async def test_delete_success(
    harness: ProviderHarness,
    mock_embeddings_client: "MockType",
) -> None:
    """on_delete completes without error (stateless resource)."""
    config = EmbeddingsConfig(
        api_key="sk-test-key",
        model="text-embedding-3-small",
    )

    result = await harness.invoke_delete(Embeddings, name="test-embeddings", config=config)

    assert result.success
    mock_embeddings_client.embeddings.create.assert_not_called()


async def test_api_error_propagates(
    harness: ProviderHarness,
    mocker: "MockerFixture",
) -> None:
    """API errors propagate correctly."""
    mock_client = mocker.MagicMock()
    mock_embeddings = mocker.MagicMock()
    mock_embeddings.create = mocker.AsyncMock(
        side_effect=APIError(
            message="Invalid API key",
            request=mocker.MagicMock(),
            body={"error": {"message": "Invalid API key"}},
        )
    )
    mock_client.embeddings = mock_embeddings

    # Support async context manager protocol
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch(
        "openai_provider.resources.embeddings.AsyncOpenAI",
        return_value=mock_client,
    )

    config = EmbeddingsConfig(
        api_key="invalid-key",
        model="text-embedding-3-small",
    )

    result = await harness.invoke_create(Embeddings, name="test-embeddings", config=config)

    assert result.failed
    assert result.error is not None
    assert "Invalid API key" in str(result.error)


async def test_embed_action_single_text(
    mock_embeddings_client: "MockType",
) -> None:
    """embed action generates embeddings for single text."""
    config = EmbeddingsConfig(
        api_key="sk-test-key",
        model="text-embedding-3-small",
    )
    resource = Embeddings(
        name="test-embeddings",
        config=config,
        outputs=EmbeddingsOutputs(
            model="text-embedding-3-small",
            dimensions=1536,
            ready=True,
        ),
    )

    result = await resource.embed(EmbedInput(text="Hello world"))

    assert result.model == "text-embedding-3-small"
    assert len(result.embeddings) == 1
    assert len(result.embeddings[0]) == 1536
    assert "prompt_tokens" in result.usage
    assert "total_tokens" in result.usage


async def test_embed_action_multiple_texts(
    mock_embeddings_client_batch: "MockType",
) -> None:
    """embed action generates embeddings for multiple texts."""
    config = EmbeddingsConfig(
        api_key="sk-test-key",
        model="text-embedding-3-small",
    )
    resource = Embeddings(
        name="test-embeddings",
        config=config,
        outputs=EmbeddingsOutputs(
            model="text-embedding-3-small",
            dimensions=1536,
            ready=True,
        ),
    )

    result = await resource.embed(EmbedInput(text=["Hello", "World"]))

    assert result.model == "text-embedding-3-small"
    assert len(result.embeddings) == 2
    assert len(result.embeddings[0]) == 1536
    assert len(result.embeddings[1]) == 1536


async def test_embed_action_with_dimensions(
    mocker: "MockerFixture",
) -> None:
    """embed action passes dimensions to API when configured."""
    mock_client = mocker.MagicMock()
    mock_embedding = mocker.MagicMock()
    mock_embedding.embedding = [0.1] * 512  # Reduced dimensions

    mock_usage = mocker.MagicMock()
    mock_usage.model_dump.return_value = {"prompt_tokens": 2, "total_tokens": 2}

    mock_response = mocker.MagicMock()
    mock_response.data = [mock_embedding]
    mock_response.model = "text-embedding-3-small"
    mock_response.usage = mock_usage

    mock_embeddings = mocker.MagicMock()
    mock_embeddings.create = mocker.AsyncMock(return_value=mock_response)
    mock_client.embeddings = mock_embeddings

    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch(
        "openai_provider.resources.embeddings.AsyncOpenAI",
        return_value=mock_client,
    )

    config = EmbeddingsConfig(
        api_key="sk-test-key",
        model="text-embedding-3-small",
        dimensions=512,
    )
    resource = Embeddings(
        name="test-embeddings",
        config=config,
        outputs=EmbeddingsOutputs(
            model="text-embedding-3-small",
            dimensions=512,
            ready=True,
        ),
    )

    result = await resource.embed(EmbedInput(text="test"))

    assert len(result.embeddings[0]) == 512
    call_kwargs = mock_embeddings.create.call_args.kwargs
    assert call_kwargs["dimensions"] == 512


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert Embeddings.provider == "openai"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Embeddings.resource == "embeddings"
