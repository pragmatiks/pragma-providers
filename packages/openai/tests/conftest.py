"""Pytest configuration for openai provider tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pragma_sdk.provider import ProviderHarness

if TYPE_CHECKING:
    from pytest_mock import MockerFixture, MockType


@pytest.fixture
def harness() -> ProviderHarness:
    """Test harness for invoking lifecycle methods."""
    return ProviderHarness()


@pytest.fixture
def mock_openai_client(mocker: "MockerFixture") -> "MockType":
    """Mock AsyncOpenAI client with async context manager support."""
    mock_client = mocker.MagicMock()

    # Mock response
    mock_message = mocker.MagicMock()
    mock_message.content = "Hello! How can I help you today?"

    mock_choice = mocker.MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_usage = mocker.MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 15

    mock_response = mocker.MagicMock()
    mock_response.id = "chatcmpl-abc123"
    mock_response.choices = [mock_choice]
    mock_response.model = "gpt-4o"
    mock_response.usage = mock_usage

    mock_completions = mocker.MagicMock()
    mock_completions.create = mocker.AsyncMock(return_value=mock_response)

    mock_chat = mocker.MagicMock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    # Support async context manager protocol
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch(
        "openai_provider.resources.chat_completions.AsyncOpenAI",
        return_value=mock_client,
    )

    return mock_client


@pytest.fixture
def mock_embeddings_client(mocker: "MockerFixture") -> "MockType":
    """Mock AsyncOpenAI client for embeddings with async context manager support."""
    mock_client = mocker.MagicMock()

    # Mock embedding response (single embedding)
    mock_embedding = mocker.MagicMock()
    mock_embedding.embedding = [0.1] * 1536  # 1536-dimensional embedding

    mock_usage = mocker.MagicMock()
    mock_usage.prompt_tokens = 2
    mock_usage.total_tokens = 2
    mock_usage.model_dump.return_value = {"prompt_tokens": 2, "total_tokens": 2}

    mock_response = mocker.MagicMock()
    mock_response.data = [mock_embedding]
    mock_response.model = "text-embedding-3-small"
    mock_response.usage = mock_usage

    mock_embeddings = mocker.MagicMock()
    mock_embeddings.create = mocker.AsyncMock(return_value=mock_response)
    mock_client.embeddings = mock_embeddings

    # Support async context manager protocol
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch(
        "openai_provider.resources.embeddings.AsyncOpenAI",
        return_value=mock_client,
    )

    return mock_client


@pytest.fixture
def mock_embeddings_client_batch(mocker: "MockerFixture") -> "MockType":
    """Mock AsyncOpenAI client for batch embeddings with async context manager support."""
    mock_client = mocker.MagicMock()

    # Mock embedding response (multiple embeddings)
    mock_embedding_1 = mocker.MagicMock()
    mock_embedding_1.embedding = [0.1] * 1536

    mock_embedding_2 = mocker.MagicMock()
    mock_embedding_2.embedding = [0.2] * 1536

    mock_usage = mocker.MagicMock()
    mock_usage.prompt_tokens = 4
    mock_usage.total_tokens = 4
    mock_usage.model_dump.return_value = {"prompt_tokens": 4, "total_tokens": 4}

    mock_response = mocker.MagicMock()
    mock_response.data = [mock_embedding_1, mock_embedding_2]
    mock_response.model = "text-embedding-3-small"
    mock_response.usage = mock_usage

    mock_embeddings = mocker.MagicMock()
    mock_embeddings.create = mocker.AsyncMock(return_value=mock_response)
    mock_client.embeddings = mock_embeddings

    # Support async context manager protocol
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch(
        "openai_provider.resources.embeddings.AsyncOpenAI",
        return_value=mock_client,
    )

    return mock_client
