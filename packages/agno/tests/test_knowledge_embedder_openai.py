"""Tests for Agno knowledge/embedder/openai resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from agno.knowledge.embedder.openai import OpenAIEmbedder
from pragma_sdk import LifecycleState
from pragma_sdk.provider import ProviderHarness
from pydantic import ValidationError

from agno_provider import (
    EmbedderOpenAI,
    EmbedderOpenAIConfig,
    EmbedderOpenAIOutputs,
)


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_embedder_openai(
    name: str = "openai-embedder",
    model_id: str = "text-embedding-3-small",
    api_key: str = "sk-test-key",
    dimensions: int | None = None,
    encoding_format: str = "float",
    organization: str | None = None,
    base_url: str | None = None,
    outputs: EmbedderOpenAIOutputs | None = None,
) -> EmbedderOpenAI:
    """Create an EmbedderOpenAI resource for testing."""
    config = EmbedderOpenAIConfig(
        id=model_id,
        api_key=api_key,
        dimensions=dimensions,
        encoding_format=encoding_format,
        organization=organization,
        base_url=base_url,
    )

    return EmbedderOpenAI(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


def test_resource_metadata_provider_name() -> None:
    """Resource has correct provider name."""
    assert EmbedderOpenAI.provider == "agno"


def test_resource_metadata_resource_type() -> None:
    """Resource has correct resource type."""
    assert EmbedderOpenAI.resource == "knowledge/embedder/openai"


def test_config_validation_default_model_id() -> None:
    """Default model ID is text-embedding-3-small."""
    config = EmbedderOpenAIConfig(api_key="sk-test")
    assert config.id == "text-embedding-3-small"


def test_config_validation_default_encoding_format() -> None:
    """Default encoding format is float."""
    config = EmbedderOpenAIConfig(api_key="sk-test")
    assert config.encoding_format == "float"


def test_config_validation_api_key_is_required() -> None:
    """Config requires api_key field."""
    with pytest.raises(ValidationError, match="api_key"):
        EmbedderOpenAIConfig()


def test_config_validation_optional_dimensions() -> None:
    """Config accepts optional dimensions."""
    config = EmbedderOpenAIConfig(api_key="sk-test", dimensions=512)
    assert config.dimensions == 512


def test_config_validation_optional_organization() -> None:
    """Config accepts optional organization."""
    config = EmbedderOpenAIConfig(api_key="sk-test", organization="org-123")
    assert config.organization == "org-123"


def test_config_validation_optional_base_url() -> None:
    """Config accepts optional base_url."""
    config = EmbedderOpenAIConfig(
        api_key="sk-test",
        base_url="https://custom-api.example.com/v1",
    )
    assert config.base_url == "https://custom-api.example.com/v1"


def test_config_validation_encoding_format_base64() -> None:
    """Config accepts base64 encoding format."""
    config = EmbedderOpenAIConfig(api_key="sk-test", encoding_format="base64")
    assert config.encoding_format == "base64"


def test_outputs_schema_contain_expected_fields() -> None:
    """Outputs contain model, dimensions, pip_dependencies, ready."""
    outputs = EmbedderOpenAIOutputs(
        model="text-embedding-3-small",
        dimensions=1536,
        pip_dependencies=[],
        ready=True,
    )

    assert outputs.model == "text-embedding-3-small"
    assert outputs.dimensions == 1536
    assert outputs.pip_dependencies == []
    assert outputs.ready is True


def test_outputs_schema_are_serializable() -> None:
    """Outputs can be serialized to JSON."""
    outputs = EmbedderOpenAIOutputs(
        model="text-embedding-3-large",
        dimensions=3072,
        pip_dependencies=[],
        ready=True,
    )

    serialized = outputs.model_dump_json()

    assert "model" in serialized
    assert "dimensions" in serialized
    assert "pip_dependencies" in serialized
    assert "ready" in serialized
    assert "text-embedding-3-large" in serialized


def test_dimensions_mapping_text_embedding_3_small() -> None:
    """text-embedding-3-small returns 1536 dimensions."""
    resource = create_embedder_openai(model_id="text-embedding-3-small")
    assert resource._get_dimensions() == 1536


def test_dimensions_mapping_text_embedding_3_large() -> None:
    """text-embedding-3-large returns 3072 dimensions."""
    resource = create_embedder_openai(model_id="text-embedding-3-large")
    assert resource._get_dimensions() == 3072


def test_dimensions_mapping_text_embedding_ada_002() -> None:
    """text-embedding-ada-002 returns 1536 dimensions."""
    resource = create_embedder_openai(model_id="text-embedding-ada-002")
    assert resource._get_dimensions() == 1536


def test_dimensions_mapping_unknown_model_defaults_to_1536() -> None:
    """Unknown model defaults to 1536 dimensions."""
    resource = create_embedder_openai(model_id="unknown-model")
    assert resource._get_dimensions() == 1536


def test_dimensions_mapping_custom_dimensions_override() -> None:
    """Custom dimensions override model defaults."""
    resource = create_embedder_openai(
        model_id="text-embedding-3-large",
        dimensions=512,
    )
    assert resource._get_dimensions() == 512


def test_embedder_method_returns_openai_embedder_instance() -> None:
    """embedder() returns an OpenAIEmbedder instance."""
    resource = create_embedder_openai()
    result = resource.embedder()
    assert isinstance(result, OpenAIEmbedder)


def test_embedder_method_passes_model_id() -> None:
    """embedder() passes model ID correctly."""
    resource = create_embedder_openai(model_id="text-embedding-3-large")
    result = resource.embedder()
    assert result.id == "text-embedding-3-large"


def test_embedder_method_passes_api_key() -> None:
    """embedder() passes API key correctly."""
    resource = create_embedder_openai(api_key="sk-secret-key")
    result = resource.embedder()
    assert result.api_key == "sk-secret-key"


def test_embedder_method_passes_encoding_format() -> None:
    """embedder() passes encoding format correctly."""
    resource = create_embedder_openai(encoding_format="base64")
    result = resource.embedder()
    assert result.encoding_format == "base64"


def test_embedder_method_passes_custom_dimensions() -> None:
    """embedder() passes custom dimensions when specified."""
    resource = create_embedder_openai(dimensions=256)
    result = resource.embedder()
    assert result.dimensions == 256


def test_embedder_method_passes_organization() -> None:
    """embedder() passes organization when specified."""
    resource = create_embedder_openai(organization="org-test-123")
    result = resource.embedder()
    assert result.organization == "org-test-123"


def test_embedder_method_passes_base_url() -> None:
    """embedder() passes base_url when specified."""
    resource = create_embedder_openai(
        base_url="https://custom-openai.example.com/v1",
    )
    result = resource.embedder()
    assert str(result.base_url) == "https://custom-openai.example.com/v1"


def test_embedder_method_with_all_parameters() -> None:
    """embedder() passes all parameters correctly."""
    resource = create_embedder_openai(
        model_id="text-embedding-3-small",
        api_key="sk-full-test",
        dimensions=512,
        encoding_format="float",
        organization="org-full",
        base_url="https://api.example.com",
    )

    result = resource.embedder()

    assert result.id == "text-embedding-3-small"
    assert result.api_key == "sk-full-test"
    assert result.dimensions == 512
    assert result.encoding_format == "float"
    assert result.organization == "org-full"


async def test_lifecycle_on_create_returns_outputs() -> None:
    """on_create returns serializable outputs."""
    resource = create_embedder_openai(
        model_id="text-embedding-3-small",
        api_key="sk-test-key",
    )

    result = await resource.on_create()

    assert isinstance(result, EmbedderOpenAIOutputs)
    assert result.model == "text-embedding-3-small"
    assert result.dimensions == 1536
    assert result.pip_dependencies == []
    assert result.ready is True


async def test_lifecycle_on_create_with_custom_dimensions() -> None:
    """on_create respects custom dimensions."""
    resource = create_embedder_openai(
        model_id="text-embedding-3-large",
        dimensions=256,
    )

    result = await resource.on_create()

    assert result.dimensions == 256


async def test_lifecycle_on_update_returns_outputs() -> None:
    """on_update returns updated outputs."""
    resource = create_embedder_openai(
        model_id="text-embedding-3-large",
    )

    previous_config = EmbedderOpenAIConfig(
        id="text-embedding-3-small",
        api_key="sk-test-key",
    )

    result = await resource.on_update(previous_config)

    assert isinstance(result, EmbedderOpenAIOutputs)
    assert result.model == "text-embedding-3-large"
    assert result.dimensions == 3072


async def test_lifecycle_on_delete_is_noop() -> None:
    """on_delete completes without error (stateless resource)."""
    resource = create_embedder_openai()
    await resource.on_delete()


async def test_harness_create_returns_outputs(harness: ProviderHarness) -> None:
    """on_create via harness returns correct outputs."""
    config = EmbedderOpenAIConfig(
        id="text-embedding-3-small",
        api_key="sk-test-key",
    )

    result = await harness.invoke_create(
        EmbedderOpenAI,
        name="test-embedder",
        config=config,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.model == "text-embedding-3-small"
    assert result.outputs.dimensions == 1536
    assert result.outputs.pip_dependencies == []
    assert result.outputs.ready is True


async def test_harness_update_returns_outputs(harness: ProviderHarness) -> None:
    """on_update via harness returns updated outputs."""
    previous = EmbedderOpenAIConfig(
        id="text-embedding-3-small",
        api_key="sk-old-key",
    )
    current = EmbedderOpenAIConfig(
        id="text-embedding-3-large",
        api_key="sk-new-key",
    )
    current_outputs = EmbedderOpenAIOutputs(
        model="text-embedding-3-small",
        dimensions=1536,
        pip_dependencies=[],
        ready=True,
    )

    result = await harness.invoke_update(
        EmbedderOpenAI,
        name="test-embedder",
        config=current,
        previous_config=previous,
        current_outputs=current_outputs,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.model == "text-embedding-3-large"
    assert result.outputs.dimensions == 3072


async def test_harness_delete_success(harness: ProviderHarness) -> None:
    """on_delete via harness completes without error."""
    config = EmbedderOpenAIConfig(
        id="text-embedding-3-small",
        api_key="sk-test-key",
    )

    result = await harness.invoke_delete(
        EmbedderOpenAI,
        name="test-embedder",
        config=config,
    )

    assert result.success


def test_mocked_embedder_instantiation(mocker: MockerFixture) -> None:
    """embedder() instantiation can be mocked."""
    mock_embedder_class = mocker.patch("agno_provider.resources.knowledge.embedder.openai.OpenAIEmbedder")
    mock_instance = mocker.MagicMock()
    mock_embedder_class.return_value = mock_instance

    resource = create_embedder_openai(
        model_id="text-embedding-3-small",
        api_key="sk-mock-key",
        dimensions=512,
    )

    result = resource.embedder()

    mock_embedder_class.assert_called_once_with(
        id="text-embedding-3-small",
        api_key="sk-mock-key",
        encoding_format="float",
        dimensions=512,
    )
    assert result is mock_instance


def test_mocked_embedder_with_organization(mocker: MockerFixture) -> None:
    """embedder() with organization passes it to constructor."""
    mock_embedder_class = mocker.patch("agno_provider.resources.knowledge.embedder.openai.OpenAIEmbedder")

    resource = create_embedder_openai(
        api_key="sk-mock-key",
        organization="org-mocked",
    )

    resource.embedder()

    call_kwargs = mock_embedder_class.call_args.kwargs
    assert call_kwargs["organization"] == "org-mocked"


def test_mocked_embedder_with_base_url(mocker: MockerFixture) -> None:
    """embedder() with base_url passes it to constructor."""
    mock_embedder_class = mocker.patch("agno_provider.resources.knowledge.embedder.openai.OpenAIEmbedder")

    resource = create_embedder_openai(
        api_key="sk-mock-key",
        base_url="https://mock-api.example.com",
    )

    resource.embedder()

    call_kwargs = mock_embedder_class.call_args.kwargs
    assert call_kwargs["base_url"] == "https://mock-api.example.com"
