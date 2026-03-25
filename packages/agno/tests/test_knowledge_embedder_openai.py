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
from agno_provider.resources.knowledge.embedder.openai import EmbedderOpenAISpec


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
    """Outputs contain pip_dependencies and spec."""
    outputs = EmbedderOpenAIOutputs(
        pip_dependencies=[],
        spec=EmbedderOpenAISpec(
            id="text-embedding-3-small",
            api_key="sk-test-key",
        ),
    )

    assert outputs.pip_dependencies == []
    assert outputs.spec.id == "text-embedding-3-small"
    assert outputs.spec.api_key == "sk-test-key"


def test_outputs_schema_are_serializable() -> None:
    """Outputs can be serialized to JSON."""
    outputs = EmbedderOpenAIOutputs(
        pip_dependencies=[],
        spec=EmbedderOpenAISpec(
            id="text-embedding-3-large",
            api_key="sk-test-key",
            dimensions=3072,
        ),
    )

    serialized = outputs.model_dump_json()

    assert "pip_dependencies" in serialized
    assert "spec" in serialized
    assert "text-embedding-3-large" in serialized


def test_from_spec_returns_openai_embedder_instance() -> None:
    """from_spec() returns an OpenAIEmbedder instance."""
    spec = EmbedderOpenAISpec(
        id="text-embedding-3-small",
        api_key="sk-test-key",
    )
    result = EmbedderOpenAI.from_spec(spec)
    assert isinstance(result, OpenAIEmbedder)


def test_from_spec_passes_model_id() -> None:
    """from_spec() passes model ID correctly."""
    spec = EmbedderOpenAISpec(
        id="text-embedding-3-large",
        api_key="sk-test-key",
    )
    result = EmbedderOpenAI.from_spec(spec)
    assert result.id == "text-embedding-3-large"


def test_from_spec_passes_api_key() -> None:
    """from_spec() passes API key correctly."""
    spec = EmbedderOpenAISpec(
        id="text-embedding-3-small",
        api_key="sk-secret-key",
    )
    result = EmbedderOpenAI.from_spec(spec)
    assert result.api_key == "sk-secret-key"


def test_from_spec_passes_encoding_format() -> None:
    """from_spec() passes encoding format correctly."""
    spec = EmbedderOpenAISpec(
        id="text-embedding-3-small",
        api_key="sk-test-key",
        encoding_format="base64",
    )
    result = EmbedderOpenAI.from_spec(spec)
    assert result.encoding_format == "base64"


def test_from_spec_passes_custom_dimensions() -> None:
    """from_spec() passes custom dimensions when specified."""
    spec = EmbedderOpenAISpec(
        id="text-embedding-3-small",
        api_key="sk-test-key",
        dimensions=256,
    )
    result = EmbedderOpenAI.from_spec(spec)
    assert result.dimensions == 256


def test_from_spec_passes_organization() -> None:
    """from_spec() passes organization when specified."""
    spec = EmbedderOpenAISpec(
        id="text-embedding-3-small",
        api_key="sk-test-key",
        organization="org-test-123",
    )
    result = EmbedderOpenAI.from_spec(spec)
    assert result.organization == "org-test-123"


def test_from_spec_passes_base_url() -> None:
    """from_spec() passes base_url when specified."""
    spec = EmbedderOpenAISpec(
        id="text-embedding-3-small",
        api_key="sk-test-key",
        base_url="https://custom-openai.example.com/v1",
    )
    result = EmbedderOpenAI.from_spec(spec)
    assert str(result.base_url) == "https://custom-openai.example.com/v1"


def test_from_spec_with_all_parameters() -> None:
    """from_spec() passes all parameters correctly."""
    spec = EmbedderOpenAISpec(
        id="text-embedding-3-small",
        api_key="sk-full-test",
        dimensions=512,
        encoding_format="float",
        organization="org-full",
        base_url="https://api.example.com",
    )

    result = EmbedderOpenAI.from_spec(spec)

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
    assert result.spec.id == "text-embedding-3-small"
    assert result.pip_dependencies == []


async def test_lifecycle_on_create_with_custom_dimensions() -> None:
    """on_create respects custom dimensions."""
    resource = create_embedder_openai(
        model_id="text-embedding-3-large",
        dimensions=256,
    )

    result = await resource.on_create()

    assert result.spec.dimensions == 256


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
    assert result.spec.id == "text-embedding-3-large"


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
    assert result.outputs.spec.id == "text-embedding-3-small"
    assert result.outputs.pip_dependencies == []


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
        pip_dependencies=[],
        spec=EmbedderOpenAISpec(
            id="text-embedding-3-small",
            api_key="sk-old-key",
        ),
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
    assert result.outputs.spec.id == "text-embedding-3-large"


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


def test_mocked_from_spec_instantiation(mocker: MockerFixture) -> None:
    """from_spec() instantiation can be mocked."""
    mock_embedder_class = mocker.patch("agno_provider.resources.knowledge.embedder.openai.OpenAIEmbedder")
    mock_instance = mocker.MagicMock()
    mock_embedder_class.return_value = mock_instance

    spec = EmbedderOpenAISpec(
        id="text-embedding-3-small",
        api_key="sk-mock-key",
        dimensions=512,
    )

    result = EmbedderOpenAI.from_spec(spec)

    mock_embedder_class.assert_called_once_with(
        id="text-embedding-3-small",
        api_key="sk-mock-key",
        encoding_format="float",
        dimensions=512,
    )
    assert result is mock_instance


def test_mocked_from_spec_with_organization(mocker: MockerFixture) -> None:
    """from_spec() with organization passes it to constructor."""
    mock_embedder_class = mocker.patch("agno_provider.resources.knowledge.embedder.openai.OpenAIEmbedder")

    spec = EmbedderOpenAISpec(
        id="text-embedding-3-small",
        api_key="sk-mock-key",
        organization="org-mocked",
    )

    EmbedderOpenAI.from_spec(spec)

    call_kwargs = mock_embedder_class.call_args.kwargs
    assert call_kwargs["organization"] == "org-mocked"


def test_mocked_from_spec_with_base_url(mocker: MockerFixture) -> None:
    """from_spec() with base_url passes it to constructor."""
    mock_embedder_class = mocker.patch("agno_provider.resources.knowledge.embedder.openai.OpenAIEmbedder")

    spec = EmbedderOpenAISpec(
        id="text-embedding-3-small",
        api_key="sk-mock-key",
        base_url="https://mock-api.example.com",
    )

    EmbedderOpenAI.from_spec(spec)

    call_kwargs = mock_embedder_class.call_args.kwargs
    assert call_kwargs["base_url"] == "https://mock-api.example.com"
