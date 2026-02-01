"""Tests for Agno OpenAI Model resource."""

from __future__ import annotations

from agno.models.openai import OpenAIChat
from pragma_sdk import LifecycleState

from agno_provider import (
    ModelOutputs,
    OpenAIModel,
    OpenAIModelConfig,
)


def create_openai_model(
    name: str = "gpt4",
    model_id: str = "gpt-4o",
    api_key: str = "sk-test-key",
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    organization: str | None = None,
    base_url: str | None = None,
    outputs: ModelOutputs | None = None,
) -> OpenAIModel:
    """Create an OpenAIModel resource for testing."""
    config = OpenAIModelConfig(
        id=model_id,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        stop=stop,
        timeout=timeout,
        max_retries=max_retries,
        organization=organization,
        base_url=base_url,
    )

    return OpenAIModel(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


def test_model_returns_openai_chat_instance() -> None:
    """Test that model() returns an actual OpenAIChat instance."""
    resource = create_openai_model(
        name="gpt4",
        model_id="gpt-4o",
        api_key="sk-test-key",
    )

    result = resource.model()

    assert isinstance(result, OpenAIChat)


def test_model_passes_api_key_to_openai_chat() -> None:
    """Test that API key is passed to OpenAIChat."""
    resource = create_openai_model(
        api_key="sk-my-secret-key",
    )

    result = resource.model()

    assert result.api_key == "sk-my-secret-key"


def test_model_passes_model_id_to_openai_chat() -> None:
    """Test that model ID is passed to OpenAIChat."""
    resource = create_openai_model(
        model_id="gpt-4o-mini",
    )

    result = resource.model()

    assert result.id == "gpt-4o-mini"


def test_model_with_max_tokens() -> None:
    """Test model() with max_tokens parameter."""
    resource = create_openai_model(
        max_tokens=4096,
    )

    result = resource.model()

    assert result.max_tokens == 4096


def test_model_with_temperature() -> None:
    """Test model() with temperature parameter."""
    resource = create_openai_model(
        temperature=0.7,
    )

    result = resource.model()

    assert result.temperature == 0.7


def test_model_with_top_p() -> None:
    """Test model() with top_p parameter."""
    resource = create_openai_model(
        top_p=0.9,
    )

    result = resource.model()

    assert result.top_p == 0.9


def test_model_with_frequency_penalty() -> None:
    """Test model() with frequency_penalty parameter."""
    resource = create_openai_model(
        frequency_penalty=0.5,
    )

    result = resource.model()

    assert result.frequency_penalty == 0.5


def test_model_with_presence_penalty() -> None:
    """Test model() with presence_penalty parameter."""
    resource = create_openai_model(
        presence_penalty=0.3,
    )

    result = resource.model()

    assert result.presence_penalty == 0.3


def test_model_with_seed() -> None:
    """Test model() with seed parameter."""
    resource = create_openai_model(
        seed=42,
    )

    result = resource.model()

    assert result.seed == 42


def test_model_with_stop_string() -> None:
    """Test model() with stop parameter as string."""
    resource = create_openai_model(
        stop="###",
    )

    result = resource.model()

    assert result.stop == "###"


def test_model_with_stop_list() -> None:
    """Test model() with stop parameter as list."""
    resource = create_openai_model(
        stop=["###", "END"],
    )

    result = resource.model()

    assert result.stop == ["###", "END"]


def test_model_with_timeout() -> None:
    """Test model() with timeout parameter."""
    resource = create_openai_model(
        timeout=30.0,
    )

    result = resource.model()

    assert result.timeout == 30.0


def test_model_with_max_retries() -> None:
    """Test model() with max_retries parameter."""
    resource = create_openai_model(
        max_retries=5,
    )

    result = resource.model()

    assert result.max_retries == 5


def test_model_with_organization() -> None:
    """Test model() with organization parameter."""
    resource = create_openai_model(
        organization="org-123",
    )

    result = resource.model()

    assert result.organization == "org-123"


def test_model_with_base_url() -> None:
    """Test model() with base_url parameter."""
    resource = create_openai_model(
        base_url="https://custom-openai.example.com/v1",
    )

    result = resource.model()

    assert str(result.base_url) == "https://custom-openai.example.com/v1"


def test_model_with_all_parameters() -> None:
    """Test model() with all optional parameters."""
    resource = create_openai_model(
        model_id="gpt-4-turbo",
        api_key="sk-full-test",
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        seed=123,
        stop=["STOP"],
        timeout=60.0,
        max_retries=3,
        organization="org-test",
        base_url="https://api.example.com",
    )

    result = resource.model()

    assert result.id == "gpt-4-turbo"
    assert result.api_key == "sk-full-test"
    assert result.max_tokens == 2048
    assert result.temperature == 0.8
    assert result.top_p == 0.95
    assert result.frequency_penalty == 0.2
    assert result.presence_penalty == 0.1
    assert result.seed == 123
    assert result.stop == ["STOP"]
    assert result.timeout == 60.0
    assert result.max_retries == 3
    assert result.organization == "org-test"


async def test_on_create_returns_serializable_outputs() -> None:
    """Test that on_create returns serializable outputs (no OpenAIChat instance)."""
    resource = create_openai_model(
        name="gpt4",
        model_id="gpt-4o",
        api_key="sk-test-key",
    )

    result = await resource.on_create()

    assert isinstance(result, ModelOutputs)
    assert result.model_id == "gpt-4o"
    assert not hasattr(result, "model") or not isinstance(getattr(result, "model", None), OpenAIChat)


async def test_on_update_returns_serializable_outputs() -> None:
    """Test that on_update returns serializable outputs."""
    resource = create_openai_model(
        name="gpt4",
        model_id="gpt-4o-mini",
        api_key="sk-test-key",
    )

    previous_config = OpenAIModelConfig(
        id="gpt-4o",
        api_key="sk-test-key",
    )

    result = await resource.on_update(previous_config)

    assert isinstance(result, ModelOutputs)
    assert result.model_id == "gpt-4o-mini"


async def test_delete_is_noop() -> None:
    """Test that delete is a no-op for stateless resource."""
    resource = create_openai_model(name="gpt4")

    await resource.on_delete()


def test_provider_name() -> None:
    """Test provider class variable."""
    assert OpenAIModel.provider == "agno"


def test_resource_type() -> None:
    """Test resource class variable."""
    assert OpenAIModel.resource == "models/openai"


def test_config_accepts_various_model_ids() -> None:
    """Test various model IDs are accepted."""
    model_ids = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"]

    for model_id in model_ids:
        config = OpenAIModelConfig(id=model_id, api_key="sk-test")
        assert config.id == model_id


def test_openai_chat_instance_is_usable() -> None:
    """Test that the returned OpenAIChat instance has expected attributes."""
    resource = create_openai_model(
        model_id="gpt-4o",
        api_key="sk-test-key",
        temperature=0.7,
    )

    result = resource.model()

    assert hasattr(result, "id")
    assert hasattr(result, "api_key")
    assert hasattr(result, "temperature")
    assert result.name == "OpenAIChat"
    assert result.provider == "OpenAI"


def test_outputs_are_serializable() -> None:
    """Test that outputs can be serialized to JSON (no arbitrary types)."""
    outputs = ModelOutputs(model_id="gpt-4o")

    json_data = outputs.model_dump_json()

    assert "gpt-4o" in json_data
    assert "model_id" in json_data
