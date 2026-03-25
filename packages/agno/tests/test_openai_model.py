"""Tests for Agno OpenAI Model resource."""

from __future__ import annotations

from agno.models.openai import OpenAIChat
from pragma_sdk import LifecycleState

from agno_provider import (
    OpenAIModel,
    OpenAIModelConfig,
)
from agno_provider.resources.models.openai import OpenAIModelOutputs, OpenAIModelSpec


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
    outputs: OpenAIModelOutputs | None = None,
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


def test_from_spec_returns_openai_chat_instance() -> None:
    """Test that from_spec() returns an actual OpenAIChat instance."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
    )

    result = OpenAIModel.from_spec(spec)

    assert isinstance(result, OpenAIChat)


def test_from_spec_passes_api_key_to_openai_chat() -> None:
    """Test that API key is passed to OpenAIChat."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-my-secret-key",
    )

    result = OpenAIModel.from_spec(spec)

    assert result.api_key == "sk-my-secret-key"


def test_from_spec_passes_model_id_to_openai_chat() -> None:
    """Test that model ID is passed to OpenAIChat."""
    spec = OpenAIModelSpec(
        id="gpt-4o-mini",
        api_key="sk-test-key",
    )

    result = OpenAIModel.from_spec(spec)

    assert result.id == "gpt-4o-mini"


def test_from_spec_with_max_tokens() -> None:
    """Test from_spec() with max_tokens parameter."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        max_tokens=4096,
    )

    result = OpenAIModel.from_spec(spec)

    assert result.max_tokens == 4096


def test_from_spec_with_temperature() -> None:
    """Test from_spec() with temperature parameter."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        temperature=0.7,
    )

    result = OpenAIModel.from_spec(spec)

    assert result.temperature == 0.7


def test_from_spec_with_top_p() -> None:
    """Test from_spec() with top_p parameter."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        top_p=0.9,
    )

    result = OpenAIModel.from_spec(spec)

    assert result.top_p == 0.9


def test_from_spec_with_frequency_penalty() -> None:
    """Test from_spec() with frequency_penalty parameter."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        frequency_penalty=0.5,
    )

    result = OpenAIModel.from_spec(spec)

    assert result.frequency_penalty == 0.5


def test_from_spec_with_presence_penalty() -> None:
    """Test from_spec() with presence_penalty parameter."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        presence_penalty=0.3,
    )

    result = OpenAIModel.from_spec(spec)

    assert result.presence_penalty == 0.3


def test_from_spec_with_seed() -> None:
    """Test from_spec() with seed parameter."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        seed=42,
    )

    result = OpenAIModel.from_spec(spec)

    assert result.seed == 42


def test_from_spec_with_stop_string() -> None:
    """Test from_spec() with stop parameter as string."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        stop="###",
    )

    result = OpenAIModel.from_spec(spec)

    assert result.stop == "###"


def test_from_spec_with_stop_list() -> None:
    """Test from_spec() with stop parameter as list."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        stop=["###", "END"],
    )

    result = OpenAIModel.from_spec(spec)

    assert result.stop == ["###", "END"]


def test_from_spec_with_timeout() -> None:
    """Test from_spec() with timeout parameter."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        timeout=30.0,
    )

    result = OpenAIModel.from_spec(spec)

    assert result.timeout == 30.0


def test_from_spec_with_max_retries() -> None:
    """Test from_spec() with max_retries parameter."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        max_retries=5,
    )

    result = OpenAIModel.from_spec(spec)

    assert result.max_retries == 5


def test_from_spec_with_organization() -> None:
    """Test from_spec() with organization parameter."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        organization="org-123",
    )

    result = OpenAIModel.from_spec(spec)

    assert result.organization == "org-123"


def test_from_spec_with_base_url() -> None:
    """Test from_spec() with base_url parameter."""
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        base_url="https://custom-openai.example.com/v1",
    )

    result = OpenAIModel.from_spec(spec)

    assert str(result.base_url) == "https://custom-openai.example.com/v1"


def test_from_spec_with_all_parameters() -> None:
    """Test from_spec() with all optional parameters."""
    spec = OpenAIModelSpec(
        id="gpt-4-turbo",
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

    result = OpenAIModel.from_spec(spec)

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

    assert isinstance(result, OpenAIModelOutputs)
    assert result.spec.id == "gpt-4o"
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

    assert isinstance(result, OpenAIModelOutputs)
    assert result.spec.id == "gpt-4o-mini"


async def test_delete_is_noop() -> None:
    """Test that delete is a no-op for stateless resource."""
    resource = create_openai_model(name="gpt4")

    await resource.on_delete()


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
    spec = OpenAIModelSpec(
        id="gpt-4o",
        api_key="sk-test-key",
        temperature=0.7,
    )

    result = OpenAIModel.from_spec(spec)

    assert hasattr(result, "id")
    assert hasattr(result, "api_key")
    assert hasattr(result, "temperature")
    assert result.name == "OpenAIChat"
    assert result.provider == "OpenAI"


def test_outputs_are_serializable() -> None:
    """Test that outputs can be serialized to JSON (no arbitrary types)."""
    outputs = OpenAIModelOutputs(
        spec=OpenAIModelSpec(id="gpt-4o", api_key="sk-test"),
    )

    json_data = outputs.model_dump_json()

    assert "gpt-4o" in json_data
    assert "spec" in json_data
