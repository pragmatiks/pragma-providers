"""Tests for Agno Anthropic model resource."""

from __future__ import annotations

import pytest
from agno.models.anthropic import Claude
from pragma_sdk.provider import ProviderHarness

from agno_provider import (
    AnthropicModel,
    AnthropicModelConfig,
)
from agno_provider.resources.models.anthropic import AnthropicModelOutputs, AnthropicModelSpec


@pytest.fixture
def harness() -> ProviderHarness:
    """Test harness for invoking lifecycle methods."""
    return ProviderHarness()


async def test_create_returns_serializable_outputs(harness: ProviderHarness) -> None:
    """on_create returns serializable outputs (not Claude instance)."""
    config = AnthropicModelConfig(
        id="claude-sonnet-4-20250514",
        api_key="sk-test-key",
    )

    result = await harness.invoke_create(AnthropicModel, name="claude-sonnet", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.id == "claude-sonnet-4-20250514"

    assert not hasattr(result.outputs, "model")


def test_from_spec_returns_claude_instance() -> None:
    """from_spec() returns a Claude instance."""
    spec = AnthropicModelSpec(
        id="claude-sonnet-4-20250514",
        api_key="sk-test-key",
    )

    claude = AnthropicModel.from_spec(spec)

    assert isinstance(claude, Claude)
    assert claude.id == "claude-sonnet-4-20250514"
    assert claude.api_key == "sk-test-key"
    assert claude.max_tokens == 8192  # Default


def test_from_spec_passes_optional_params() -> None:
    """from_spec() passes optional parameters to Claude."""
    spec = AnthropicModelSpec(
        id="claude-sonnet-4-20250514",
        api_key="sk-test-key",
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        stop_sequences=["STOP", "END"],
    )

    claude = AnthropicModel.from_spec(spec)

    assert claude.max_tokens == 4096
    assert claude.temperature == 0.7
    assert claude.top_p == 0.9
    assert claude.top_k == 50
    assert claude.stop_sequences == ["STOP", "END"]


def test_from_spec_omits_none_optional_params() -> None:
    """from_spec() does not pass None values to Claude."""
    spec = AnthropicModelSpec(
        id="claude-sonnet-4-20250514",
        api_key="sk-test-key",
    )

    claude = AnthropicModel.from_spec(spec)

    assert claude.temperature is None
    assert claude.top_p is None
    assert claude.top_k is None
    assert claude.stop_sequences is None


async def test_update_returns_serializable_outputs(harness: ProviderHarness) -> None:
    """on_update returns serializable outputs with updated model_id."""
    previous = AnthropicModelConfig(
        id="claude-3-5-sonnet-20241022",
        api_key="sk-old-key",
    )
    current = AnthropicModelConfig(
        id="claude-sonnet-4-20250514",
        api_key="sk-new-key",
    )

    result = await harness.invoke_update(
        AnthropicModel,
        name="claude-sonnet",
        config=current,
        previous_config=previous,
        current_outputs=AnthropicModelOutputs(
            spec=AnthropicModelSpec(id="claude-3-5-sonnet-20241022", api_key="sk-old-key"),
        ),
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.id == "claude-sonnet-4-20250514"


async def test_update_from_spec_uses_new_config(harness: ProviderHarness) -> None:
    """After update, from_spec() returns Claude with new config."""
    previous = AnthropicModelConfig(
        id="claude-3-5-sonnet-20241022",
        api_key="sk-old-key",
    )
    current = AnthropicModelConfig(
        id="claude-sonnet-4-20250514",
        api_key="sk-new-key",
    )

    result = await harness.invoke_update(
        AnthropicModel,
        name="claude-sonnet",
        config=current,
        previous_config=previous,
        current_outputs=AnthropicModelOutputs(
            spec=AnthropicModelSpec(id="claude-3-5-sonnet-20241022", api_key="sk-old-key"),
        ),
    )

    assert result.success
    claude = AnthropicModel.from_spec(result.outputs.spec)

    assert claude.id == "claude-sonnet-4-20250514"
    assert claude.api_key == "sk-new-key"


async def test_delete_success(harness: ProviderHarness) -> None:
    """on_delete completes without error (stateless resource)."""
    config = AnthropicModelConfig(
        id="claude-sonnet-4-20250514",
        api_key="sk-test-key",
    )

    result = await harness.invoke_delete(AnthropicModel, name="claude-sonnet", config=config)

    assert result.success


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert AnthropicModel.resource == "models/anthropic"


def test_config_defaults() -> None:
    """Config has correct default values."""
    config = AnthropicModelConfig(
        id="claude-sonnet-4-20250514",
        api_key="sk-test-key",
    )

    assert config.max_tokens == 8192  # Agno default
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None
    assert config.stop_sequences is None


def test_outputs_are_serializable() -> None:
    """Outputs contain only serializable data."""
    outputs = AnthropicModelOutputs(
        spec=AnthropicModelSpec(id="claude-sonnet-4-20250514", api_key="sk-test"),
    )

    assert outputs.spec.id == "claude-sonnet-4-20250514"

    serialized = outputs.model_dump_json()
    assert "claude-sonnet-4-20250514" in serialized
