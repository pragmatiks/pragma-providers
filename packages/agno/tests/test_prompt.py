"""Tests for Agno prompt resource."""

from __future__ import annotations

import pytest
from pragma_sdk.provider import ProviderHarness
from pydantic import ValidationError

from agno_provider import (
    Prompt,
    PromptConfig,
    PromptOutputs,
)


async def test_create_with_instructions_only(harness: ProviderHarness) -> None:
    """on_create with instructions returns rendered text."""
    config = PromptConfig(
        instructions=["You are a helpful assistant.", "Be concise."],
    )

    result = await harness.invoke_create(Prompt, name="system-prompt", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.text == "You are a helpful assistant.\nBe concise."
    assert result.outputs.instruction_count == 2


async def test_create_with_template_only(harness: ProviderHarness) -> None:
    """on_create with template interpolates variables."""
    config = PromptConfig(
        template="Hello {{name}}, your role is {{role}}.",
        variables={"name": "Assistant", "role": "helper"},
    )

    result = await harness.invoke_create(Prompt, name="greeting", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.text == "Hello Assistant, your role is helper."
    assert result.outputs.instruction_count == 1


async def test_create_with_instructions_and_template(harness: ProviderHarness) -> None:
    """on_create with both instructions and template combines them."""
    config = PromptConfig(
        instructions=["Base instruction."],
        template="Dynamic: {{action}}",
        variables={"action": "assist users"},
    )

    result = await harness.invoke_create(Prompt, name="combined", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.text == "Base instruction.\nDynamic: assist users"
    assert result.outputs.instruction_count == 2


async def test_render_method_returns_text(harness: ProviderHarness) -> None:
    """render() method returns the formatted prompt text."""
    config = PromptConfig(
        instructions=["Line 1", "Line 2"],
    )

    result = await harness.invoke_create(Prompt, name="render-test", config=config)

    assert result.success
    assert result.resource is not None

    text = result.resource.render()

    assert text == "Line 1\nLine 2"


def test_validation_empty_config_fails() -> None:
    """Config validation fails when neither instructions nor template provided."""
    with pytest.raises(ValidationError) as exc_info:
        PromptConfig()

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "At least one of 'instructions' or 'template' must be provided" in str(errors[0]["msg"])


def test_validation_missing_variable_fails() -> None:
    """Config validation fails when template references undefined variable."""
    with pytest.raises(ValidationError) as exc_info:
        PromptConfig(
            template="Hello {{name}}, {{greeting}}!",
            variables={"name": "World"},
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "Missing variables for template placeholders" in str(errors[0]["msg"])
    assert "greeting" in str(errors[0]["msg"])


def test_validation_extra_variable_allowed() -> None:
    """Config allows extra variables not referenced in template."""
    config = PromptConfig(
        template="Hello {{name}}!",
        variables={"name": "World", "unused": "value"},
    )

    assert config.variables["unused"] == "value"


async def test_update_changes_outputs(harness: ProviderHarness) -> None:
    """on_update re-renders with new configuration."""
    previous = PromptConfig(
        instructions=["Old instruction."],
    )
    current = PromptConfig(
        instructions=["New instruction.", "Added line."],
    )
    previous_outputs = PromptOutputs(text="Old instruction.", instruction_count=1)

    result = await harness.invoke_update(
        Prompt,
        name="system-prompt",
        config=current,
        previous_config=previous,
        current_outputs=previous_outputs,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.text == "New instruction.\nAdded line."
    assert result.outputs.instruction_count == 2


async def test_update_with_variable_changes(harness: ProviderHarness) -> None:
    """on_update re-renders when variables change."""
    previous = PromptConfig(
        template="Hello {{name}}!",
        variables={"name": "Alice"},
    )
    current = PromptConfig(
        template="Hello {{name}}!",
        variables={"name": "Bob"},
    )
    previous_outputs = PromptOutputs(text="Hello Alice!", instruction_count=1)

    result = await harness.invoke_update(
        Prompt,
        name="greeting",
        config=current,
        previous_config=previous,
        current_outputs=previous_outputs,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.text == "Hello Bob!"


async def test_delete_success(harness: ProviderHarness) -> None:
    """on_delete completes without error (stateless resource)."""
    config = PromptConfig(
        instructions=["Test instruction."],
    )

    result = await harness.invoke_delete(Prompt, name="system-prompt", config=config)

    assert result.success


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert Prompt.provider == "agno"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Prompt.resource == "prompt"


def test_outputs_are_serializable() -> None:
    """Outputs contain only serializable data."""
    outputs = PromptOutputs(text="Test prompt", instruction_count=1)

    assert outputs.text == "Test prompt"
    assert outputs.instruction_count == 1

    serialized = outputs.model_dump_json()
    assert "Test prompt" in serialized
    assert "1" in serialized


async def test_multiline_template(harness: ProviderHarness) -> None:
    """Template can span multiple lines."""
    config = PromptConfig(
        template="Line 1: {{first}}\nLine 2: {{second}}",
        variables={"first": "A", "second": "B"},
    )

    result = await harness.invoke_create(Prompt, name="multiline", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.text == "Line 1: A\nLine 2: B"
    assert result.outputs.instruction_count == 2


async def test_empty_instructions_with_template(harness: ProviderHarness) -> None:
    """Empty instructions list with template still works."""
    config = PromptConfig(
        instructions=[],
        template="Just template {{var}}",
        variables={"var": "text"},
    )

    result = await harness.invoke_create(Prompt, name="empty-list", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.text == "Just template text"


async def test_multiple_same_variable(harness: ProviderHarness) -> None:
    """Same variable used multiple times in template."""
    config = PromptConfig(
        template="{{name}} says hello. Remember, I am {{name}}.",
        variables={"name": "Assistant"},
    )

    result = await harness.invoke_create(Prompt, name="repeated-var", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.text == "Assistant says hello. Remember, I am Assistant."
