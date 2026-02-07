"""Tests for Agno memory/manager resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pragma_sdk import Dependency
from pragma_sdk.provider import ProviderHarness

from agno_provider import (
    DbPostgres,
    DbPostgresConfig,
)
from agno_provider.resources.db.postgres import DbPostgresOutputs, DbPostgresSpec
from agno_provider.resources.memory import (
    MemoryManager,
    MemoryManagerConfig,
    MemoryManagerOutputs,
)
from agno_provider.resources.memory.manager import MemoryManagerSpec
from agno_provider.resources.models.anthropic import AnthropicModel
from agno_provider.resources.models.openai import OpenAIModel, OpenAIModelSpec


MemoryManagerConfig.model_rebuild(
    _types_namespace={
        "AnthropicModel": AnthropicModel,
        "OpenAIModel": OpenAIModel,
        "DbPostgres": DbPostgres,
    }
)


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_db_resource(mocker: MockerFixture) -> DbPostgres:
    """Create a mock DbPostgres resource."""
    config = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/testdb",
        db_schema="test_schema",
    )
    resource = DbPostgres(name="test-db", config=config)
    resource.outputs = DbPostgresOutputs(
        spec=DbPostgresSpec(
            connection_url="postgresql://user:pass@localhost:5432/testdb",
            db_schema="test_schema",
        ),
    )
    return resource


@pytest.fixture
def mock_db_dependency(mocker: MockerFixture, mock_db_resource: DbPostgres) -> Dependency[DbPostgres]:
    """Create a mock DbPostgres dependency with resolved resource."""
    dep = Dependency[DbPostgres](
        provider="agno",
        resource="db/postgres",
        name="test-db",
    )
    dep._resolved = mock_db_resource
    return dep


@pytest.fixture
def mock_model_dependency(mocker: MockerFixture, mock_model_resource) -> Dependency[OpenAIModel]:
    """Create a mock OpenAIModel dependency with resolved resource."""
    dep = Dependency[OpenAIModel](
        provider="agno",
        resource="models/openai",
        name="test-model",
    )
    dep._resolved = mock_model_resource
    return dep


def test_resource_metadata_provider_name() -> None:
    """Resource has correct provider name."""
    assert MemoryManager.provider == "agno"


def test_resource_metadata_resource_type() -> None:
    """Resource has correct resource type."""
    assert MemoryManager.resource == "memory/manager"


def test_config_with_db_only(mock_db_dependency: Dependency[DbPostgres]) -> None:
    """Config with only db dependency is valid."""
    config = MemoryManagerConfig(db=mock_db_dependency)

    assert config.db is not None
    assert config.model is None
    assert config.system_message is None
    assert config.memory_capture_instructions is None
    assert config.additional_instructions is None
    assert config.add_memories is True
    assert config.update_memories is True
    assert config.delete_memories is False
    assert config.clear_memories is False
    assert config.debug_mode is False


def test_config_with_model(
    mock_db_dependency: Dependency[DbPostgres],
    mock_model_dependency: Dependency[OpenAIModel],
) -> None:
    """Config with db and model dependency is valid."""
    config = MemoryManagerConfig(
        db=mock_db_dependency,
        model=mock_model_dependency,
    )

    assert config.db is not None
    assert config.model is not None
    assert config.model.name == "test-model"


def test_config_with_all_options(
    mock_db_dependency: Dependency[DbPostgres],
    mock_model_dependency: Dependency[OpenAIModel],
) -> None:
    """Config with all options set is valid."""
    config = MemoryManagerConfig(
        db=mock_db_dependency,
        model=mock_model_dependency,
        system_message="You are a memory classifier.",
        memory_capture_instructions="Capture important details.",
        additional_instructions="Be concise.",
        add_memories=True,
        update_memories=False,
        delete_memories=True,
        clear_memories=True,
        debug_mode=True,
    )

    assert config.system_message == "You are a memory classifier."
    assert config.memory_capture_instructions == "Capture important details."
    assert config.additional_instructions == "Be concise."
    assert config.add_memories is True
    assert config.update_memories is False
    assert config.delete_memories is True
    assert config.clear_memories is True
    assert config.debug_mode is True


def test_outputs_are_serializable() -> None:
    """Outputs contain only serializable data."""
    outputs = MemoryManagerOutputs(
        spec=MemoryManagerSpec(
            db_spec=DbPostgresSpec(
                connection_url="postgresql://user:pass@localhost:5432/db",
                db_schema="ai",
            ),
        ),
    )

    assert outputs.spec is not None
    assert outputs.spec.db_spec.db_schema == "ai"

    serialized = outputs.model_dump_json()
    assert "spec" in serialized
    assert "db_spec" in serialized


async def test_create_with_db_only(
    harness: ProviderHarness,
    mock_db_dependency: Dependency[DbPostgres],
) -> None:
    """on_create with only db dependency returns correct outputs."""
    config = MemoryManagerConfig(db=mock_db_dependency)

    result = await harness.invoke_create(MemoryManager, name="test-memory", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.db_spec.db_schema == "test_schema"
    assert result.outputs.spec.model_spec is None


async def test_create_with_model(
    harness: ProviderHarness,
    mock_db_dependency: Dependency[DbPostgres],
    mock_model_dependency: Dependency[OpenAIModel],
) -> None:
    """on_create with db and model returns outputs with model_spec."""
    config = MemoryManagerConfig(
        db=mock_db_dependency,
        model=mock_model_dependency,
    )

    result = await harness.invoke_create(MemoryManager, name="test-memory", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.db_spec.db_schema == "test_schema"
    assert result.outputs.spec.model_spec is not None


async def test_create_with_all_options(
    harness: ProviderHarness,
    mock_db_dependency: Dependency[DbPostgres],
    mock_model_dependency: Dependency[OpenAIModel],
) -> None:
    """on_create with all options returns correct outputs."""
    config = MemoryManagerConfig(
        db=mock_db_dependency,
        model=mock_model_dependency,
        system_message="Custom system message",
        memory_capture_instructions="Custom capture instructions",
        additional_instructions="Custom additional instructions",
        add_memories=False,
        update_memories=False,
        delete_memories=True,
        clear_memories=True,
        debug_mode=True,
    )

    result = await harness.invoke_create(MemoryManager, name="test-memory", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.model_spec is not None
    assert result.outputs.spec.system_message == "Custom system message"
    assert result.outputs.spec.debug_mode is True


def test_from_spec_with_db_only(mocker: MockerFixture) -> None:
    """from_spec() returns configured AgnoMemoryManager instance with db only."""
    spec = MemoryManagerSpec(
        db_spec=DbPostgresSpec(
            connection_url="postgresql://user:pass@localhost:5432/testdb",
            db_schema="test_schema",
        ),
        system_message="Test system message",
        memory_capture_instructions="Test capture instructions",
        add_memories=True,
        update_memories=False,
        delete_memories=True,
        clear_memories=False,
        debug_mode=True,
    )

    mock_init = mocker.patch(
        "agno.memory.manager.MemoryManager.__init__",
        return_value=None,
    )

    MemoryManager.from_spec(spec)

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args.kwargs

    assert call_kwargs["system_message"] == "Test system message"
    assert call_kwargs["memory_capture_instructions"] == "Test capture instructions"
    assert call_kwargs["add_memories"] is True
    assert call_kwargs["update_memories"] is False
    assert call_kwargs["delete_memories"] is True
    assert call_kwargs["clear_memories"] is False
    assert call_kwargs["debug_mode"] is True
    assert call_kwargs["db"] is not None
    assert call_kwargs["model"] is None


def test_from_spec_with_model(mocker: MockerFixture) -> None:
    """from_spec() constructs model from nested spec."""
    spec = MemoryManagerSpec(
        db_spec=DbPostgresSpec(
            connection_url="postgresql://user:pass@localhost:5432/testdb",
            db_schema="test_schema",
        ),
        model_spec=OpenAIModelSpec(
            id="gpt-4o",
            api_key="sk-test-key",
        ),
    )

    mock_init = mocker.patch(
        "agno.memory.manager.MemoryManager.__init__",
        return_value=None,
    )

    MemoryManager.from_spec(spec)

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args.kwargs

    assert call_kwargs["db"] is not None
    assert call_kwargs["model"] is not None


async def test_update(
    harness: ProviderHarness,
    mock_db_dependency: Dependency[DbPostgres],
    mock_model_dependency: Dependency[OpenAIModel],
) -> None:
    """on_update returns same outputs as create with updated config."""
    previous = MemoryManagerConfig(db=mock_db_dependency)
    current = MemoryManagerConfig(
        db=mock_db_dependency,
        model=mock_model_dependency,
    )
    previous_outputs = MemoryManagerOutputs(
        spec=MemoryManagerSpec(
            db_spec=DbPostgresSpec(
                connection_url="postgresql://user:pass@localhost:5432/testdb",
                db_schema="test_schema",
            ),
        ),
    )

    result = await harness.invoke_update(
        MemoryManager,
        name="test-memory",
        config=current,
        previous_config=previous,
        current_outputs=previous_outputs,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.spec.db_spec.db_schema == "test_schema"
    assert result.outputs.spec.model_spec is not None


async def test_delete(
    harness: ProviderHarness,
    mock_db_dependency: Dependency[DbPostgres],
) -> None:
    """on_delete completes without error (stateless resource)."""
    config = MemoryManagerConfig(db=mock_db_dependency)

    result = await harness.invoke_delete(MemoryManager, name="test-memory", config=config)

    assert result.success
