"""Tests for Agno memory/manager resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pragma_sdk import Dependency
from pragma_sdk.provider import ProviderHarness

from agno_provider import (
    DbPostgres,
    DbPostgresConfig,
    Model,
)
from agno_provider.resources.memory import (
    MemoryManager,
    MemoryManagerConfig,
    MemoryManagerOutputs,
)


MemoryManagerConfig.model_rebuild(_types_namespace={"Model": Model, "DbPostgres": DbPostgres})


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
def mock_model_dependency(mocker: MockerFixture, mock_model_resource) -> Dependency[Model]:
    """Create a mock Model dependency with resolved resource."""
    dep = Dependency[Model](
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
    mock_model_dependency: Dependency[Model],
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
    mock_model_dependency: Dependency[Model],
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
        ready=True,
        db_schema="ai",
        has_model=True,
    )

    assert outputs.ready is True
    assert outputs.db_schema == "ai"
    assert outputs.has_model is True

    serialized = outputs.model_dump_json()
    assert "ready" in serialized
    assert "db_schema" in serialized
    assert "has_model" in serialized


async def test_create_with_db_only(
    harness: ProviderHarness,
    mock_db_dependency: Dependency[DbPostgres],
) -> None:
    """on_create with only db dependency returns correct outputs."""
    config = MemoryManagerConfig(db=mock_db_dependency)

    result = await harness.invoke_create(MemoryManager, name="test-memory", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.ready is True
    assert result.outputs.db_schema == "test_schema"
    assert result.outputs.has_model is False


async def test_create_with_model(
    harness: ProviderHarness,
    mock_db_dependency: Dependency[DbPostgres],
    mock_model_dependency: Dependency[Model],
) -> None:
    """on_create with db and model returns has_model=True."""
    config = MemoryManagerConfig(
        db=mock_db_dependency,
        model=mock_model_dependency,
    )

    result = await harness.invoke_create(MemoryManager, name="test-memory", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.ready is True
    assert result.outputs.db_schema == "test_schema"
    assert result.outputs.has_model is True


async def test_create_with_all_options(
    harness: ProviderHarness,
    mock_db_dependency: Dependency[DbPostgres],
    mock_model_dependency: Dependency[Model],
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
    assert result.outputs.ready is True
    assert result.outputs.has_model is True


async def test_memory_manager_method(
    mocker: MockerFixture,
    mock_db_dependency: Dependency[DbPostgres],
    mock_model_dependency: Dependency[Model],
) -> None:
    """memory_manager() returns configured AgnoMemoryManager instance."""
    config = MemoryManagerConfig(
        db=mock_db_dependency,
        model=mock_model_dependency,
        system_message="Test system message",
        memory_capture_instructions="Test capture instructions",
        add_memories=True,
        update_memories=False,
        delete_memories=True,
        clear_memories=False,
        debug_mode=True,
    )

    resource = MemoryManager(name="test-memory", config=config)

    mock_init = mocker.patch(
        "agno.memory.manager.MemoryManager.__init__",
        return_value=None,
    )

    await resource.memory_manager()

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
    assert call_kwargs["model"] is not None


async def test_memory_manager_method_without_model(
    mocker: MockerFixture,
    mock_db_dependency: Dependency[DbPostgres],
) -> None:
    """memory_manager() works without model dependency."""
    config = MemoryManagerConfig(db=mock_db_dependency)

    resource = MemoryManager(name="test-memory", config=config)

    mock_init = mocker.patch(
        "agno.memory.manager.MemoryManager.__init__",
        return_value=None,
    )

    await resource.memory_manager()

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args.kwargs

    assert call_kwargs["db"] is not None
    assert call_kwargs["model"] is None


async def test_update(
    harness: ProviderHarness,
    mock_db_dependency: Dependency[DbPostgres],
    mock_model_dependency: Dependency[Model],
) -> None:
    """on_update returns same outputs as create with updated config."""
    previous = MemoryManagerConfig(db=mock_db_dependency)
    current = MemoryManagerConfig(
        db=mock_db_dependency,
        model=mock_model_dependency,
    )
    previous_outputs = MemoryManagerOutputs(
        ready=True,
        db_schema="test_schema",
        has_model=False,
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
    assert result.outputs.ready is True
    assert result.outputs.db_schema == "test_schema"
    assert result.outputs.has_model is True


async def test_delete(
    harness: ProviderHarness,
    mock_db_dependency: Dependency[DbPostgres],
) -> None:
    """on_delete completes without error (stateless resource)."""
    config = MemoryManagerConfig(db=mock_db_dependency)

    result = await harness.invoke_delete(MemoryManager, name="test-memory", config=config)

    assert result.success
