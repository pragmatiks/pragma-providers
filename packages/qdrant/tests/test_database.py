"""Tests for Qdrant Database resource."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from pragma_sdk import Dependency, LifecycleState

from qdrant_provider import (
    Database,
    DatabaseConfig,
    DatabaseOutputs,
    ResourceConfig,
    StorageConfig,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_database_with_mocked_dependency(
    name: str,
    replicas: int = 1,
    storage: StorageConfig | None = None,
    resources: ResourceConfig | None = None,
    api_key: str | None = None,
    outputs: DatabaseOutputs | None = None,
) -> Database:
    """Create a Database instance with mocked dependency.

    Note: The actual lifecycle methods (on_create, on_update) call
    apply() and wait_ready() which require mocking the SDK context.
    """
    dep = Dependency(provider="gcp", resource="gke", name="test-cluster")

    config = DatabaseConfig(
        cluster=dep,
        replicas=replicas,
        storage=storage,
        resources=resources,
        api_key=api_key,
    )

    return Database(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


@pytest.fixture
def mock_apply_and_wait(mocker: "MockerFixture"):
    """Mock the apply(), wait_ready(), and _wait_for_load_balancer_ip methods."""
    mock_apply = mocker.patch(
        "pragma_sdk.models.base.Resource.apply",
        new_callable=AsyncMock,
    )
    mock_wait = mocker.patch(
        "pragma_sdk.models.base.Resource.wait_ready",
        new_callable=AsyncMock,
    )
    mock_lb_ip = mocker.patch(
        "qdrant_provider.resources.database.Database._wait_for_load_balancer_ip",
        new_callable=AsyncMock,
        return_value="34.123.45.67",
    )
    return mock_apply, mock_wait, mock_lb_ip


async def test_create_database_success(
    mock_apply_and_wait: tuple[AsyncMock, AsyncMock, AsyncMock],
) -> None:
    """on_create creates child resources and returns outputs."""
    mock_apply, mock_wait, mock_lb_ip = mock_apply_and_wait

    db = create_database_with_mocked_dependency(name="test-db")

    result = await db.on_create()

    assert result.url == "http://34.123.45.67:6333"
    assert result.grpc_url == "http://34.123.45.67:6334"
    assert result.api_key is None
    assert result.ready is True

    assert mock_apply.call_count == 3
    assert mock_wait.call_count == 3
    mock_lb_ip.assert_called_once()


async def test_create_database_with_storage(
    mock_apply_and_wait: tuple[AsyncMock, AsyncMock, AsyncMock],
) -> None:
    """on_create handles storage configuration."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        storage=StorageConfig(size="20Gi", class_="premium-rwo"),
    )

    result = await db.on_create()

    assert result.ready is True


async def test_create_database_with_resources(
    mock_apply_and_wait: tuple[AsyncMock, AsyncMock, AsyncMock],
) -> None:
    """on_create handles resource limits."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        replicas=3,
        resources=ResourceConfig(memory="4Gi", cpu="2"),
    )

    result = await db.on_create()

    assert result.ready is True


async def test_create_database_with_api_key(
    mock_apply_and_wait: tuple[AsyncMock, AsyncMock, AsyncMock],
) -> None:
    """on_create configures API key authentication."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        api_key="secret-api-key",
    )

    result = await db.on_create()

    assert result.api_key == "secret-api-key"
    assert result.ready is True

    sts = db._build_statefulset()
    assert sts.config.containers[0].env is not None
    assert len(sts.config.containers[0].env) == 1
    assert sts.config.containers[0].env[0].name == "QDRANT__SERVICE__API_KEY"
    assert sts.config.containers[0].env[0].value == "secret-api-key"


async def test_update_unchanged_returns_existing(
    mock_apply_and_wait: tuple[AsyncMock, AsyncMock, AsyncMock],
) -> None:
    """on_update returns existing outputs when config unchanged."""
    mock_apply, mock_wait, mock_lb_ip = mock_apply_and_wait
    mock_apply.reset_mock()
    mock_wait.reset_mock()
    mock_lb_ip.reset_mock()

    existing_outputs = DatabaseOutputs(
        url="http://34.123.45.67:6333",
        grpc_url="http://34.123.45.67:6334",
        api_key=None,
        ready=True,
    )

    db = create_database_with_mocked_dependency(
        name="test-db",
        outputs=existing_outputs,
    )

    previous_config = DatabaseConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        replicas=1,
    )

    result = await db.on_update(previous_config)

    assert result == existing_outputs
    mock_apply.assert_not_called()
    mock_lb_ip.assert_not_called()


async def test_update_replicas_applies_changes(
    mock_apply_and_wait: tuple[AsyncMock, AsyncMock, AsyncMock],
) -> None:
    """on_update applies changes when replicas change."""
    mock_apply, mock_wait, mock_lb_ip = mock_apply_and_wait

    db = create_database_with_mocked_dependency(
        name="test-db",
        replicas=3,
        outputs=DatabaseOutputs(
            url="http://34.123.45.67:6333",
            grpc_url="http://34.123.45.67:6334",
            api_key=None,
            ready=True,
        ),
    )

    previous_config = DatabaseConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        replicas=1,
    )

    result = await db.on_update(previous_config)

    assert result.ready is True
    assert mock_apply.call_count == 3
    mock_lb_ip.assert_called_once()


async def test_update_rejects_cluster_change() -> None:
    """on_update rejects cluster changes."""
    db = create_database_with_mocked_dependency(name="test-db")

    previous_config = DatabaseConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="other-cluster"),
        replicas=1,
    )

    with pytest.raises(ValueError, match="cluster"):
        await db.on_update(previous_config)


async def test_delete_is_noop() -> None:
    """on_delete is a no-op (cascading deletes handle cleanup)."""
    db = create_database_with_mocked_dependency(name="test-db")

    await db.on_delete()


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert Database.provider == "qdrant"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Database.resource == "database"


def test_storage_config_alias() -> None:
    """StorageConfig uses 'class' alias for class_ field."""
    storage = StorageConfig(size="10Gi")
    assert storage.class_ == "standard-rwo"

    storage = StorageConfig.model_validate({"size": "20Gi", "class": "premium-rwo"})
    assert storage.class_ == "premium-rwo"


def test_resource_config_defaults() -> None:
    """ResourceConfig has sensible defaults."""
    resources = ResourceConfig()
    assert resources.memory == "2Gi"
    assert resources.cpu == "1"


async def test_build_outputs(mocker: "MockerFixture") -> None:
    """_build_outputs creates correct external URLs."""
    mocker.patch(
        "qdrant_provider.resources.database.Database._wait_for_load_balancer_ip",
        new_callable=AsyncMock,
        return_value="34.123.45.67",
    )

    db = create_database_with_mocked_dependency(name="my-qdrant")

    outputs = await db._build_outputs()

    assert outputs.url == "http://34.123.45.67:6333"
    assert outputs.grpc_url == "http://34.123.45.67:6334"
    assert outputs.api_key is None
    assert outputs.ready is True


async def test_build_outputs_with_api_key(mocker: "MockerFixture") -> None:
    """_build_outputs includes api_key when configured."""
    mocker.patch(
        "qdrant_provider.resources.database.Database._wait_for_load_balancer_ip",
        new_callable=AsyncMock,
        return_value="34.123.45.67",
    )

    db = create_database_with_mocked_dependency(name="my-qdrant", api_key="my-secret")

    outputs = await db._build_outputs()

    assert outputs.api_key == "my-secret"


def test_service_names() -> None:
    """Service names follow naming convention."""
    db = create_database_with_mocked_dependency(name="test-db")

    assert db._headless_service_name() == "qdrant-test-db-headless"
    assert db._client_service_name() == "qdrant-test-db"
    assert db._statefulset_name() == "qdrant-test-db"


def test_build_headless_service() -> None:
    """_build_headless_service creates correct Service config."""
    db = create_database_with_mocked_dependency(name="test-db")

    svc = db._build_headless_service()

    assert svc.name == "qdrant-test-db-headless"
    assert svc.config.type == "Headless"
    assert len(svc.config.ports) == 2


def test_build_client_service() -> None:
    """_build_client_service creates LoadBalancer Service."""
    db = create_database_with_mocked_dependency(name="test-db")

    svc = db._build_client_service()

    assert svc.name == "qdrant-test-db"
    assert svc.config.type == "LoadBalancer"
    assert len(svc.config.ports) == 2


def test_build_statefulset() -> None:
    """_build_statefulset creates correct StatefulSet config."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        replicas=3,
        storage=StorageConfig(size="20Gi", class_="premium-rwo"),
    )

    sts = db._build_statefulset()

    assert sts.name == "qdrant-test-db"
    assert sts.config.replicas == 3
    assert sts.config.service_name == "qdrant-test-db-headless"
    assert len(sts.config.containers) == 1
    assert sts.config.containers[0].name == "qdrant"
    assert sts.config.containers[0].env is None
    assert len(sts.config.volume_claim_templates) == 1
    assert sts.config.volume_claim_templates[0].storage == "20Gi"
    assert sts.config.volume_claim_templates[0].storage_class == "premium-rwo"


def test_build_statefulset_with_api_key() -> None:
    """_build_statefulset configures API key environment variable."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        api_key="test-secret-key",
    )

    sts = db._build_statefulset()

    assert sts.config.containers[0].env is not None
    assert len(sts.config.containers[0].env) == 1
    assert sts.config.containers[0].env[0].name == "QDRANT__SERVICE__API_KEY"
    assert sts.config.containers[0].env[0].value == "test-secret-key"


async def test_health_delegates_to_statefulset(
    mock_apply_and_wait: tuple[AsyncMock, AsyncMock, AsyncMock],
    mocker: "MockerFixture",
) -> None:
    """health() delegates to underlying StatefulSet."""
    from pragma_sdk import HealthStatus

    mock_sts_health = mocker.patch(
        "kubernetes_provider.StatefulSet.health",
        new_callable=AsyncMock,
        return_value=HealthStatus(status="healthy", message="All replicas ready"),
    )

    db = create_database_with_mocked_dependency(name="test-db")

    result = await db.health()

    assert result.status == "healthy"
    mock_sts_health.assert_called_once()


async def test_logs_delegates_to_statefulset(
    mock_apply_and_wait: tuple[AsyncMock, AsyncMock, AsyncMock],
    mocker: "MockerFixture",
) -> None:
    """logs() delegates to underlying StatefulSet."""
    from datetime import datetime, timezone

    from pragma_sdk import LogEntry

    async def mock_logs(*args, **kwargs):
        yield LogEntry(
            timestamp=datetime.now(timezone.utc),
            level="info",
            message="test log",
            metadata={"pod": "qdrant-test-db-0"},
        )

    mocker.patch(
        "kubernetes_provider.StatefulSet.logs",
        side_effect=mock_logs,
    )

    db = create_database_with_mocked_dependency(name="test-db")

    entries = []
    async for entry in db.logs():
        entries.append(entry)

    assert len(entries) == 1
    assert entries[0].message == "test log"
