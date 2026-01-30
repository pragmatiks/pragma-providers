"""Tests for GCP Cloud SQL resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from googleapiclient.errors import HttpError
from pragma_sdk.provider import ProviderHarness

from gcp_provider import CloudSQL, CloudSQLConfig, CloudSQLOutputs

if TYPE_CHECKING:
    from pytest_mock import MagicMock, MockerFixture


async def test_create_instance_success(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_create creates instance and waits for RUNNABLE state."""
    config = CloudSQLConfig(
        project_id="test-project",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="test-db",
        database_version="POSTGRES_15",
        tier="db-f1-micro",
        database_name="myapp",
    )

    result = await harness.invoke_create(CloudSQL, name="test-db", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.connection_name == "test-project:europe-west4:test-db"
    assert result.outputs.public_ip == "10.0.0.5"
    assert result.outputs.database_name == "myapp"
    assert result.outputs.ready is True

    mock_sqladmin_service.instances().insert.assert_called_once()
    mock_sqladmin_service.instances().get.assert_called()


async def test_create_instance_idempotent(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_create handles existing instance (idempotent retry)."""
    config = CloudSQLConfig(
        project_id="test-project",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="existing-db",
    )

    result = await harness.invoke_create(CloudSQL, name="existing", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.ready is True


async def test_create_instance_with_authorized_networks(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_create includes authorized networks when specified."""
    config = CloudSQLConfig(
        project_id="test-project",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="test-db",
        authorized_networks=["10.0.0.0/8", "192.168.0.0/16"],
    )

    result = await harness.invoke_create(CloudSQL, name="test-db", config=config)

    assert result.success
    insert_calls = [c for c in mock_sqladmin_service.instances().insert.call_args_list if c.kwargs]

    if insert_calls:
        instance_body = insert_calls[0].kwargs["body"]
        networks = instance_body["settings"]["ipConfiguration"]["authorizedNetworks"]
        assert len(networks) == 2


async def test_create_instance_regional_availability(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_create supports REGIONAL availability type."""
    config = CloudSQLConfig(
        project_id="test-project",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="ha-db",
        availability_type="REGIONAL",
    )

    result = await harness.invoke_create(CloudSQL, name="ha-db", config=config)

    assert result.success
    insert_calls = [c for c in mock_sqladmin_service.instances().insert.call_args_list if c.kwargs]

    if insert_calls:
        instance_body = insert_calls[0].kwargs["body"]
        assert instance_body["settings"]["availabilityType"] == "REGIONAL"


async def test_create_instance_failed_state(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
    mocker: MockerFixture,
) -> None:
    """on_create fails when instance enters FAILED state."""
    failed_instance = {
        "name": "failed-db",
        "state": "FAILED",
        "ipAddresses": [],
    }
    mock_sqladmin_service.instances().get().execute.return_value = failed_instance

    config = CloudSQLConfig(
        project_id="test-project",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="failed-db",
    )

    result = await harness.invoke_create(CloudSQL, name="failed-db", config=config)

    assert result.failed
    assert result.error is not None
    assert "FAILED state" in str(result.error)


async def test_update_unchanged_returns_existing(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update returns existing outputs when config unchanged."""
    previous = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db",
        database_version="POSTGRES_15",
    )
    current = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db",
        database_version="POSTGRES_15",
    )
    existing_outputs = CloudSQLOutputs(
        connection_name="proj:europe-west4:db",
        public_ip="10.0.0.5",
        private_ip=None,
        database_name="app",
        url="postgresql://USER:PASSWORD@10.0.0.5:5432/app",
        ready=True,
        console_url="https://console.cloud.google.com/sql/instances/db/overview?project=proj",
        logs_url="https://console.cloud.google.com/logs/query",
    )

    result = await harness.invoke_update(
        CloudSQL,
        name="db",
        config=current,
        previous_config=previous,
        current_outputs=existing_outputs,
    )

    assert result.success
    assert result.outputs == existing_outputs


async def test_update_rejects_project_change(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update rejects project_id changes."""
    previous = CloudSQLConfig(
        project_id="proj-a",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db",
    )
    current = CloudSQLConfig(
        project_id="proj-b",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db",
    )

    result = await harness.invoke_update(
        CloudSQL,
        name="db",
        config=current,
        previous_config=previous,
    )

    assert result.failed
    assert "project_id" in str(result.error)


async def test_update_rejects_region_change(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update rejects region changes."""
    previous = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db",
    )
    current = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="us-central1",
        instance_name="db",
    )

    result = await harness.invoke_update(
        CloudSQL,
        name="db",
        config=current,
        previous_config=previous,
    )

    assert result.failed
    assert "region" in str(result.error)


async def test_update_rejects_instance_name_change(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update rejects instance_name changes."""
    previous = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db-a",
    )
    current = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db-b",
    )

    result = await harness.invoke_update(
        CloudSQL,
        name="db",
        config=current,
        previous_config=previous,
    )

    assert result.failed
    assert "instance_name" in str(result.error)


async def test_update_rejects_database_version_change(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update rejects database_version changes."""
    previous = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db",
        database_version="POSTGRES_15",
    )
    current = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db",
        database_version="POSTGRES_14",
    )

    result = await harness.invoke_update(
        CloudSQL,
        name="db",
        config=current,
        previous_config=previous,
    )

    assert result.failed
    assert "database_version" in str(result.error)


async def test_delete_success(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
    mocker: MockerFixture,
) -> None:
    """on_delete removes instance."""
    mock_resp = mocker.MagicMock()
    mock_resp.status = 404
    mock_sqladmin_service.instances().get().execute.side_effect = HttpError(mock_resp, b"not found")

    config = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db",
    )

    result = await harness.invoke_delete(CloudSQL, name="db", config=config)

    assert result.success
    mock_sqladmin_service.instances().delete.assert_called()


async def test_delete_idempotent(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
    mocker: MockerFixture,
) -> None:
    """on_delete succeeds when instance doesn't exist."""
    mock_resp = mocker.MagicMock()
    mock_resp.status = 404
    mock_sqladmin_service.instances().delete().execute.side_effect = HttpError(mock_resp, b"not found")
    mock_sqladmin_service.instances().get().execute.side_effect = HttpError(mock_resp, b"not found")

    config = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db",
    )

    result = await harness.invoke_delete(CloudSQL, name="db", config=config)

    assert result.success


async def test_health_healthy(
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
) -> None:
    """health returns healthy when instance is RUNNABLE."""
    config = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db",
    )
    resource = CloudSQL(name="db", config=config, outputs=None)

    health = await resource.health()

    assert health.status == "healthy"
    assert "running" in health.message.lower()


async def test_health_unhealthy_not_found(
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
    mocker: MockerFixture,
) -> None:
    """health returns unhealthy when instance not found."""
    mock_resp = mocker.MagicMock()
    mock_resp.status = 404
    mock_sqladmin_service.instances().get().execute.side_effect = HttpError(mock_resp, b"not found")

    config = CloudSQLConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="db",
    )
    resource = CloudSQL(name="db", config=config, outputs=None)

    health = await resource.health()

    assert health.status == "unhealthy"
    assert "not found" in health.message.lower()


async def test_config_validation_invalid_instance_name() -> None:
    """Config validation rejects invalid instance names."""
    with pytest.raises(ValueError, match="start with a letter"):
        CloudSQLConfig(
            project_id="proj",
            credentials={"type": "service_account"},
            region="europe-west4",
            instance_name="123-invalid",
        )


async def test_config_validation_invalid_database_version() -> None:
    """Config validation rejects unsupported database versions."""
    with pytest.raises(ValueError, match="Unsupported database version"):
        CloudSQLConfig(
            project_id="proj",
            credentials={"type": "service_account"},
            region="europe-west4",
            instance_name="db",
            database_version="ORACLE_19",
        )


async def test_mysql_url_format(
    harness: ProviderHarness,
    mock_sqladmin_service: MagicMock,
    sample_credentials: dict,
) -> None:
    """MySQL instances have correct URL format."""
    config = CloudSQLConfig(
        project_id="test-project",
        credentials=sample_credentials,
        region="europe-west4",
        instance_name="mysql-db",
        database_version="MYSQL_8_0",
    )

    result = await harness.invoke_create(CloudSQL, name="mysql-db", config=config)

    assert result.success
    assert result.outputs is not None
    assert "mysql://" in result.outputs.url
    assert ":3306/" in result.outputs.url
