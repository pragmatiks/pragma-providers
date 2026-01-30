"""Tests for GCP Cloud SQL resources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from googleapiclient.errors import HttpError
from pragma_sdk.provider import ProviderHarness

from gcp_provider import (
    Database,
    DatabaseConfig,
    DatabaseInstance,
    DatabaseInstanceConfig,
    User,
    UserConfig,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def make_database_config(
    mocker: MockerFixture,
    project_id: str = "test-project",
    credentials: dict | None = None,
    instance_name: str = "test-db",
    database_name: str = "myapp",
) -> DatabaseConfig:
    """Create a DatabaseConfig with mocked instance dependency for testing."""
    if credentials is None:
        credentials = {"type": "service_account"}

    mock_instance = mocker.MagicMock()
    mock_instance.config = DatabaseInstanceConfig(
        project_id=project_id,
        credentials=credentials,
        region="europe-west4",
        instance_name=instance_name,
    )

    return DatabaseConfig.model_construct(
        instance=mock_instance,
        database_name=database_name,
    )


def make_user_config(
    mocker: MockerFixture,
    project_id: str = "test-project",
    credentials: dict | None = None,
    instance_name: str = "test-db",
    username: str = "appuser",
    password: str = "secret123",
) -> UserConfig:
    """Create a UserConfig with mocked instance dependency for testing."""
    if credentials is None:
        credentials = {"type": "service_account"}

    mock_instance = mocker.MagicMock()
    mock_instance.config = DatabaseInstanceConfig(
        project_id=project_id,
        credentials=credentials,
        region="europe-west4",
        instance_name=instance_name,
    )

    return UserConfig.model_construct(
        instance=mock_instance,
        username=username,
        password=password,
    )


class TestDatabaseInstance:
    """Tests for CloudSQL DatabaseInstance resource."""

    async def test_create_instance_success(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
    ) -> None:
        """on_create creates instance and waits for RUNNABLE state."""
        config = DatabaseInstanceConfig(
            project_id="test-project",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="test-db",
            database_version="POSTGRES_15",
            tier="db-f1-micro",
        )

        result = await harness.invoke_create(DatabaseInstance, name="test-db", config=config)

        assert result.success
        assert result.outputs is not None
        assert result.outputs.connection_name == "test-project:europe-west4:test-db"
        assert result.outputs.public_ip == "10.0.0.5"
        assert result.outputs.ready is True

        mock_sqladmin_service.instances().insert.assert_called_once()
        mock_sqladmin_service.instances().get.assert_called()

    async def test_create_instance_idempotent(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
    ) -> None:
        """on_create handles existing instance (idempotent retry)."""
        config = DatabaseInstanceConfig(
            project_id="test-project",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="existing-db",
        )

        result = await harness.invoke_create(DatabaseInstance, name="existing", config=config)

        assert result.success
        assert result.outputs is not None
        assert result.outputs.ready is True

    async def test_create_instance_with_authorized_networks(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
    ) -> None:
        """on_create includes authorized networks when specified."""
        config = DatabaseInstanceConfig(
            project_id="test-project",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="test-db",
            authorized_networks=["10.0.0.0/8", "192.168.0.0/16"],
        )

        result = await harness.invoke_create(DatabaseInstance, name="test-db", config=config)

        assert result.success
        insert_calls = [c for c in mock_sqladmin_service.instances().insert.call_args_list if c.kwargs]

        if insert_calls:
            instance_body = insert_calls[0].kwargs["body"]
            networks = instance_body["settings"]["ipConfiguration"]["authorizedNetworks"]
            assert len(networks) == 2

    async def test_create_instance_regional_availability(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
    ) -> None:
        """on_create supports REGIONAL availability type."""
        config = DatabaseInstanceConfig(
            project_id="test-project",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="ha-db",
            availability_type="REGIONAL",
        )

        result = await harness.invoke_create(DatabaseInstance, name="ha-db", config=config)

        assert result.success
        insert_calls = [c for c in mock_sqladmin_service.instances().insert.call_args_list if c.kwargs]

        if insert_calls:
            instance_body = insert_calls[0].kwargs["body"]
            assert instance_body["settings"]["availabilityType"] == "REGIONAL"

    async def test_create_instance_failed_state(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
    ) -> None:
        """on_create fails when instance enters FAILED state."""
        failed_instance = {
            "name": "failed-db",
            "state": "FAILED",
            "ipAddresses": [],
        }
        mock_sqladmin_service.instances().get().execute.return_value = failed_instance

        config = DatabaseInstanceConfig(
            project_id="test-project",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="failed-db",
        )

        result = await harness.invoke_create(DatabaseInstance, name="failed-db", config=config)

        assert result.failed
        assert result.error is not None
        assert "FAILED state" in str(result.error)

    async def test_update_applies_mutable_changes(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
    ) -> None:
        """on_update applies mutable config changes via patch API."""
        previous = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db",
            database_version="POSTGRES_15",
            tier="db-f1-micro",
        )
        current = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db",
            database_version="POSTGRES_15",
            tier="db-custom-1-3840",
        )

        result = await harness.invoke_update(
            DatabaseInstance,
            name="db",
            config=current,
            previous_config=previous,
        )

        assert result.success
        assert result.outputs is not None
        assert result.outputs.ready is True
        mock_sqladmin_service.instances().patch.assert_called()

    async def test_update_rejects_project_change(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
    ) -> None:
        """on_update rejects project_id changes."""
        previous = DatabaseInstanceConfig(
            project_id="proj-a",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db",
        )
        current = DatabaseInstanceConfig(
            project_id="proj-b",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db",
        )

        result = await harness.invoke_update(
            DatabaseInstance,
            name="db",
            config=current,
            previous_config=previous,
        )

        assert result.failed
        assert "project_id" in str(result.error)

    async def test_update_rejects_region_change(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
    ) -> None:
        """on_update rejects region changes."""
        previous = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db",
        )
        current = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="us-central1",
            instance_name="db",
        )

        result = await harness.invoke_update(
            DatabaseInstance,
            name="db",
            config=current,
            previous_config=previous,
        )

        assert result.failed
        assert "region" in str(result.error)

    async def test_update_rejects_instance_name_change(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
    ) -> None:
        """on_update rejects instance_name changes."""
        previous = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db-a",
        )
        current = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db-b",
        )

        result = await harness.invoke_update(
            DatabaseInstance,
            name="db",
            config=current,
            previous_config=previous,
        )

        assert result.failed
        assert "instance_name" in str(result.error)

    async def test_update_rejects_database_version_change(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
    ) -> None:
        """on_update rejects database_version changes."""
        previous = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db",
            database_version="POSTGRES_15",
        )
        current = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db",
            database_version="POSTGRES_14",
        )

        result = await harness.invoke_update(
            DatabaseInstance,
            name="db",
            config=current,
            previous_config=previous,
        )

        assert result.failed
        assert "database_version" in str(result.error)

    async def test_delete_success(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_delete removes instance."""
        mock_resp = mocker.MagicMock()
        mock_resp.status = 404
        mock_sqladmin_service.instances().get().execute.side_effect = HttpError(mock_resp, b"not found")

        config = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db",
        )

        result = await harness.invoke_delete(DatabaseInstance, name="db", config=config)

        assert result.success
        mock_sqladmin_service.instances().delete.assert_called()

    async def test_delete_idempotent(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_delete succeeds when instance doesn't exist."""
        mock_resp = mocker.MagicMock()
        mock_resp.status = 404
        mock_sqladmin_service.instances().delete().execute.side_effect = HttpError(mock_resp, b"not found")
        mock_sqladmin_service.instances().get().execute.side_effect = HttpError(mock_resp, b"not found")

        config = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db",
        )

        result = await harness.invoke_delete(DatabaseInstance, name="db", config=config)

        assert result.success

    async def test_health_healthy(
        self,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
    ) -> None:
        """health returns healthy when instance is RUNNABLE."""
        config = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db",
        )
        resource = DatabaseInstance(name="db", config=config, outputs=None)

        health = await resource.health()

        assert health.status == "healthy"
        assert "running" in health.message.lower()

    async def test_health_unhealthy_not_found(
        self,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """health returns unhealthy when instance not found."""
        mock_resp = mocker.MagicMock()
        mock_resp.status = 404
        mock_sqladmin_service.instances().get().execute.side_effect = HttpError(mock_resp, b"not found")

        config = DatabaseInstanceConfig(
            project_id="proj",
            credentials=sample_credentials,
            region="europe-west4",
            instance_name="db",
        )
        resource = DatabaseInstance(name="db", config=config, outputs=None)

        health = await resource.health()

        assert health.status == "unhealthy"
        assert "not found" in health.message.lower()

    async def test_config_validation_invalid_instance_name(self) -> None:
        """Config validation rejects invalid instance names."""
        with pytest.raises(ValueError, match="start with a letter"):
            DatabaseInstanceConfig(
                project_id="proj",
                credentials={"type": "service_account"},
                region="europe-west4",
                instance_name="123-invalid",
            )

    async def test_config_validation_invalid_database_version(self) -> None:
        """Config validation rejects unsupported database versions."""
        with pytest.raises(ValueError, match="Unsupported database version"):
            DatabaseInstanceConfig(
                project_id="proj",
                credentials={"type": "service_account"},
                region="europe-west4",
                instance_name="db",
                database_version="ORACLE_19",
            )


class TestDatabase:
    """Tests for CloudSQL Database resource."""

    async def test_create_database_success(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_create creates database in instance."""
        config = make_database_config(
            mocker,
            project_id="test-project",
            credentials=sample_credentials,
            instance_name="test-db",
            database_name="myapp",
        )

        result = await harness.invoke_create(Database, name="myapp-db", config=config)

        assert result.success
        assert result.outputs is not None
        assert result.outputs.database_name == "myapp"
        assert result.outputs.instance_name == "test-db"
        assert "myapp" in result.outputs.url

        mock_sqladmin_service.databases().insert.assert_called()

    async def test_create_database_idempotent(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_create handles existing database (idempotent retry)."""
        mock_sqladmin_service.databases().get().execute.return_value = {
            "name": "myapp",
            "instance": "test-db",
        }

        config = make_database_config(
            mocker,
            project_id="test-project",
            credentials=sample_credentials,
            instance_name="test-db",
            database_name="myapp",
        )

        result = await harness.invoke_create(Database, name="myapp-db", config=config)

        assert result.success
        assert result.outputs is not None
        assert result.outputs.database_name == "myapp"

    async def test_create_database_mysql_url(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """MySQL databases have correct URL format."""
        mock_sqladmin_service.instances().get().execute.return_value = {
            "name": "mysql-db",
            "state": "RUNNABLE",
            "settings": {"tier": "db-f1-micro"},
            "databaseVersion": "MYSQL_8_0",
            "region": "europe-west4",
            "ipAddresses": [{"type": "PRIMARY", "ipAddress": "10.0.0.5"}],
        }

        config = make_database_config(
            mocker,
            project_id="test-project",
            credentials=sample_credentials,
            instance_name="mysql-db",
            database_name="myapp",
        )

        result = await harness.invoke_create(Database, name="myapp-db", config=config)

        assert result.success
        assert result.outputs is not None
        assert "mysql://" in result.outputs.url
        assert ":3306/" in result.outputs.url

    async def test_update_instance_change_triggers_replacement(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_update replaces database when instance changes (delete + create)."""
        previous = make_database_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="old-instance",
            database_name="myapp",
        )
        current = make_database_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="new-instance",
            database_name="myapp",
        )

        result = await harness.invoke_update(
            Database,
            name="myapp-db",
            config=current,
            previous_config=previous,
        )

        assert result.success
        mock_sqladmin_service.databases().delete.assert_called()
        mock_sqladmin_service.databases().insert.assert_called()

    async def test_update_rejects_database_name_change(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_update rejects database_name changes."""
        previous = make_database_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="db",
            database_name="myapp-a",
        )
        current = make_database_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="db",
            database_name="myapp-b",
        )

        result = await harness.invoke_update(
            Database,
            name="myapp-db",
            config=current,
            previous_config=previous,
        )

        assert result.failed
        assert "database_name" in str(result.error)

    async def test_delete_success(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_delete removes database."""
        config = make_database_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="db",
            database_name="myapp",
        )

        result = await harness.invoke_delete(Database, name="myapp-db", config=config)

        assert result.success
        mock_sqladmin_service.databases().delete.assert_called()

    async def test_delete_idempotent(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_delete succeeds when database doesn't exist."""
        mock_resp = mocker.MagicMock()
        mock_resp.status = 404
        mock_sqladmin_service.databases().delete().execute.side_effect = HttpError(mock_resp, b"not found")

        config = make_database_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="db",
            database_name="myapp",
        )

        result = await harness.invoke_delete(Database, name="myapp-db", config=config)

        assert result.success


class TestUser:
    """Tests for CloudSQL User resource."""

    async def test_create_user_success(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_create creates user in instance."""
        mock_sqladmin_service.users().list().execute.return_value = {"items": []}
        mock_sqladmin_service.users().insert().execute.return_value = {"name": "operation-123"}

        config = make_user_config(
            mocker,
            project_id="test-project",
            credentials=sample_credentials,
            instance_name="test-db",
            username="appuser",
            password="secret123",
        )

        result = await harness.invoke_create(User, name="appuser", config=config)

        assert result.success
        assert result.outputs is not None
        assert result.outputs.username == "appuser"
        assert result.outputs.instance_name == "test-db"
        assert result.outputs.host == "%"

        mock_sqladmin_service.users().insert.assert_called()

    async def test_create_user_idempotent(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_create handles existing user (idempotent retry)."""
        mock_sqladmin_service.users().list().execute.return_value = {"items": [{"name": "appuser", "host": "%"}]}

        config = make_user_config(
            mocker,
            project_id="test-project",
            credentials=sample_credentials,
            instance_name="test-db",
            username="appuser",
            password="secret123",
        )

        result = await harness.invoke_create(User, name="appuser", config=config)

        assert result.success
        assert result.outputs is not None
        assert result.outputs.username == "appuser"

    async def test_update_password_change(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_update updates password when changed."""
        mock_sqladmin_service.users().list().execute.return_value = {"items": [{"name": "appuser", "host": "%"}]}
        mock_sqladmin_service.users().update().execute.return_value = {"name": "operation-123"}

        previous = make_user_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="db",
            username="appuser",
            password="old-password",
        )
        current = make_user_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="db",
            username="appuser",
            password="new-password",
        )

        result = await harness.invoke_update(
            User,
            name="appuser",
            config=current,
            previous_config=previous,
        )

        assert result.success
        mock_sqladmin_service.users().update.assert_called()

    async def test_update_instance_change_triggers_replacement(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_update replaces user when instance changes (delete + create)."""
        mock_sqladmin_service.users().list().execute.return_value = {"items": []}

        previous = make_user_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="old-instance",
            username="appuser",
            password="secret123",
        )
        current = make_user_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="new-instance",
            username="appuser",
            password="secret123",
        )

        result = await harness.invoke_update(
            User,
            name="appuser",
            config=current,
            previous_config=previous,
        )

        assert result.success
        mock_sqladmin_service.users().delete.assert_called()
        mock_sqladmin_service.users().insert.assert_called()

    async def test_update_rejects_username_change(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_update rejects username changes."""
        previous = make_user_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="db",
            username="user-a",
            password="secret123",
        )
        current = make_user_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="db",
            username="user-b",
            password="secret123",
        )

        result = await harness.invoke_update(
            User,
            name="appuser",
            config=current,
            previous_config=previous,
        )

        assert result.failed
        assert "username" in str(result.error)

    async def test_delete_success(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_delete removes user."""
        config = make_user_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="db",
            username="appuser",
            password="secret123",
        )

        result = await harness.invoke_delete(User, name="appuser", config=config)

        assert result.success
        mock_sqladmin_service.users().delete.assert_called()

    async def test_delete_idempotent(
        self,
        harness: ProviderHarness,
        mock_sqladmin_service: Any,
        sample_credentials: dict,
        mocker: MockerFixture,
    ) -> None:
        """on_delete succeeds when user doesn't exist."""
        mock_resp = mocker.MagicMock()
        mock_resp.status = 404
        mock_sqladmin_service.users().delete().execute.side_effect = HttpError(mock_resp, b"not found")

        config = make_user_config(
            mocker,
            project_id="proj",
            credentials=sample_credentials,
            instance_name="db",
            username="appuser",
            password="secret123",
        )

        result = await harness.invoke_delete(User, name="appuser", config=config)

        assert result.success
