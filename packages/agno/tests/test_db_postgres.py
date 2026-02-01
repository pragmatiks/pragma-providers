"""Tests for Agno db/postgres resource."""

from __future__ import annotations

import pytest
from agno.db.postgres import AsyncPostgresDb
from pragma_sdk.provider import ProviderHarness

from agno_provider import (
    DbPostgres,
    DbPostgresConfig,
    DbPostgresOutputs,
)


@pytest.fixture
def harness() -> ProviderHarness:
    """Test harness for invoking lifecycle methods."""
    return ProviderHarness()


def test_config_valid_connection_url_only() -> None:
    """connection_url alone is valid (credentials in URL)."""
    config = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/db",
    )

    assert config.connection_url == "postgresql://user:pass@localhost:5432/db"


def test_config_valid_connection_url_with_credentials() -> None:
    """connection_url with separate credentials is valid."""
    config = DbPostgresConfig(
        connection_url="postgresql://localhost:5432/db",
        username="user",
        password="pass",
    )

    assert config.connection_url == "postgresql://localhost:5432/db"
    assert config.username == "user"
    assert config.password == "pass"


def test_config_valid_separate_fields() -> None:
    """Host + database + username + password is valid."""
    config = DbPostgresConfig(
        host="localhost",
        database="mydb",
        username="user",
        password="pass",
    )

    assert config.host == "localhost"
    assert config.database == "mydb"
    assert config.username == "user"
    assert config.password == "pass"
    assert config.port == 5432


def test_config_invalid_no_connection_info() -> None:
    """Missing both connection_url and host+database raises error."""
    with pytest.raises(ValueError, match="Either connection_url or"):
        DbPostgresConfig(
            username="user",
            password="pass",
        )


def test_config_invalid_host_without_database() -> None:
    """Host without database raises error."""
    with pytest.raises(ValueError, match="Either connection_url or"):
        DbPostgresConfig(
            host="localhost",
            username="user",
            password="pass",
        )


def test_config_invalid_separate_fields_missing_credentials() -> None:
    """Host + database without credentials raises error."""
    with pytest.raises(ValueError, match="username and password are required"):
        DbPostgresConfig(
            host="localhost",
            database="mydb",
        )


def test_config_defaults() -> None:
    """Config has correct default values."""
    config = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/db",
    )

    assert config.port == 5432
    assert config.db_schema == "ai"
    assert config.session_table is None
    assert config.memory_table is None


def test_url_building_connection_url_converts_scheme(harness: ProviderHarness) -> None:
    """connection_url scheme is converted to async driver."""
    config = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/db",
    )

    resource = DbPostgres(name="test-db", config=config)

    url = resource._build_url()

    assert url == "postgresql+psycopg_async://user:pass@localhost:5432/db"


def test_url_building_connection_url_injects_credentials(harness: ProviderHarness) -> None:
    """Separate credentials are injected into connection_url."""
    config = DbPostgresConfig(
        connection_url="postgresql://localhost:5432/db",
        username="injected",
        password="secret",
    )

    resource = DbPostgres(name="test-db", config=config)

    url = resource._build_url()

    assert url == "postgresql+psycopg_async://injected:secret@localhost:5432/db"


def test_url_building_connection_url_preserves_existing_credentials(harness: ProviderHarness) -> None:
    """Existing credentials in URL are preserved over separate ones."""
    config = DbPostgresConfig(
        connection_url="postgresql://existing:creds@localhost:5432/db",
        username="ignored",
        password="ignored",
    )

    resource = DbPostgres(name="test-db", config=config)

    url = resource._build_url()

    assert url == "postgresql+psycopg_async://existing:creds@localhost:5432/db"


def test_url_building_separate_fields_builds_url(harness: ProviderHarness) -> None:
    """Separate fields are assembled into connection URL."""
    config = DbPostgresConfig(
        host="db.example.com",
        port=5433,
        database="myapp",
        username="appuser",
        password="secret123",
    )

    resource = DbPostgres(name="test-db", config=config)

    url = resource._build_url()

    assert url == "postgresql+psycopg_async://appuser:secret123@db.example.com:5433/myapp"


def test_url_building_separate_fields_uses_default_port(harness: ProviderHarness) -> None:
    """Default port 5432 is used when not specified."""
    config = DbPostgresConfig(
        host="localhost",
        database="test",
        username="user",
        password="pass",
    )

    resource = DbPostgres(name="test-db", config=config)

    url = resource._build_url()

    assert url == "postgresql+psycopg_async://user:pass@localhost:5432/test"


def test_db_method_returns_async_postgres_instance(harness: ProviderHarness) -> None:
    """db() returns configured AsyncPostgresDb instance."""
    config = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/db",
    )

    resource = DbPostgres(name="test-db", config=config)

    db = resource.db()

    assert isinstance(db, AsyncPostgresDb)


def test_db_method_passes_schema(harness: ProviderHarness) -> None:
    """db() passes schema to AsyncPostgresDb."""
    config = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/db",
        db_schema="custom_schema",
    )

    resource = DbPostgres(name="test-db", config=config)

    db = resource.db()

    assert db.db_schema == "custom_schema"


def test_db_method_passes_table_names(harness: ProviderHarness) -> None:
    """db() passes custom table names to AsyncPostgresDb."""
    config = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/db",
        session_table="my_sessions",
        memory_table="my_memories",
    )

    resource = DbPostgres(name="test-db", config=config)

    db = resource.db()

    assert db.session_table_name == "my_sessions"
    assert db.memory_table_name == "my_memories"


async def test_lifecycle_create_returns_outputs(harness: ProviderHarness) -> None:
    """on_create returns serializable outputs."""
    config = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/db",
        db_schema="test_schema",
    )

    result = await harness.invoke_create(DbPostgres, name="test-db", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.ready is True
    assert result.outputs.db_schema == "test_schema"


async def test_lifecycle_update_returns_outputs(harness: ProviderHarness) -> None:
    """on_update returns serializable outputs with updated schema."""
    previous = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/db",
        db_schema="old_schema",
    )
    current = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/db",
        db_schema="new_schema",
    )

    result = await harness.invoke_update(
        DbPostgres,
        name="test-db",
        config=current,
        previous_config=previous,
        current_outputs=DbPostgresOutputs(ready=True, db_schema="old_schema"),
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.db_schema == "new_schema"


async def test_lifecycle_delete_success(harness: ProviderHarness) -> None:
    """on_delete completes without error (stateless resource)."""
    config = DbPostgresConfig(
        connection_url="postgresql://user:pass@localhost:5432/db",
    )

    result = await harness.invoke_delete(DbPostgres, name="test-db", config=config)

    assert result.success


def test_url_building_encodes_special_characters_in_password(harness: ProviderHarness) -> None:
    """Special characters in password are URL-encoded."""
    config = DbPostgresConfig(
        host="localhost",
        database="test",
        username="user",
        password="p@ss:word/with%special",
    )

    resource = DbPostgres(name="test-db", config=config)

    url = resource._build_url()

    assert url == "postgresql+psycopg_async://user:p%40ss%3Aword%2Fwith%25special@localhost:5432/test"


def test_url_building_injects_encoded_credentials(harness: ProviderHarness) -> None:
    """Injected credentials are URL-encoded."""
    config = DbPostgresConfig(
        connection_url="postgresql://localhost:5432/db",
        username="user@domain",
        password="pass:word",
    )

    resource = DbPostgres(name="test-db", config=config)

    url = resource._build_url()

    assert url == "postgresql+psycopg_async://user%40domain:pass%3Aword@localhost:5432/db"


def test_resource_metadata_provider_name() -> None:
    """Resource has correct provider name."""
    assert DbPostgres.provider == "agno"


def test_resource_metadata_resource_type() -> None:
    """Resource has correct resource type."""
    assert DbPostgres.resource == "db/postgres"


def test_outputs_are_serializable() -> None:
    """Outputs contain only serializable data."""
    outputs = DbPostgresOutputs(ready=True, db_schema="ai")

    assert outputs.ready is True
    assert outputs.db_schema == "ai"

    serialized = outputs.model_dump_json()
    assert "ready" in serialized
    assert "db_schema" in serialized
