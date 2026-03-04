"""Agno AsyncPostgresDb resource for agent storage."""

from __future__ import annotations

from typing import ClassVar
from urllib.parse import quote, urlparse, urlunparse

from agno.db.postgres import AsyncPostgresDb
from pragma_sdk import Config, Field, Outputs
from pydantic import computed_field, model_validator

from agno_provider.resources.base import AgnoResource, AgnoSpec


class DbPostgresSpec(AgnoSpec):
    """Specification for AsyncPostgresDb runtime construction.

    Contains all configuration needed to reconstruct an AsyncPostgresDb
    at runtime.

    Attributes:
        connection_url: Full connection URL.
        host: Database host.
        port: Database port.
        database: Database name.
        user: Database username.
        password: Database password.
        db_schema: Database schema for Agno tables.
        session_table: Custom table name for sessions.
        memory_table: Custom table name for memories.
    """

    connection_url: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None
    user: str | None = None
    password: str | None = None
    db_schema: str = "ai"
    session_table: str | None = None
    memory_table: str | None = None

    @computed_field
    @property
    def db_url(self) -> str:
        """Build the async PostgreSQL connection URL from spec fields.

        Returns:
            Connection URL with postgresql+psycopg_async:// scheme.

        Raises:
            ValueError: If required fields are missing when not using connection_url.
        """
        if self.connection_url is not None:
            url = self.connection_url

            if self.user is not None and self.password is not None:
                url = self._inject_credentials(url, self.user, self.password)

            return url.replace("postgresql://", "postgresql+psycopg_async://")

        if self.user is None or self.password is None:
            msg = "user and password are required when not using connection_url"
            raise ValueError(msg)

        if self.host is None or self.database is None:
            msg = "host and database are required when not using connection_url"
            raise ValueError(msg)

        encoded_user = quote(self.user, safe="")
        encoded_pass = quote(self.password, safe="")
        port = self.port or 5432

        return f"postgresql+psycopg_async://{encoded_user}:{encoded_pass}@{self.host}:{port}/{self.database}"

    @staticmethod
    def _inject_credentials(url: str, username: str, password: str) -> str:
        """Inject credentials into URL if not already present.

        Args:
            url: The connection URL.
            username: Username to inject.
            password: Password to inject.

        Returns:
            URL with credentials injected and encoded.
        """
        parsed = urlparse(url)

        if parsed.username:
            return url

        encoded_user = quote(username, safe="")
        encoded_pass = quote(password, safe="")
        netloc = f"{encoded_user}:{encoded_pass}@{parsed.hostname}"

        if parsed.port:
            netloc += f":{parsed.port}"

        return urlunparse(parsed._replace(netloc=netloc))


class DbPostgresConfig(Config):
    """Configuration for Agno AsyncPostgresDb.

    Supports two configuration patterns:
    1. Full URL: connection_url (credentials may be in URL or provided separately)
    2. Separate fields: host + database + username + password

    Attributes:
        connection_url: Full PostgreSQL connection URL.
        host: Database host (IP or hostname).
        port: Database port. Defaults to 5432.
        database: Database name.
        username: Database username (required for separate fields, optional for URL).
        password: Database password (required for separate fields, optional for URL).
        db_schema: Database schema for Agno tables. Defaults to "ai".
        session_table: Custom table name for sessions.
        memory_table: Custom table name for memories.
    """

    connection_url: Field[str] | None = None

    host: Field[str] | None = None
    port: Field[int] = 5432
    database: Field[str] | None = None

    username: Field[str] | None = None
    password: Field[str] | None = None

    db_schema: Field[str] = "ai"
    session_table: Field[str] | None = None
    memory_table: Field[str] | None = None

    @model_validator(mode="after")
    def validate_connection_config(self) -> DbPostgresConfig:
        """Validate that either connection_url or host+database is provided.

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If neither connection_url nor host+database is provided,
                or if host+database is used without username+password.
        """
        has_url = self.connection_url is not None
        has_fields = self.host is not None and self.database is not None

        if not has_url and not has_fields:
            msg = "Either connection_url or (host + database) must be provided"
            raise ValueError(msg)

        if has_fields and (self.username is None or self.password is None):
            msg = "username and password are required when using host + database"
            raise ValueError(msg)

        return self


class DbPostgresOutputs(Outputs):
    """Outputs from Agno AsyncPostgresDb resource.

    Attributes:
        spec: The database specification for runtime reconstruction.
    """

    spec: DbPostgresSpec


class DbPostgres(AgnoResource[DbPostgresConfig, DbPostgresOutputs, DbPostgresSpec]):
    """Agno AsyncPostgresDb resource for agent storage.

    Wraps Agno's AsyncPostgresDb to provide PostgreSQL storage for agents.
    The database instance is created via from_spec() at runtime.

    Runtime reconstruction via spec:
        db = DbPostgres.from_spec(spec)
        agent = Agent(db=db, ...)

    Lifecycle:
        - on_create: Return serializable metadata (no actual DB setup)
        - on_update: Return serializable metadata
        - on_delete: No-op (stateless wrapper)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "db/postgres"
    description = "Configures a PostgreSQL database for agent storage."

    @staticmethod
    def from_spec(spec: DbPostgresSpec) -> AsyncPostgresDb:
        """Factory: construct Agno AsyncPostgresDb from spec.

        Args:
            spec: The database specification.

        Returns:
            Configured AsyncPostgresDb instance.
        """
        kwargs: dict = {
            "db_url": spec.db_url,
            "db_schema": spec.db_schema,
        }

        if spec.session_table:
            kwargs["session_table"] = spec.session_table

        if spec.memory_table:
            kwargs["memory_table"] = spec.memory_table

        return AsyncPostgresDb(**kwargs)

    def _build_spec(self) -> DbPostgresSpec:
        """Build the database specification for runtime reconstruction.

        Returns:
            DbPostgresSpec with configuration.
        """
        connection_url = None
        password = None
        host = None
        port = None
        database = None
        user = None

        if self.config.connection_url is not None:
            connection_url = str(self.config.connection_url)

            if self.config.username is not None:
                user = str(self.config.username)

            if self.config.password is not None:
                password = str(self.config.password)

        else:
            host = str(self.config.host)
            port = self.config.port
            database = str(self.config.database)
            user = str(self.config.username)
            password = str(self.config.password)

        return DbPostgresSpec(
            connection_url=connection_url,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            db_schema=self.config.db_schema,
            session_table=self.config.session_table,
            memory_table=self.config.memory_table,
        )

    def _build_outputs(self) -> DbPostgresOutputs:
        """Build outputs from current config.

        Returns:
            DbPostgresOutputs with spec.
        """
        return DbPostgresOutputs(spec=self._build_spec())

    def _build_url(self) -> str:
        """Build the async PostgreSQL connection URL.

        Returns:
            Connection URL with postgresql+psycopg_async:// scheme.
        """
        if self.config.connection_url is not None:
            url = str(self.config.connection_url)

            if self.config.username is not None and self.config.password is not None:
                url = self._inject_credentials(url, str(self.config.username), str(self.config.password))

            return url.replace("postgresql://", "postgresql+psycopg_async://")

        encoded_user = quote(str(self.config.username), safe="")
        encoded_pass = quote(str(self.config.password), safe="")

        return (
            f"postgresql+psycopg_async://"
            f"{encoded_user}:{encoded_pass}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )

    def _inject_credentials(self, url: str, username: str, password: str) -> str:
        """Inject credentials into URL if not already present.

        Credentials are URL-encoded to handle special characters like @, :, /, %.

        Args:
            url: The connection URL (must be postgresql:// scheme).
            username: Username to inject.
            password: Password to inject.

        Returns:
            URL with credentials injected and encoded.
        """
        parsed = urlparse(url)

        if parsed.username:
            return url

        encoded_user = quote(username, safe="")
        encoded_pass = quote(password, safe="")
        netloc = f"{encoded_user}:{encoded_pass}@{parsed.hostname}"

        if parsed.port:
            netloc += f":{parsed.port}"

        return urlunparse(parsed._replace(netloc=netloc))

    async def on_create(self) -> DbPostgresOutputs:
        """Create resource and return serializable outputs.

        Returns:
            DbPostgresOutputs with configuration metadata.
        """
        return self._build_outputs()

    async def on_update(self, previous_config: DbPostgresConfig) -> DbPostgresOutputs:  # noqa: ARG002
        """Update resource and return serializable outputs.

        Returns:
            DbPostgresOutputs with updated configuration metadata.
        """
        return self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
