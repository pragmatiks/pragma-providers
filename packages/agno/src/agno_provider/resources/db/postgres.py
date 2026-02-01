"""Agno AsyncPostgresDb resource for agent storage."""

from __future__ import annotations

from typing import ClassVar
from urllib.parse import quote, urlparse, urlunparse

from agno.db.postgres import AsyncPostgresDb
from pragma_sdk import Config, Field, Outputs, Resource
from pydantic import model_validator


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
    port: int = 5432
    database: Field[str] | None = None

    username: Field[str] | None = None
    password: Field[str] | None = None

    db_schema: str = "ai"
    session_table: str | None = None
    memory_table: str | None = None

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
        ready: Whether the database connection is configured.
        db_schema: The configured database schema.
    """

    ready: bool
    db_schema: str


class DbPostgres(Resource[DbPostgresConfig, DbPostgresOutputs]):
    """Agno AsyncPostgresDb resource for agent storage.

    Wraps Agno's AsyncPostgresDb to provide PostgreSQL storage for agents.
    Dependent resources receive this resource and call db() to get the
    configured AsyncPostgresDb instance.

    Usage by dependent resources:
        storage: Dependency[DbPostgres]

        async def on_create(self):
            db = self.storage.db()
            agent = Agent(db=db, ...)

    Lifecycle:
        - on_create: Return serializable metadata (no actual DB setup)
        - on_update: Return serializable metadata
        - on_delete: No-op (stateless wrapper)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "db/postgres"

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

    def db(self) -> AsyncPostgresDb:
        """Return configured AsyncPostgresDb instance.

        Called by dependent resources (e.g., agno/agent) that need
        the database instance for agent storage.

        Returns:
            Configured AsyncPostgresDb ready for use.
        """
        url = self._build_url()

        kwargs: dict = {
            "db_url": url,
            "db_schema": self.config.db_schema,
        }

        if self.config.session_table:
            kwargs["session_table"] = self.config.session_table

        if self.config.memory_table:
            kwargs["memory_table"] = self.config.memory_table

        return AsyncPostgresDb(**kwargs)

    async def on_create(self) -> DbPostgresOutputs:
        """Create resource and return serializable outputs.

        Returns:
            DbPostgresOutputs with configuration metadata.
        """
        return DbPostgresOutputs(ready=True, db_schema=self.config.db_schema)

    async def on_update(self, previous_config: DbPostgresConfig) -> DbPostgresOutputs:  # noqa: ARG002
        """Update resource and return serializable outputs.

        Returns:
            DbPostgresOutputs with updated configuration metadata.
        """
        return DbPostgresOutputs(ready=True, db_schema=self.config.db_schema)

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
