"""GCP Cloud SQL database resource."""

from __future__ import annotations

from typing import Any

from pragma_sdk import Config, ImmutableDependency, ImmutableField, Outputs, Resource

from gcp_provider.resources.cloudsql.database_instance import (
    DatabaseInstance,
    DatabaseInstanceConfig,
    resolve_instance_config,
)
from gcp_provider.resources.cloudsql.helpers import (
    connection_info,
    execute,
    extract_ips,
    get_credentials,
    get_sqladmin_service,
    run_instance_operation,
)
from gcp_provider.resources.polling import compute_operation_deadline


class DatabaseConfig(Config):
    """Configuration for a Cloud SQL database.

    Attributes:
        instance: The Cloud SQL instance that hosts this database.
        database_name: Name of the database to create.
    """

    instance: ImmutableDependency[DatabaseInstance]
    database_name: ImmutableField[str]


class DatabaseOutputs(Outputs):
    """Outputs from Cloud SQL database creation.

    Attributes:
        database_name: Name of the created database.
        instance_name: Name of the hosting instance.
        host: Database host (IP address or connection name).
        port: Database port (5432 for postgres, 3306 for mysql).
        url: Connection URL format (without credentials).
    """

    database_name: str
    instance_name: str
    host: str
    port: int
    url: str


class Database(Resource[DatabaseConfig, DatabaseOutputs]):
    """GCP Cloud SQL database resource.

    Creates and manages databases within a Cloud SQL instance. Requires a
    dependency on a ``gcp/cloudsql/database_instance`` resource, from which
    it inherits credentials and connection details.

    Lifecycle:
        - on_create: Creates the database in the target instance. Idempotent --
          succeeds if the database already exists.
        - on_observe: Reads the database by ``database_name``.
        - on_update: Returns current state; ``instance`` and ``database_name`` are immutable.
        - on_delete: Drops the database from the instance. Idempotent --
          succeeds silently if the database does not exist.

    Outputs include a connection URL in the format
    ``{db_type}://host:port/database_name`` (without credentials).

    Example::

        resources:
          - name: app-database
            provider: gcp
            type: cloudsql/database
            config:
              instance:
                provider: gcp
                resource: cloudsql/database_instance
                name: prod-db-instance
              database_name: myapp
    """

    async def on_create(self) -> DatabaseOutputs:
        """Create the database in the Cloud SQL instance and wait until it exists.

        Idempotent: If the database already exists, returns its current state.

        Returns:
            DatabaseOutputs with database details.

        Raises:
            RuntimeError: If Cloud SQL refuses the write because another operation runs on the instance,
                or the operation finished with errors.
            TimeoutError: If a pending operation, or the write's operation, is not done by the operation deadline.
        """
        instance_config = await resolve_instance_config(self.config.instance)
        service = await get_sqladmin_service(get_credentials(instance_config.credentials))

        await run_instance_operation(
            service,
            instance_config,
            self.name,
            lambda: execute(
                service.databases().insert(
                    project=instance_config.project_id,
                    instance=instance_config.instance_name,
                    body={
                        "name": self.config.database_name,
                        "project": instance_config.project_id,
                        "instance": instance_config.instance_name,
                    },
                ),
                ignore_exists=True,
            ),
            compute_operation_deadline(),
        )

        return await self.fetch_outputs(instance_config, service)

    async def on_observe(self) -> DatabaseOutputs | None:
        """Read the database named ``database_name`` in the resolved instance.

        Returns:
            DatabaseOutputs for the database, or None if it does not exist.
        """
        instance_config = await resolve_instance_config(self.config.instance)
        service = await get_sqladmin_service(get_credentials(instance_config.credentials))

        database = await execute(
            service.databases().get(
                project=instance_config.project_id,
                instance=instance_config.instance_name,
                database=self.config.database_name,
            ),
            ignore_404=True,
        )

        if database is None:
            return None

        return await self.fetch_outputs(instance_config, service)

    async def on_update(self, previous_config: DatabaseConfig | None) -> DatabaseOutputs:
        """Return the current state of the database.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            DatabaseOutputs with database details.
        """
        instance_config = await resolve_instance_config(self.config.instance)
        service = await get_sqladmin_service(get_credentials(instance_config.credentials))

        return await self.fetch_outputs(instance_config, service)

    async def on_delete(self) -> None:
        """Drop the database and wait until it is gone.

        Idempotent: Succeeds if the database does not exist.

        Raises:
            RuntimeError: If Cloud SQL refuses the write because another operation runs on the instance,
                or the operation finished with errors.
            TimeoutError: If a pending operation, or the write's operation, is not done by the operation deadline.
        """
        instance_config = await resolve_instance_config(self.config.instance)
        service = await get_sqladmin_service(get_credentials(instance_config.credentials))

        await run_instance_operation(
            service,
            instance_config,
            self.name,
            lambda: execute(
                service.databases().delete(
                    project=instance_config.project_id,
                    instance=instance_config.instance_name,
                    database=self.config.database_name,
                ),
                ignore_404=True,
            ),
            compute_operation_deadline(),
        )

    async def fetch_outputs(self, instance_config: DatabaseInstanceConfig, service: Any) -> DatabaseOutputs:
        """Fetch instance info and build outputs.

        Args:
            instance_config: Configuration of the hosting Cloud SQL instance.
            service: Cloud SQL Admin API service.

        Returns:
            DatabaseOutputs with connection details.
        """
        instance = await execute(
            service.instances().get(
                project=instance_config.project_id,
                instance=instance_config.instance_name,
            )
        )

        public_ip, private_ip = extract_ips(instance)
        db_type, db_port = connection_info(instance.get("databaseVersion", "POSTGRES_15"))
        host = (
            public_ip
            or private_ip
            or f"{instance_config.project_id}:{instance.get('region')}:{instance_config.instance_name}"
        )

        return DatabaseOutputs(
            database_name=self.config.database_name,
            instance_name=instance_config.instance_name,
            host=host,
            port=int(db_port),
            url=f"{db_type}://{host}:{db_port}/{self.config.database_name}",
        )
