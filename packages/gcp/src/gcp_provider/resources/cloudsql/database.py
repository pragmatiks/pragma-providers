"""GCP Cloud SQL database resource."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field
from pragma_sdk import Config, Dependency, Outputs, Resource

from gcp_provider.resources.cloudsql.database_instance import DatabaseInstance
from gcp_provider.resources.cloudsql.helpers import (
    connection_info,
    execute,
    extract_ips,
    get_credentials,
    get_sqladmin_service,
)


class DatabaseConfig(Config):
    """Configuration for a Cloud SQL database.

    Attributes:
        instance: The Cloud SQL instance that hosts this database.
        database_name: Name of the database to create.
    """

    instance: Dependency[DatabaseInstance]
    database_name: str = Field(json_schema_extra={"immutable": True})


class DatabaseOutputs(Outputs):
    """Outputs from Cloud SQL database creation.

    Attributes:
        database_name: Name of the created database.
        instance_name: Name of the hosting instance.
        url: Connection URL format (without credentials).
    """

    database_name: str
    instance_name: str
    url: str


class Database(Resource[DatabaseConfig, DatabaseOutputs]):
    """GCP Cloud SQL database resource.

    Creates and manages databases within a Cloud SQL instance.
    Changing the instance triggers replacement (delete from old, create in new).
    """

    provider: ClassVar[str] = "gcp"
    resource: ClassVar[str] = "cloudsql/database"

    async def on_create(self) -> DatabaseOutputs:
        """Create database in the Cloud SQL instance.

        Idempotent: If database already exists, returns its current state.
        """
        inst = self.config.instance.config
        service = get_sqladmin_service(get_credentials(inst.credentials))

        await execute(
            service.databases().insert(
                project=inst.project_id,
                instance=inst.instance_name,
                body={
                    "name": self.config.database_name,
                    "project": inst.project_id,
                    "instance": inst.instance_name,
                },
            ),
            ignore_exists=True,
        )

        return await self._build_outputs(inst, service)

    async def on_update(self, previous_config: DatabaseConfig) -> DatabaseOutputs:
        """Handle database updates.

        If instance changed, delete from old instance and create in new one.
        database_name cannot be changed (truly immutable).
        """
        if previous_config.database_name != self.config.database_name:
            msg = "Cannot change database_name; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.instance != self.config.instance:
            await self._delete(previous_config)
            return await self.on_create()

        inst = self.config.instance.config
        service = get_sqladmin_service(get_credentials(inst.credentials))

        return await self._build_outputs(inst, service)

    async def on_delete(self) -> None:
        """Delete database. Idempotent: succeeds if database doesn't exist."""
        await self._delete(self.config)

    async def _delete(self, config: DatabaseConfig) -> None:
        """Delete database from instance. Idempotent: succeeds if not found."""
        inst = config.instance.config
        service = get_sqladmin_service(get_credentials(inst.credentials))

        await execute(
            service.databases().delete(
                project=inst.project_id,
                instance=inst.instance_name,
                database=config.database_name,
            ),
            ignore_404=True,
        )

    async def _build_outputs(self, inst: Any, service: Any) -> DatabaseOutputs:
        """Fetch instance info and build outputs."""
        instance = await execute(
            service.instances().get(
                project=inst.project_id,
                instance=inst.instance_name,
            )
        )

        public_ip, private_ip = extract_ips(instance)
        db_type, port = connection_info(instance.get("databaseVersion", "POSTGRES_15"))
        host = public_ip or private_ip or f"{inst.project_id}:{instance.get('region')}:{inst.instance_name}"

        return DatabaseOutputs(
            database_name=self.config.database_name,
            instance_name=inst.instance_name,
            url=f"{db_type}://USER:PASSWORD@{host}:{port}/{self.config.database_name}",
        )
