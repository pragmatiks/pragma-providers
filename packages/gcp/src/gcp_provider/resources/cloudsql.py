"""GCP Cloud SQL database instance resource."""

from __future__ import annotations

import asyncio
import json
import secrets
import string
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, ClassVar, Literal

from google.cloud.logging_v2 import Client as LoggingClient
from google.oauth2 import service_account
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from pydantic import Field, field_validator
from pragma_sdk import Config, HealthStatus, LogEntry, Outputs, Resource


class CloudSQLConfig(Config):
    """Configuration for a Cloud SQL instance.

    Attributes:
        project_id: GCP project ID where the instance will be created.
        credentials: GCP service account credentials JSON object or string.
        region: GCP region (e.g., europe-west4).
        instance_name: Name of the Cloud SQL instance (must be unique per project).
        database_version: Database version (e.g., POSTGRES_15, POSTGRES_14, MYSQL_8_0).
        tier: Machine tier (e.g., db-f1-micro, db-custom-1-3840).
        database_name: Name of the database to create.
        availability_type: ZONAL (single zone) or REGIONAL (high availability).
        backup_enabled: Whether automatic backups are enabled.
        deletion_protection: Whether deletion protection is enabled.
        authorized_networks: List of CIDR ranges to allow connections from.
        enable_public_ip: Whether to assign a public IP address.
    """

    project_id: str
    credentials: dict[str, Any] | str
    region: str
    instance_name: str
    database_version: str = "POSTGRES_15"
    tier: str = "db-f1-micro"
    database_name: str = "app"
    availability_type: Literal["ZONAL", "REGIONAL"] = "ZONAL"
    backup_enabled: bool = True
    deletion_protection: bool = False
    authorized_networks: list[str] = Field(default_factory=list)
    enable_public_ip: bool = True

    @field_validator("instance_name")
    @classmethod
    def validate_instance_name(cls, v: str) -> str:
        """Validate instance name follows Cloud SQL naming rules."""
        if len(v) < 1 or len(v) > 98:
            msg = "Instance name must be 1-98 characters"
            raise ValueError(msg)

        if not v[0].isalpha():
            msg = "Instance name must start with a letter"
            raise ValueError(msg)

        allowed = set(string.ascii_lowercase + string.digits + "-")
        if not all(c in allowed for c in v.lower()):
            msg = "Instance name can only contain letters, numbers, and hyphens"
            raise ValueError(msg)

        return v

    @field_validator("database_version")
    @classmethod
    def validate_database_version(cls, v: str) -> str:
        """Validate database version is supported."""
        supported_prefixes = ("POSTGRES_", "MYSQL_", "SQLSERVER_")
        if not any(v.startswith(prefix) for prefix in supported_prefixes):
            msg = f"Unsupported database version: {v}. Must start with POSTGRES_, MYSQL_, or SQLSERVER_"
            raise ValueError(msg)

        return v


class CloudSQLOutputs(Outputs):
    """Outputs from Cloud SQL instance creation.

    Attributes:
        connection_name: Cloud SQL connection name (project:region:instance).
        public_ip: Public IP address (if enabled).
        private_ip: Private IP address (if enabled).
        database_name: Name of the created database.
        url: Connection URL format (without credentials).
        ready: Whether the instance is running and accessible.
        console_url: URL to view instance in GCP Console.
        logs_url: URL to view instance logs in Cloud Logging.
    """

    connection_name: str
    public_ip: str | None
    private_ip: str | None
    database_name: str
    url: str
    ready: bool
    console_url: str
    logs_url: str


_POLL_INTERVAL_SECONDS = 30
_MAX_POLL_ATTEMPTS = 30


def _generate_root_password() -> str:
    """Generate a secure random password for the root user."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(24))


class CloudSQL(Resource[CloudSQLConfig, CloudSQLOutputs]):
    """GCP Cloud SQL database instance resource.

    Creates and manages Cloud SQL instances (PostgreSQL, MySQL, SQL Server)
    using user-provided service account credentials (multi-tenant SaaS pattern).

    Lifecycle:
        - on_create: Creates instance, waits for RUNNABLE state, creates database
        - on_update: Limited updates (some require recreation)
        - on_delete: Deletes instance (respects deletion_protection)
    """

    provider: ClassVar[str] = "gcp"
    resource: ClassVar[str] = "cloudsql"

    def _get_credentials(self) -> service_account.Credentials:
        """Get credentials from config."""
        creds_data = self.config.credentials

        if isinstance(creds_data, str):
            creds_data = json.loads(creds_data)

        return service_account.Credentials.from_service_account_info(creds_data)

    def _get_service(self) -> Any:
        """Get Cloud SQL Admin API service."""
        return discovery.build("sqladmin", "v1", credentials=self._get_credentials())

    def _connection_name(self) -> str:
        """Build Cloud SQL connection name."""
        return f"{self.config.project_id}:{self.config.region}:{self.config.instance_name}"

    def _build_outputs(self, instance: dict, database_name: str) -> CloudSQLOutputs:
        """Build outputs from instance dict."""
        project = self.config.project_id
        name = self.config.instance_name

        public_ip = None
        private_ip = None
        for ip_addr in instance.get("ipAddresses", []):
            if ip_addr.get("type") == "PRIMARY":
                public_ip = ip_addr.get("ipAddress")
            elif ip_addr.get("type") == "PRIVATE":
                private_ip = ip_addr.get("ipAddress")

        db_type = "postgresql" if self.config.database_version.startswith("POSTGRES") else "mysql"
        port = "5432" if db_type == "postgresql" else "3306"
        host = public_ip or private_ip or self._connection_name()
        url = f"{db_type}://USER:PASSWORD@{host}:{port}/{database_name}"

        console_url = f"https://console.cloud.google.com/sql/instances/{name}/overview?project={project}"
        logs_url = (
            f"https://console.cloud.google.com/logs/query;"
            f"query=resource.type%3D%22cloudsql_database%22%0A"
            f"resource.labels.database_id%3D%22{project}%3A{name}%22"
            f"?project={project}"
        )

        return CloudSQLOutputs(
            connection_name=self._connection_name(),
            public_ip=public_ip,
            private_ip=private_ip,
            database_name=database_name,
            url=url,
            ready=instance.get("state") == "RUNNABLE",
            console_url=console_url,
            logs_url=logs_url,
        )

    async def _run_in_executor(self, func: Any) -> Any:
        """Run a blocking function in the default executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)

    async def _get_instance(self, service: Any) -> dict | None:
        """Get instance if it exists, None otherwise."""
        try:
            request = service.instances().get(project=self.config.project_id, instance=self.config.instance_name)
            return await self._run_in_executor(request.execute)
        except HttpError as e:
            if e.resp.status == 404:
                return None
            raise

    async def _wait_for_runnable(self, service: Any) -> dict:
        """Poll instance until it reaches RUNNABLE state."""
        for _ in range(_MAX_POLL_ATTEMPTS):
            instance = await self._get_instance(service)

            if instance is None:
                msg = "Instance not found during polling"
                raise RuntimeError(msg)

            state = instance.get("state")

            if state == "RUNNABLE":
                return instance

            if state in ("FAILED", "SUSPENDED"):
                msg = f"Instance entered {state} state"
                raise RuntimeError(msg)

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        msg = f"Instance did not reach RUNNABLE state within {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS} seconds"
        raise TimeoutError(msg)

    async def _wait_for_deletion(self, service: Any) -> None:
        """Poll until instance is deleted."""
        for _ in range(_MAX_POLL_ATTEMPTS):
            instance = await self._get_instance(service)

            if instance is None:
                return

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        msg = f"Instance was not deleted within {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS} seconds"
        raise TimeoutError(msg)

    def _build_instance_body(self) -> dict:
        """Build instance body for create request."""
        authorized_networks = [
            {"name": f"network-{i}", "value": network} for i, network in enumerate(self.config.authorized_networks)
        ]

        ip_configuration: dict[str, Any] = {
            "ipv4Enabled": self.config.enable_public_ip,
        }

        if authorized_networks:
            ip_configuration["authorizedNetworks"] = authorized_networks

        settings: dict[str, Any] = {
            "tier": self.config.tier,
            "availabilityType": self.config.availability_type,
            "ipConfiguration": ip_configuration,
            "deletionProtectionEnabled": self.config.deletion_protection,
        }

        if self.config.backup_enabled:
            settings["backupConfiguration"] = {"enabled": True, "startTime": "03:00"}

        return {
            "name": self.config.instance_name,
            "region": self.config.region,
            "databaseVersion": self.config.database_version,
            "settings": settings,
            "rootPassword": _generate_root_password(),
        }

    async def _create_database(self, service: Any) -> None:
        """Create the application database."""
        database_body = {
            "name": self.config.database_name,
            "project": self.config.project_id,
            "instance": self.config.instance_name,
        }

        try:
            request = service.databases().insert(
                project=self.config.project_id,
                instance=self.config.instance_name,
                body=database_body,
            )
            await self._run_in_executor(request.execute)
        except HttpError as e:
            if "already exists" not in str(e).lower():
                raise

    async def on_create(self) -> CloudSQLOutputs:
        """Create Cloud SQL instance and wait for RUNNABLE state.

        Idempotent: If instance already exists, returns its current state.

        Returns:
            CloudSQLOutputs with instance details.
        """
        service = self._get_service()

        existing = await self._get_instance(service)

        if existing is None:
            request = service.instances().insert(project=self.config.project_id, body=self._build_instance_body())
            await self._run_in_executor(request.execute)

        instance = await self._wait_for_runnable(service)

        await self._create_database(service)

        return self._build_outputs(instance, self.config.database_name)

    async def on_update(self, previous_config: CloudSQLConfig) -> CloudSQLOutputs:
        """Update instance configuration.

        Most Cloud SQL instance properties require recreation. This method validates
        that immutable fields haven't changed and returns current state.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            CloudSQLOutputs with current instance state.

        Raises:
            ValueError: If immutable fields changed (requires delete + create).
        """
        if previous_config.project_id != self.config.project_id:
            msg = "Cannot change project_id; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.region != self.config.region:
            msg = "Cannot change region; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.instance_name != self.config.instance_name:
            msg = "Cannot change instance_name; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.database_version != self.config.database_version:
            msg = "Cannot change database_version; delete and recreate resource"
            raise ValueError(msg)

        if self.outputs is not None:
            return self.outputs

        service = self._get_service()
        instance = await self._get_instance(service)

        if instance is None:
            msg = "Instance not found"
            raise RuntimeError(msg)

        return self._build_outputs(instance, self.config.database_name)

    async def on_delete(self) -> None:
        """Delete instance.

        Idempotent: Succeeds if instance doesn't exist.

        Note: Respects deletion_protection setting on the instance.
        """
        service = self._get_service()

        try:
            request = service.instances().delete(project=self.config.project_id, instance=self.config.instance_name)
            await self._run_in_executor(request.execute)
            await self._wait_for_deletion(service)
        except HttpError as e:
            if e.resp.status != 404:
                raise

    async def health(self) -> HealthStatus:
        """Check instance health by querying instance status."""
        service = self._get_service()

        instance = await self._get_instance(service)

        if instance is None:
            return HealthStatus(
                status="unhealthy",
                message="Instance not found",
            )

        state = instance.get("state")

        if state == "RUNNABLE":
            return HealthStatus(
                status="healthy",
                message="Instance is running",
                details={"tier": instance.get("settings", {}).get("tier")},
            )

        if state in ("PENDING_CREATE", "MAINTENANCE"):
            return HealthStatus(
                status="degraded",
                message=f"Instance is {state.lower().replace('_', ' ')}",
            )

        return HealthStatus(
            status="unhealthy",
            message=f"Instance state: {state}",
        )

    async def logs(
        self,
        since: datetime | None = None,
        tail: int = 100,
    ) -> AsyncIterator[LogEntry]:
        """Fetch instance logs from Cloud Logging."""
        creds_data = self.config.credentials

        if isinstance(creds_data, str):
            creds_data = json.loads(creds_data)

        credentials = service_account.Credentials.from_service_account_info(creds_data)
        logging_client = LoggingClient(credentials=credentials, project=self.config.project_id)

        filter_parts = [
            'resource.type="cloudsql_database"',
            f'resource.labels.database_id="{self.config.project_id}:{self.config.instance_name}"',
        ]

        if since:
            filter_parts.append(f'timestamp>="{since.isoformat()}Z"')

        filter_str = " AND ".join(filter_parts)

        entries = logging_client.list_entries(
            filter_=filter_str,
            order_by="timestamp desc",
            max_results=tail,
        )

        for entry in entries:
            level = "info"

            if hasattr(entry, "severity"):
                severity = str(entry.severity).lower()

                if "error" in severity or "critical" in severity:
                    level = "error"
                elif "warn" in severity:
                    level = "warn"
                elif "debug" in severity:
                    level = "debug"

            yield LogEntry(
                timestamp=entry.timestamp,
                level=level,
                message=str(entry.payload) if entry.payload else "",
                metadata={"log_name": entry.log_name} if entry.log_name else None,
            )
