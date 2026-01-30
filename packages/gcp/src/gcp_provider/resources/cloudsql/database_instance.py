"""GCP Cloud SQL database instance resource."""

from __future__ import annotations

import asyncio
import secrets
import string
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, ClassVar, Literal

from google.cloud.logging_v2 import Client as LoggingClient
from pydantic import Field, field_validator
from pragma_sdk import Config, HealthStatus, LogEntry, Outputs, Resource

from gcp_provider.resources.cloudsql.helpers import (
    execute,
    extract_ips,
    get_credentials,
    get_sqladmin_service,
    run_in_executor,
)


_POLL_INTERVAL_SECONDS = 30
_MAX_POLL_ATTEMPTS = 30


class DatabaseInstanceConfig(Config):
    """Configuration for a Cloud SQL database instance.

    Attributes:
        project_id: GCP project ID where the instance will be created.
        credentials: GCP service account credentials JSON object or string.
        region: GCP region (e.g., europe-west4).
        instance_name: Name of the Cloud SQL instance (must be unique per project).
        database_version: Database version (e.g., POSTGRES_15, POSTGRES_14, MYSQL_8_0).
        tier: Machine tier (e.g., db-f1-micro, db-custom-1-3840).
        availability_type: ZONAL (single zone) or REGIONAL (high availability).
        backup_enabled: Whether automatic backups are enabled.
        deletion_protection: Whether deletion protection is enabled.
        authorized_networks: List of CIDR ranges to allow connections from.
        enable_public_ip: Whether to assign a public IP address.
    """

    project_id: str = Field(json_schema_extra={"immutable": True})
    credentials: dict[str, Any] | str
    region: str = Field(json_schema_extra={"immutable": True})
    instance_name: str = Field(json_schema_extra={"immutable": True})
    database_version: str = Field(default="POSTGRES_15", json_schema_extra={"immutable": True})
    tier: str = "db-f1-micro"
    availability_type: Literal["ZONAL", "REGIONAL"] = "ZONAL"
    backup_enabled: bool = True
    deletion_protection: bool = False
    authorized_networks: list[str] = Field(default_factory=list)
    enable_public_ip: bool = True

    def validate_update(self, previous: DatabaseInstanceConfig) -> None:
        """Validate that immutable fields have not changed.

        Raises:
            ValueError: If any immutable field differs from previous config.
        """
        for name, field_info in self.__class__.model_fields.items():
            extra = field_info.json_schema_extra

            if isinstance(extra, dict) and extra.get("immutable"):
                current_value = getattr(self, name)
                previous_value = getattr(previous, name)

                if current_value != previous_value:
                    msg = f"Cannot change {name}; delete and recreate resource"
                    raise ValueError(msg)

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


class DatabaseInstanceOutputs(Outputs):
    """Outputs from Cloud SQL database instance creation.

    Attributes:
        connection_name: Cloud SQL connection name (project:region:instance).
        public_ip: Public IP address (if enabled).
        private_ip: Private IP address (if enabled).
        ready: Whether the instance is running and accessible.
        console_url: URL to view instance in GCP Console.
        logs_url: URL to view instance logs in Cloud Logging.
    """

    connection_name: str
    public_ip: str | None
    private_ip: str | None
    ready: bool
    console_url: str
    logs_url: str


class DatabaseInstance(Resource[DatabaseInstanceConfig, DatabaseInstanceOutputs]):
    """GCP Cloud SQL database instance resource.

    Creates and manages Cloud SQL instances (PostgreSQL, MySQL, SQL Server)
    using user-provided service account credentials (multi-tenant SaaS pattern).

    Lifecycle:
        - on_create: Creates instance, waits for RUNNABLE state
        - on_update: Limited updates (some require recreation)
        - on_delete: Deletes instance (respects deletion_protection)
    """

    provider: ClassVar[str] = "gcp"
    resource: ClassVar[str] = "cloudsql/database_instance"

    async def on_create(self) -> DatabaseInstanceOutputs:
        """Create Cloud SQL instance and wait for RUNNABLE state.

        Idempotent: If instance already exists, returns its current state.

        Returns:
            DatabaseInstanceOutputs with instance details.
        """
        service = get_sqladmin_service(get_credentials(self.config.credentials))

        existing = await execute(
            service.instances().get(project=self.config.project_id, instance=self.config.instance_name),
            ignore_404=True,
        )

        if existing is None:
            await execute(service.instances().insert(project=self.config.project_id, body=self._build_instance_body()))

        instance = await self._wait_for_runnable(service)

        return self._build_outputs(instance)

    async def on_update(self, previous_config: DatabaseInstanceConfig) -> DatabaseInstanceOutputs:
        """Update instance configuration.

        Updates mutable settings (tier, availability, backups, network config).
        Immutable fields (project, region, name, database version) cannot be changed.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            DatabaseInstanceOutputs with updated instance state.

        Raises:
            ValueError: If immutable fields changed (requires delete + create).
        """
        self.config.validate_update(previous_config)

        service = get_sqladmin_service(get_credentials(self.config.credentials))

        await execute(
            service.instances().patch(
                project=self.config.project_id,
                instance=self.config.instance_name,
                body=self._build_patch_body(),
            )
        )

        instance = await self._wait_for_runnable(service)

        return self._build_outputs(instance)

    async def on_delete(self) -> None:
        """Delete instance.

        Idempotent: Succeeds if instance doesn't exist.

        Note: Respects deletion_protection setting on the instance.
        """
        service = get_sqladmin_service(get_credentials(self.config.credentials))

        result = await execute(
            service.instances().delete(project=self.config.project_id, instance=self.config.instance_name),
            ignore_404=True,
        )

        if result is not None:
            await self._wait_for_deletion(service)

    async def health(self) -> HealthStatus:
        """Check instance health by querying instance status."""
        service = get_sqladmin_service(get_credentials(self.config.credentials))

        instance = await execute(
            service.instances().get(project=self.config.project_id, instance=self.config.instance_name),
            ignore_404=True,
        )

        if instance is None:
            return HealthStatus(status="unhealthy", message="Instance not found")

        state = instance.get("state")
        status_map = {
            "RUNNABLE": ("healthy", "Instance is running"),
            "PENDING_CREATE": ("degraded", "Instance is pending create"),
            "MAINTENANCE": ("degraded", "Instance is in maintenance"),
        }
        status, message = status_map.get(state, ("unhealthy", f"Instance state: {state}"))

        return HealthStatus(
            status=status,
            message=message,
            details={"tier": instance.get("settings", {}).get("tier")} if status == "healthy" else None,
        )

    async def logs(
        self,
        since: datetime | None = None,
        tail: int = 100,
    ) -> AsyncIterator[LogEntry]:
        """Fetch instance logs from Cloud Logging."""
        credentials = get_credentials(self.config.credentials)
        project = self.config.project_id

        filter_parts = [
            'resource.type="cloudsql_database"',
            f'resource.labels.database_id="{self.config.project_id}:{self.config.instance_name}"',
        ]

        if since:
            filter_parts.append(f'timestamp>="{since.isoformat()}Z"')

        filter_str = " AND ".join(filter_parts)

        def fetch_logs() -> list:
            logging_client = LoggingClient(credentials=credentials, project=project)
            return list(
                logging_client.list_entries(
                    filter_=filter_str,
                    order_by="timestamp desc",
                    max_results=tail,
                )
            )

        entries = await run_in_executor(fetch_logs)

        for entry in entries:
            yield LogEntry(
                timestamp=entry.timestamp,
                level=self._severity_to_level(entry),
                message=str(entry.payload) if entry.payload else "",
                metadata={"log_name": entry.log_name} if entry.log_name else None,
            )

    @staticmethod
    def _severity_to_level(entry: Any) -> str:
        """Convert Cloud Logging severity to log level."""
        if not hasattr(entry, "severity"):
            return "info"

        severity = str(entry.severity).lower()

        if "error" in severity or "critical" in severity:
            return "error"
        if "warn" in severity:
            return "warn"
        if "debug" in severity:
            return "debug"

        return "info"

    @staticmethod
    def _generate_root_password() -> str:
        """Generate a secure random password for the root user."""
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(24))

    def _build_outputs(self, instance: dict) -> DatabaseInstanceOutputs:
        """Build outputs from instance dict."""
        public_ip, private_ip = extract_ips(instance)

        console_url = f"https://console.cloud.google.com/sql/instances/{self.config.instance_name}/overview?project={self.config.project_id}"
        logs_url = (
            f"https://console.cloud.google.com/logs/query;"
            f"query=resource.type%3D%22cloudsql_database%22%0A"
            f"resource.labels.database_id%3D%22{self.config.project_id}%3A{self.config.instance_name}%22"
            f"?project={self.config.project_id}"
        )

        return DatabaseInstanceOutputs(
            connection_name=f"{self.config.project_id}:{self.config.region}:{self.config.instance_name}",
            public_ip=public_ip,
            private_ip=private_ip,
            ready=instance.get("state") == "RUNNABLE",
            console_url=console_url,
            logs_url=logs_url,
        )

    async def _wait_for_runnable(self, service: Any) -> dict:
        """Poll instance until it reaches RUNNABLE state."""
        for _ in range(_MAX_POLL_ATTEMPTS):
            instance = await execute(
                service.instances().get(project=self.config.project_id, instance=self.config.instance_name),
                ignore_404=True,
            )

            if instance is None:
                raise RuntimeError("Instance not found during polling")

            state = instance.get("state")

            if state == "RUNNABLE":
                return instance

            if state in ("FAILED", "SUSPENDED"):
                raise RuntimeError(f"Instance entered {state} state")

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        raise TimeoutError(
            f"Instance did not reach RUNNABLE state within {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS} seconds"
        )

    async def _wait_for_deletion(self, service: Any) -> None:
        """Poll until instance is deleted."""
        for _ in range(_MAX_POLL_ATTEMPTS):
            instance = await execute(
                service.instances().get(project=self.config.project_id, instance=self.config.instance_name),
                ignore_404=True,
            )

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
            "rootPassword": self._generate_root_password(),
        }

    def _build_patch_body(self) -> dict:
        """Build patch body for update request (mutable settings only)."""
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
            "backupConfiguration": {"enabled": self.config.backup_enabled, "startTime": "03:00"},
        }

        return {"settings": settings}
