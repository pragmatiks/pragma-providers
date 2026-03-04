"""GCP Cloud SQL database instance resource."""

from __future__ import annotations

import asyncio
import secrets
import string
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, ClassVar, Literal

from google.cloud.logging_v2 import Client as LoggingClient
from pragma_sdk import Config, Field, HealthStatus, ImmutableField, LogEntry, Outputs, Resource
from pydantic import Field as PydanticField
from pydantic import field_validator

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

    project_id: ImmutableField[str] = PydanticField(
        description="GCP project ID where the instance will be created.",
    )
    credentials: Field[dict[str, Any] | str] = PydanticField(
        description="GCP service account credentials as a JSON object or JSON string.",
    )
    region: ImmutableField[str] = PydanticField(
        description="GCP region for the instance (e.g., europe-west4).",
    )
    instance_name: ImmutableField[str] = PydanticField(
        description="Cloud SQL instance name. Must be unique per project, 1-98 characters, "
        "start with a letter, and contain only letters, numbers, and hyphens.",
    )
    database_version: ImmutableField[str] = PydanticField(
        default="POSTGRES_15",
        description="Database engine and version (e.g., POSTGRES_15, MYSQL_8_0, SQLSERVER_2019_STANDARD).",
    )
    tier: Field[str] = PydanticField(
        default="db-f1-micro",
        description="Machine tier for the instance (e.g., db-f1-micro, db-custom-1-3840, db-custom-2-7680).",
    )
    availability_type: Field[Literal["ZONAL", "REGIONAL"]] = PydanticField(
        default="ZONAL",
        description="ZONAL for single-zone or REGIONAL for high availability with automatic failover.",
    )
    backup_enabled: Field[bool] = PydanticField(
        default=True,
        description="Enable automatic daily backups at 03:00 UTC.",
    )
    deletion_protection: Field[bool] = PydanticField(
        default=False,
        description="Prevent accidental deletion. Must be disabled before the instance can be deleted.",
    )
    authorized_networks: list[Field[str]] = PydanticField(
        default=[],
        description="CIDR ranges allowed to connect (e.g., ['10.0.0.0/8', '192.168.1.0/24']).",
    )
    enable_public_ip: Field[bool] = PydanticField(
        default=True,
        description="Assign a public IP address to the instance.",
    )

    @field_validator("instance_name")
    @classmethod
    def validate_instance_name(cls, v: str) -> str:
        """Validate instance name follows Cloud SQL naming rules.

        Returns:
            The validated instance name.

        Raises:
            ValueError: If instance name violates naming rules.
        """
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
        """Validate database version is supported.

        Returns:
            The validated database version.

        Raises:
            ValueError: If database version is not supported.
        """
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

    connection_name: str = PydanticField(
        description="Cloud SQL connection name in project:region:instance format. "
        "Used by Cloud SQL Proxy and client libraries.",
    )
    public_ip: str | None = PydanticField(
        default=None,
        description="Public IP address of the instance, if public IP is enabled.",
    )
    private_ip: str | None = PydanticField(
        default=None,
        description="Private IP address of the instance, if private IP is configured.",
    )
    ready: bool = PydanticField(
        description="Whether the instance is in RUNNABLE state and accepting connections.",
    )
    console_url: str = PydanticField(description="URL to view the instance in the GCP Console.")
    logs_url: str = PydanticField(description="URL to view instance logs in Cloud Logging.")


class DatabaseInstance(Resource[DatabaseInstanceConfig, DatabaseInstanceOutputs]):
    """GCP Cloud SQL database instance resource.

    Creates and manages Cloud SQL instances for PostgreSQL, MySQL, and SQL Server
    using user-provided service account credentials (multi-tenant SaaS pattern).
    Supports configurable machine tiers, high availability, automatic backups,
    and network access control. Includes health checks and log streaming from
    Cloud Logging.

    Lifecycle:
        - on_create: Creates the instance with a randomly generated root password
          and polls until RUNNABLE (up to 15 minutes). Idempotent -- if the
          instance already exists, waits for RUNNABLE and returns current state.
        - on_update: Patches mutable settings (tier, availability_type, backups,
          network config, deletion_protection) and waits for RUNNABLE.
        - on_delete: Deletes the instance. Respects ``deletion_protection`` --
          disable it first to allow deletion. Idempotent -- succeeds silently
          if the instance does not exist.

    Required IAM role: ``roles/cloudsql.admin``

    Required APIs: ``sqladmin.googleapis.com``, ``logging.googleapis.com``

    Example::

        resources:
          - name: prod-instance
            provider: gcp
            type: cloudsql/database_instance
            config:
              project_id: my-project
              region: europe-west4
              instance_name: prod-postgres
              database_version: POSTGRES_15
              tier: db-custom-2-7680
              availability_type: REGIONAL
              backup_enabled: true
              credentials:
                $ref: gcp-credentials
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

        Args:
            previous_config: The previous configuration before update.

        Returns:
            DatabaseInstanceOutputs with updated instance state.
        """
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
        """Check instance health by querying instance status.

        Returns:
            HealthStatus indicating instance health.
        """
        service = get_sqladmin_service(get_credentials(self.config.credentials))

        instance = await execute(
            service.instances().get(project=self.config.project_id, instance=self.config.instance_name),
            ignore_404=True,
        )

        if instance is None:
            return HealthStatus(status="unhealthy", message="Instance not found")

        state = instance.get("state")
        status_map: dict[str, tuple[Literal["healthy", "unhealthy", "degraded"], str]] = {
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
        """Fetch instance logs from Cloud Logging.

        Yields:
            LogEntry objects from Cloud Logging.
        """
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
    def _severity_to_level(entry: Any) -> Literal["debug", "info", "warn", "error"]:
        """Convert Cloud Logging severity to log level.

        Returns:
            Log level string.
        """
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
        """Generate a secure random password for the root user.

        Returns:
            Random 24-character password.
        """
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(24))

    def _build_outputs(self, instance: dict) -> DatabaseInstanceOutputs:
        """Build outputs from instance dict.

        Returns:
            DatabaseInstanceOutputs with instance details.
        """
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
        """Poll instance until it reaches RUNNABLE state.

        Returns:
            Instance dict in RUNNABLE state.

        Raises:
            RuntimeError: If instance not found or enters FAILED/SUSPENDED state.
            TimeoutError: If instance doesn't reach RUNNABLE in time.
        """
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
        """Poll until instance is deleted.

        Raises:
            TimeoutError: If instance doesn't delete in time.
        """
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
        """Build instance body for create request.

        Returns:
            Instance body dict for Cloud SQL API.
        """
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
        """Build patch body for update request (mutable settings only).

        Returns:
            Patch body dict for Cloud SQL API.
        """
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
