"""GCP GKE cluster resource supporting both Autopilot and Standard modes."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any, ClassVar, Literal, Self

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud.container_v1 import ClusterManagerAsyncClient
from google.cloud.container_v1.types import (
    Cluster,
    CreateClusterRequest,
    DeleteClusterRequest,
    GetClusterRequest,
    NodeConfig,
    NodePool,
)
from google.cloud.logging_v2 import Client as LoggingClient
from google.oauth2 import service_account
from pydantic import Field, field_validator, model_validator
from pragma_sdk import Config, HealthStatus, LogEntry, Outputs, Resource

_CLUSTER_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,38}[a-z0-9]$|^[a-z]$")


class GKEConfig(Config):
    """Configuration for a GKE cluster.

    Attributes:
        project_id: GCP project ID where the cluster will be created.
        credentials: GCP service account credentials JSON object or string.
        location: GCP location - either a region (e.g., europe-west4) for regional
            clusters or a zone (e.g., europe-west4-a) for zonal clusters.
        name: Name of the GKE cluster.
        autopilot: Whether to create an Autopilot cluster. Defaults to True.
        network: VPC network name. Defaults to "default".
        subnetwork: VPC subnetwork name. If not specified, uses network default.
        release_channel: Release channel for cluster updates.
        initial_node_count: Number of nodes in default pool (standard clusters only).
        machine_type: Machine type for nodes (standard clusters only).
        disk_size_gb: Boot disk size in GB (standard clusters only).
    """

    project_id: str
    credentials: dict[str, Any] | str
    location: str
    name: str
    autopilot: bool = True
    network: str = "default"
    subnetwork: str | None = None
    release_channel: Literal["RAPID", "REGULAR", "STABLE"] = "REGULAR"
    initial_node_count: Annotated[int, Field(ge=1)] = 1
    machine_type: str = "e2-medium"
    disk_size_gb: Annotated[int, Field(ge=10)] = 100

    @field_validator("name")
    @classmethod
    def validate_cluster_name(cls, v: str) -> str:
        """Validate cluster name follows GCP naming rules."""
        if not _CLUSTER_NAME_PATTERN.match(v):
            msg = (
                "Cluster name must start with a lowercase letter, contain only "
                "lowercase letters, numbers, and hyphens, and be 1-40 characters"
            )
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_standard_cluster_config(self) -> Self:
        """Validate node configuration for standard clusters."""
        if not self.autopilot and self.initial_node_count < 1:
            msg = "Standard clusters require initial_node_count >= 1"
            raise ValueError(msg)
        return self


class GKEOutputs(Outputs):
    """Outputs from GKE cluster creation.

    Attributes:
        name: Cluster name.
        endpoint: Cluster API server endpoint URL.
        cluster_ca_certificate: Base64-encoded cluster CA certificate.
        location: Cluster location (region or zone).
        status: Cluster status (RUNNING, PROVISIONING, etc.).
        console_url: URL to view cluster in GCP Console.
        logs_url: URL to view cluster logs in Cloud Logging.
    """

    name: str
    endpoint: str
    cluster_ca_certificate: str
    location: str
    status: str
    console_url: str
    logs_url: str


# Polling configuration for cluster operations
_POLL_INTERVAL_SECONDS = 30
_MAX_POLL_ATTEMPTS = 40  # 40 * 30s = 20 minutes max wait


class GKE(Resource[GKEConfig, GKEOutputs]):
    """GCP GKE cluster resource supporting Autopilot and Standard modes.

    Creates and manages GKE clusters using user-provided service account
    credentials (multi-tenant SaaS pattern).

    Modes:
        - Autopilot (default): Fully managed, no node configuration needed
        - Standard: Manual node pool with configurable machine type and count

    Lifecycle:
        - on_create: Creates cluster, waits for RUNNING state
        - on_update: Limited updates (some require recreation)
        - on_delete: Deletes cluster, waits for completion
    """

    provider: ClassVar[str] = "gcp"
    resource: ClassVar[str] = "gke"

    def _get_client(self) -> ClusterManagerAsyncClient:
        """Get Cluster Manager async client with user-provided credentials.

        Returns:
            Configured Cluster Manager async client.

        Raises:
            ValueError: If credentials format is invalid.
        """
        creds_data = self.config.credentials

        if isinstance(creds_data, str):
            creds_data = json.loads(creds_data)

        credentials = service_account.Credentials.from_service_account_info(creds_data)
        return ClusterManagerAsyncClient(credentials=credentials)

    def _cluster_path(self) -> str:
        """Build cluster resource path.

        Returns:
            Full GCP resource path for this cluster.
        """
        return f"projects/{self.config.project_id}/locations/{self.config.location}/clusters/{self.config.name}"

    def _parent_path(self) -> str:
        """Build parent resource path for cluster creation.

        Returns:
            Parent path (project/location).
        """
        return f"projects/{self.config.project_id}/locations/{self.config.location}"

    def _build_outputs(self, cluster: Cluster) -> GKEOutputs:
        """Build outputs from cluster object."""
        project = self.config.project_id
        location = self.config.location
        name = self.config.name

        console_url = (
            f"https://console.cloud.google.com/kubernetes/clusters/details/"
            f"{location}/{name}/details?project={project}"
        )
        logs_url = (
            f"https://console.cloud.google.com/logs/query;query="
            f"resource.type%3D%22k8s_cluster%22%0A"
            f"resource.labels.cluster_name%3D%22{name}%22%0A"
            f"resource.labels.location%3D%22{location}%22"
            f"?project={project}"
        )

        return GKEOutputs(
            name=cluster.name,
            endpoint=cluster.endpoint,
            cluster_ca_certificate=cluster.master_auth.cluster_ca_certificate,
            location=cluster.location,
            status=Cluster.Status(cluster.status).name,
            console_url=console_url,
            logs_url=logs_url,
        )

    async def _wait_for_running(self, client: ClusterManagerAsyncClient) -> Cluster:
        """Poll cluster until it reaches RUNNING state.

        Args:
            client: Cluster Manager client.

        Returns:
            Cluster in RUNNING state.

        Raises:
            TimeoutError: If cluster doesn't reach RUNNING in time.
            RuntimeError: If cluster enters ERROR state.
        """
        for _ in range(_MAX_POLL_ATTEMPTS):
            cluster = await client.get_cluster(request=GetClusterRequest(name=self._cluster_path()))

            if cluster.status == Cluster.Status.RUNNING:
                return cluster

            if cluster.status == Cluster.Status.ERROR:
                msg = f"Cluster entered ERROR state: {cluster.status_message}"
                raise RuntimeError(msg)

            if cluster.status in (
                Cluster.Status.STOPPING,
                Cluster.Status.DEGRADED,
            ):
                msg = f"Cluster in unexpected state: {cluster.status.name}"
                raise RuntimeError(msg)

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        msg = f"Cluster did not reach RUNNING state within {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS} seconds"
        raise TimeoutError(msg)

    async def _wait_for_deletion(self, client: ClusterManagerAsyncClient) -> None:
        """Poll until cluster is deleted.

        Args:
            client: Cluster Manager client.

        Raises:
            TimeoutError: If cluster doesn't delete in time.
        """
        for _ in range(_MAX_POLL_ATTEMPTS):
            try:
                await client.get_cluster(request=GetClusterRequest(name=self._cluster_path()))
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            except NotFound:
                return

        msg = f"Cluster was not deleted within {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS} seconds"
        raise TimeoutError(msg)

    def _build_cluster_config(self) -> Cluster:
        """Build cluster configuration object.

        Returns:
            Cluster configuration for create request.
        """
        cluster = Cluster(
            name=self.config.name,
            network=self.config.network,
            release_channel={"channel": self.config.release_channel},
        )

        if self.config.subnetwork:
            cluster.subnetwork = self.config.subnetwork

        if self.config.autopilot:
            cluster.autopilot = {"enabled": True}
        else:
            cluster.node_pools = [
                NodePool(
                    name="default-pool",
                    initial_node_count=self.config.initial_node_count,
                    config=NodeConfig(
                        machine_type=self.config.machine_type,
                        disk_size_gb=self.config.disk_size_gb,
                    ),
                )
            ]

        return cluster

    async def on_create(self) -> GKEOutputs:
        """Create GKE cluster and wait for RUNNING state.

        Idempotent: If cluster already exists, returns its current state.

        Returns:
            GKEOutputs with cluster details.
        """
        client = self._get_client()

        try:
            await client.create_cluster(
                request=CreateClusterRequest(
                    parent=self._parent_path(),
                    cluster=self._build_cluster_config(),
                )
            )
        except AlreadyExists:
            pass

        cluster = await self._wait_for_running(client)

        return self._build_outputs(cluster)

    async def on_update(self, previous_config: GKEConfig) -> GKEOutputs:
        """Update cluster configuration.

        Most GKE cluster properties require recreation. This method validates
        that immutable fields haven't changed and returns current state.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            GKEOutputs with current cluster state.

        Raises:
            ValueError: If immutable fields changed (requires delete + create).
        """
        if previous_config.project_id != self.config.project_id:
            msg = "Cannot change project_id; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.location != self.config.location:
            msg = "Cannot change location; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.name != self.config.name:
            msg = "Cannot change name; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.network != self.config.network:
            msg = "Cannot change network; delete and recreate resource"
            raise ValueError(msg)

        if previous_config.autopilot != self.config.autopilot:
            msg = "Cannot change autopilot mode; delete and recreate resource"
            raise ValueError(msg)

        # For unchanged configs, return existing outputs if available
        if self.outputs is not None:
            return self.outputs

        # Fetch current cluster state
        client = self._get_client()
        cluster = await client.get_cluster(request=GetClusterRequest(name=self._cluster_path()))

        return self._build_outputs(cluster)

    async def on_delete(self) -> None:
        """Delete cluster and wait for completion.

        Idempotent: Succeeds if cluster doesn't exist.
        """
        client = self._get_client()

        try:
            await client.delete_cluster(request=DeleteClusterRequest(name=self._cluster_path()))
            await self._wait_for_deletion(client)
        except NotFound:
            pass

    async def health(self) -> HealthStatus:
        """Check cluster health by querying cluster status."""
        client = self._get_client()

        try:
            cluster = await client.get_cluster(
                request=GetClusterRequest(name=self._cluster_path())
            )
        except NotFound:
            return HealthStatus(
                status="unhealthy",
                message="Cluster not found",
            )

        status = Cluster.Status(cluster.status)

        if status == Cluster.Status.RUNNING:
            return HealthStatus(
                status="healthy",
                message="Cluster is running",
                details={"node_count": sum(np.initial_node_count for np in cluster.node_pools)},
            )

        if status in (Cluster.Status.PROVISIONING, Cluster.Status.RECONCILING):
            return HealthStatus(
                status="degraded",
                message=f"Cluster is {status.name.lower()}",
            )

        return HealthStatus(
            status="unhealthy",
            message=f"Cluster status: {status.name}",
            details={"status_message": cluster.status_message} if cluster.status_message else None,
        )

    async def logs(
        self,
        since: datetime | None = None,
        tail: int = 100,
    ) -> AsyncIterator[LogEntry]:
        """Fetch cluster logs from Cloud Logging."""
        creds_data = self.config.credentials
        if isinstance(creds_data, str):
            creds_data = json.loads(creds_data)

        credentials = service_account.Credentials.from_service_account_info(creds_data)
        logging_client = LoggingClient(credentials=credentials, project=self.config.project_id)

        filter_parts = [
            'resource.type="k8s_cluster"',
            f'resource.labels.cluster_name="{self.config.name}"',
            f'resource.labels.location="{self.config.location}"',
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
