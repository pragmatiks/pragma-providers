"""GCP GKE Autopilot cluster resource."""

from __future__ import annotations

import asyncio
import json
from typing import Any, ClassVar, Literal

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud.container_v1 import ClusterManagerAsyncClient
from google.cloud.container_v1.types import Cluster, CreateClusterRequest, DeleteClusterRequest, GetClusterRequest
from google.oauth2 import service_account
from pragma_sdk import Config, Outputs, Resource


class GKEConfig(Config):
    """Configuration for a GKE Autopilot cluster.

    Attributes:
        project_id: GCP project ID where the cluster will be created.
        credentials: GCP service account credentials JSON object or string.
        region: GCP region for the cluster (e.g., europe-west4).
        name: Name of the GKE cluster.
        autopilot: Whether to create an Autopilot cluster. Defaults to True.
        network: VPC network name. Defaults to "default".
        subnetwork: VPC subnetwork name. If not specified, uses network default.
        release_channel: Release channel for cluster updates.
    """

    project_id: str
    credentials: dict[str, Any] | str
    region: str
    name: str
    autopilot: bool = True
    network: str = "default"
    subnetwork: str | None = None
    release_channel: Literal["RAPID", "REGULAR", "STABLE"] = "REGULAR"


class GKEOutputs(Outputs):
    """Outputs from GKE cluster creation.

    Attributes:
        name: Cluster name.
        endpoint: Cluster API server endpoint URL.
        cluster_ca_certificate: Base64-encoded cluster CA certificate.
        location: Cluster location (region).
        status: Cluster status (RUNNING, PROVISIONING, etc.).
    """

    name: str
    endpoint: str
    cluster_ca_certificate: str
    location: str
    status: str


# Polling configuration for cluster operations
_POLL_INTERVAL_SECONDS = 30
_MAX_POLL_ATTEMPTS = 40  # 40 * 30s = 20 minutes max wait


class GKE(Resource[GKEConfig, GKEOutputs]):
    """GCP GKE Autopilot cluster resource.

    Creates and manages GKE Autopilot clusters using user-provided
    service account credentials (multi-tenant SaaS pattern).

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
        return f"projects/{self.config.project_id}/locations/{self.config.region}/clusters/{self.config.name}"

    def _parent_path(self) -> str:
        """Build parent resource path for cluster creation.

        Returns:
            Parent path (project/location).
        """
        return f"projects/{self.config.project_id}/locations/{self.config.region}"

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
            cluster = await client.get_cluster(
                request=GetClusterRequest(name=self._cluster_path())
            )

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
                await client.get_cluster(
                    request=GetClusterRequest(name=self._cluster_path())
                )
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

        return GKEOutputs(
            name=cluster.name,
            endpoint=cluster.endpoint,
            cluster_ca_certificate=cluster.master_auth.cluster_ca_certificate,
            location=cluster.location,
            status=Cluster.Status(cluster.status).name,
        )

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

        if previous_config.region != self.config.region:
            msg = "Cannot change region; delete and recreate resource"
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
        cluster = await client.get_cluster(
            request=GetClusterRequest(name=self._cluster_path())
        )

        return GKEOutputs(
            name=cluster.name,
            endpoint=cluster.endpoint,
            cluster_ca_certificate=cluster.master_auth.cluster_ca_certificate,
            location=cluster.location,
            status=Cluster.Status(cluster.status).name,
        )

    async def on_delete(self) -> None:
        """Delete cluster and wait for completion.

        Idempotent: Succeeds if cluster doesn't exist.
        """
        client = self._get_client()

        try:
            await client.delete_cluster(
                request=DeleteClusterRequest(name=self._cluster_path())
            )
            await self._wait_for_deletion(client)
        except NotFound:
            pass
