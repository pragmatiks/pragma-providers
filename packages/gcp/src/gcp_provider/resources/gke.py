"""GCP GKE cluster resource supporting both Autopilot and Standard modes."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Any, Literal, Self, cast

from google.api_core.exceptions import AlreadyExists, FailedPrecondition, NotFound
from google.cloud.container_v1 import ClusterManagerAsyncClient
from google.cloud.container_v1.types import (
    Autopilot,
    Cluster,
    ClusterUpdate,
    CreateClusterRequest,
    DeleteClusterRequest,
    GetClusterRequest,
    GetOperationRequest,
    ListOperationsRequest,
    NodeConfig,
    NodePool,
    Operation,
    ReleaseChannel,
    UpdateClusterRequest,
)
from google.cloud.logging_v2 import Client as LoggingClient
from google.oauth2 import service_account
from pragma_sdk import Config, Field, HealthStatus, ImmutableField, LogEntry, Outputs, Resource
from pydantic import Field as PydanticField
from pydantic import field_validator, model_validator

from gcp_provider.resources.polling import compute_operation_deadline, poll_until


logger = logging.getLogger(__name__)

_CLUSTER_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,38}[a-z0-9]$|^[a-z]$")


def targets_cluster(target_link: str, cluster_name: str) -> bool:
    """Tell whether an operation's target is the cluster or one of its node pools.

    Args:
        target_link: The operation's ``target_link``.
        cluster_name: Name of the cluster.

    Returns:
        True if the target is the cluster itself or a resource under it.
    """
    cluster_suffix = f"/clusters/{cluster_name}"
    return target_link.endswith(cluster_suffix) or f"{cluster_suffix}/" in target_link


async def wait_for_operation_done(client: ClusterManagerAsyncClient, operation_path: str, deadline: float) -> Operation:
    """Wait until a GKE operation is done, whatever its outcome.

    Args:
        client: Cluster Manager client.
        operation_path: Operation resource path
            (``projects/{project}/locations/{location}/operations/{operation}``).
        deadline: Deadline on the ``time.monotonic`` clock, shared by the handler's waits.

    Returns:
        The finished operation, including any error it ended with.

    Raises:
        TimeoutError: If the operation is not done by ``deadline``.
    """
    operation = Operation()

    async for _ in poll_until(deadline):
        operation = await client.get_operation(request=GetOperationRequest(name=operation_path))

        if operation.status == Operation.Status.DONE:
            return operation

    msg = (
        f"GKE {operation.operation_type.name} operation {operation_path} was still "
        f"{operation.status.name} at the operation deadline"
    )
    raise TimeoutError(msg)


async def wait_for_operation_success(client: ClusterManagerAsyncClient, operation_path: str, deadline: float) -> None:
    """Wait until a GKE operation is done and require that it succeeded.

    Args:
        client: Cluster Manager client.
        operation_path: Operation resource path
            (``projects/{project}/locations/{location}/operations/{operation}``).
        deadline: Deadline on the ``time.monotonic`` clock, shared by the handler's waits.

    Raises:
        RuntimeError: If the operation finished with an error.
        TimeoutError: If the operation is not done by ``deadline``.
    """
    operation = await wait_for_operation_done(client, operation_path, deadline)

    if operation.error.code:
        msg = f"GKE {operation.operation_type.name} operation {operation_path} failed: {operation.error.message}"
        raise RuntimeError(msg)


async def wait_for_pending_operations(
    client: ClusterManagerAsyncClient,
    config: GKEConfig,
    resource_name: str,
    deadline: float,
) -> None:
    """Wait until every operation on a GKE cluster and its node pools is done, whatever its outcome.

    Logs ``gke_pending_operation_wait`` at WARNING before each wait, then ``gke_pending_operation_done``,
    or ``gke_pending_operation_failed`` with the error the operation ended with. That error is not raised.

    Args:
        client: Cluster Manager client.
        config: Configuration that locates the cluster.
        resource_name: Name of the Pragmatiks resource waiting, logged with each event.
        deadline: Deadline on the ``time.monotonic`` clock, shared by the handler's waits.

    Raises:
        TimeoutError: If a pending operation is not done by ``deadline``.
    """
    parent_path = f"projects/{config.project_id}/locations/{config.location}"
    response = await client.list_operations(request=ListOperationsRequest(parent=parent_path))
    pending = [
        operation
        for operation in response.operations
        if operation.status != Operation.Status.DONE and targets_cluster(operation.target_link, cast(str, config.name))
    ]

    for operation in pending:
        operation_path = f"{parent_path}/operations/{operation.name}"
        operation_type = operation.operation_type.name
        logger.warning(
            "gke_pending_operation_wait resource=%s operation_type=%s operation=%s",
            resource_name,
            operation_type,
            operation_path,
        )
        finished = await wait_for_operation_done(client, operation_path, deadline)

        if finished.error.code:
            logger.warning(
                "gke_pending_operation_failed resource=%s operation_type=%s operation=%s error=%s",
                resource_name,
                operation_type,
                operation_path,
                finished.error.message,
            )
        else:
            logger.warning(
                "gke_pending_operation_done resource=%s operation_type=%s operation=%s",
                resource_name,
                operation_type,
                operation_path,
            )


def is_cluster_busy(error: FailedPrecondition) -> bool:
    """Tell whether GKE refused a write because another operation runs on the cluster.

    Args:
        error: Error GKE raised for the write.

    Returns:
        True when the error carries the reason ``CLUSTER_ALREADY_HAS_OPERATION``, or, without that reason,
        a message GKE uses for a concurrent operation.
    """
    if error.reason == "CLUSTER_ALREADY_HAS_OPERATION":
        return True

    return "running incompatible operation" in error.message or "try again once it is done" in error.message


async def run_cluster_operation(
    client: ClusterManagerAsyncClient,
    config: GKEConfig,
    resource_name: str,
    write: Callable[[], Awaitable[Operation | None]],
    deadline: float,
) -> None:
    """Send a write to a GKE cluster once its pending operations are done, and wait until it succeeds.

    Args:
        client: Cluster Manager client.
        config: Configuration that locates the cluster.
        resource_name: Name of the Pragmatiks resource writing, logged with each pending-operation event.
        write: Sends the write and returns its operation, or None when no write was needed.
        deadline: Deadline on the ``time.monotonic`` clock, shared by the handler's waits.

    Raises:
        RuntimeError: If GKE refuses the write because another operation runs on the cluster,
            or the write's operation finished with an error.
        FailedPrecondition: If GKE refuses the write for any other precondition.
        TimeoutError: If a pending operation, or the write's operation, is not done by ``deadline``.
    """
    await wait_for_pending_operations(client, config, resource_name, deadline)

    try:
        operation = await write()
    except FailedPrecondition as error:
        if is_cluster_busy(error):
            message = (
                f"Cluster projects/{config.project_id}/locations/{config.location}/clusters/{config.name} "
                "is busy with another operation"
            )
            raise RuntimeError(message) from error
        raise

    if operation is None:
        return

    operation_path = f"projects/{config.project_id}/locations/{config.location}/operations/{operation.name}"
    await wait_for_operation_success(client, operation_path, deadline)


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

    project_id: ImmutableField[str]
    credentials: Field[dict[str, Any] | str]
    location: ImmutableField[str]
    name: ImmutableField[str]
    autopilot: ImmutableField[bool] = True
    network: ImmutableField[str] = "default"
    subnetwork: ImmutableField[str] | None = None
    release_channel: Field[Literal["RAPID", "REGULAR", "STABLE"]] = "REGULAR"
    initial_node_count: ImmutableField[int] = PydanticField(default=1, ge=1)
    machine_type: ImmutableField[str] = "e2-medium"
    disk_size_gb: ImmutableField[int] = PydanticField(default=100, ge=10)

    @field_validator("name")
    @classmethod
    def validate_cluster_name(cls, v: str) -> str:
        """Validate cluster name follows GCP naming rules.

        Returns:
            The validated cluster name.

        Raises:
            ValueError: If cluster name violates naming rules.
        """
        if not _CLUSTER_NAME_PATTERN.match(v):
            msg = (
                "Cluster name must start with a lowercase letter, contain only "
                "lowercase letters, numbers, and hyphens, and be 1-40 characters"
            )
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_standard_cluster_config(self) -> Self:
        """Validate node configuration for standard clusters.

        Returns:
            Self after validation.

        Raises:
            ValueError: If standard cluster has invalid node count.
        """
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
        console_url: URL to view cluster in GCP Console.
        logs_url: URL to view cluster logs in Cloud Logging.
    """

    name: str
    endpoint: str
    cluster_ca_certificate: str
    location: str
    console_url: str
    logs_url: str


class GKE(Resource[GKEConfig, GKEOutputs]):
    """GCP GKE cluster resource supporting Autopilot and Standard modes.

    Creates and manages GKE clusters using user-provided service account
    credentials (multi-tenant SaaS pattern). Supports health checks and
    log streaming from Cloud Logging.

    Modes:
        - **Autopilot** (default): Fully managed node infrastructure. GCP
          automatically provisions and scales nodes. No ``initial_node_count``,
          ``machine_type``, or ``disk_size_gb`` configuration needed.
        - **Standard**: Manual node pool with configurable machine type, count,
          and disk size. Set ``autopilot: false`` to use this mode.

    Lifecycle:
        - on_create: Creates the cluster and polls until it serves, in RUNNING
          or DEGRADED state (up to 19 min). Idempotent -- if the cluster
          already exists, waits for it to serve.
        - on_observe: Reads the cluster by project, location and name.
        - on_update: Applies ``release_channel`` when it differs and waits for
          the cluster to serve. ``credentials`` is mutable too; every other field
          is immutable and requires delete and recreate.
        - on_delete: Deletes the cluster and polls until fully removed.
          Idempotent -- succeeds silently if the cluster does not exist.

    Required IAM role: ``roles/container.admin``

    Required APIs: ``container.googleapis.com``, ``logging.googleapis.com``

    Example::

        resources:
          - name: prod-cluster
            provider: gcp
            type: gke
            config:
              project_id: my-project
              location: europe-west4
              name: prod-cluster
              autopilot: true
              release_channel: STABLE
              credentials:
                provider: pragma
                resource: secret
                name: gcp-credentials
                field: outputs.credentials_json
    """

    def _get_client(self) -> ClusterManagerAsyncClient:
        """Get Cluster Manager async client with user-provided credentials.

        Returns:
            Configured Cluster Manager async client.
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
        """Build outputs from cluster object.

        Returns:
            GKEOutputs with cluster details.
        """
        project = self.config.project_id
        location = self.config.location
        name = self.config.name

        console_url = (
            f"https://console.cloud.google.com/kubernetes/clusters/details/{location}/{name}/details?project={project}"
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
            console_url=console_url,
            logs_url=logs_url,
        )

    async def wait_for_serving(self, client: ClusterManagerAsyncClient, deadline: float) -> Cluster:
        """Poll the cluster until it serves, in RUNNING or DEGRADED state.

        Args:
            client: Cluster Manager client.
            deadline: Deadline on the ``time.monotonic`` clock, shared by the handler's waits.

        Returns:
            Cluster in RUNNING or DEGRADED state.

        Raises:
            TimeoutError: If the cluster does not serve by ``deadline``; the message names the last status seen.
            RuntimeError: If the cluster enters ERROR or STOPPING state.
        """
        cluster = Cluster()

        async for _ in poll_until(deadline):
            cluster = await client.get_cluster(request=GetClusterRequest(name=self._cluster_path()))

            if cluster.status in (Cluster.Status.RUNNING, Cluster.Status.DEGRADED):
                return cluster

            if cluster.status == Cluster.Status.ERROR:
                msg = f"Cluster entered ERROR state: {cluster.status_message}"
                raise RuntimeError(msg)

            if cluster.status == Cluster.Status.STOPPING:
                msg = f"Cluster in unexpected state: {cluster.status.name}"
                raise RuntimeError(msg)

        message = (
            f"Cluster {self._cluster_path()} did not start serving by the operation deadline; "
            f"last status: {cluster.status.name} ({cluster.status_message or 'no status message'})"
        )
        raise TimeoutError(message)

    async def _wait_for_deletion(self, client: ClusterManagerAsyncClient, deadline: float) -> None:
        """Poll until the cluster is deleted.

        Args:
            client: Cluster Manager client.
            deadline: Deadline on the ``time.monotonic`` clock, shared by the handler's waits.

        Raises:
            TimeoutError: If the cluster is not deleted by ``deadline``.
        """
        async for _ in poll_until(deadline):
            try:
                await client.get_cluster(request=GetClusterRequest(name=self._cluster_path()))
            except NotFound:
                return

        msg = f"Cluster {self._cluster_path()} was not deleted by the operation deadline"
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
            cluster.autopilot = Autopilot(enabled=True)
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
        """Create GKE cluster and wait until it serves, in RUNNING or DEGRADED state.

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

        cluster = await self.wait_for_serving(client, compute_operation_deadline())

        return self._build_outputs(cluster)

    async def on_observe(self) -> GKEOutputs | None:
        """Read the cluster named by ``project_id``, ``location`` and ``name``.

        Returns:
            GKEOutputs for the cluster, or None if it does not exist.
        """
        client = self._get_client()

        try:
            cluster = await client.get_cluster(request=GetClusterRequest(name=self._cluster_path()))
        except NotFound:
            return None

        return self._build_outputs(cluster)

    async def on_update(self, previous_config: GKEConfig | None) -> GKEOutputs:
        """Apply ``release_channel`` to the cluster when its live channel differs, and wait until it serves.

        Waits for the cluster's pending operations, then reads the live channel and updates it when it differs.
        Either way, waits until the cluster serves, in RUNNING or DEGRADED state.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            GKEOutputs with current cluster state.

        Raises:
            TimeoutError: If a pending operation, the release-channel update or the cluster serving
                does not finish by the operation deadline.
            RuntimeError: If GKE refuses the release-channel update because another operation runs on the cluster,
                the update fails, or the cluster is in ERROR or STOPPING state.
        """
        client = self._get_client()
        deadline = compute_operation_deadline()

        await run_cluster_operation(
            client, self.config, self.name, lambda: self.apply_release_channel(client), deadline
        )
        serving_cluster = await self.wait_for_serving(client, deadline)

        return self._build_outputs(serving_cluster)

    async def apply_release_channel(self, client: ClusterManagerAsyncClient) -> Operation | None:
        """Update the cluster to the configured ``release_channel`` when its live channel differs.

        Args:
            client: Cluster Manager client.

        Returns:
            The operation GKE started for the update, or None when the live channel already matches.
        """
        channel = ReleaseChannel.Channel[cast(str, self.config.release_channel)]
        live_cluster = await client.get_cluster(request=GetClusterRequest(name=self._cluster_path()))

        if live_cluster.release_channel.channel == channel:
            return None

        return await client.update_cluster(
            request=UpdateClusterRequest(
                name=self._cluster_path(),
                update=ClusterUpdate(desired_release_channel=ReleaseChannel(channel=channel)),
            )
        )

    async def on_delete(self) -> None:
        """Delete cluster and wait for completion.

        Idempotent: Succeeds if cluster doesn't exist.

        Raises:
            RuntimeError: If GKE refuses the delete because another operation runs on the cluster,
                or the delete operation finished with an error.
            TimeoutError: If a pending operation, the delete operation or the deletion does not finish by the
                operation deadline.
        """
        client = self._get_client()
        deadline = compute_operation_deadline()

        try:
            await run_cluster_operation(
                client,
                self.config,
                self.name,
                lambda: client.delete_cluster(request=DeleteClusterRequest(name=self._cluster_path())),
                deadline,
            )
            await self._wait_for_deletion(client, deadline)
        except NotFound:
            pass

    async def health(self) -> HealthStatus:
        """Check cluster health by querying cluster status.

        Returns:
            HealthStatus indicating cluster health.
        """
        client = self._get_client()

        try:
            cluster = await client.get_cluster(request=GetClusterRequest(name=self._cluster_path()))
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

        if status == Cluster.Status.DEGRADED:
            return HealthStatus(
                status="degraded",
                message=f"Cluster is serving but degraded: {cluster.status_message or 'no reason reported'}",
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
        """Fetch cluster logs from Cloud Logging.

        Yields:
            LogEntry objects from Cloud Logging.
        """
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
