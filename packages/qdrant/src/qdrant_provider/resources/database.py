"""Qdrant Database resource - deploys Qdrant to GKE via Helm."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field as PydanticField
from pragma_sdk import Config, Dependency, Outputs, Resource

if TYPE_CHECKING:
    from gcp_provider import GKE


class StorageConfig(BaseModel):
    """Storage configuration for Qdrant database.

    Attributes:
        size: Persistent volume size (e.g., "10Gi").
        class_: Kubernetes storage class name.
    """

    model_config = {"populate_by_name": True}

    size: str = "10Gi"
    class_: str = PydanticField(default="standard-rwo", alias="class")


class ResourceConfig(BaseModel):
    """Resource limits for Qdrant database.

    Attributes:
        memory: Memory limit (e.g., "2Gi").
        cpu: CPU limit (e.g., "1" or "500m").
    """

    memory: str = "2Gi"
    cpu: str = "1"


class DatabaseConfig(Config):
    """Configuration for a Qdrant database deployment.

    Attributes:
        cluster: GKE cluster dependency providing Kubernetes credentials.
        replicas: Number of Qdrant replicas (StatefulSet pods).
        storage: Persistent storage configuration.
        resources: CPU and memory limits.
    """

    cluster: Dependency["GKE"]
    replicas: int = 1
    storage: StorageConfig | None = None
    resources: ResourceConfig | None = None


class DatabaseOutputs(Outputs):
    """Outputs from Qdrant database deployment.

    Attributes:
        url: HTTP endpoint for Qdrant REST API (in-cluster URL).
        grpc_url: gRPC endpoint for Qdrant (in-cluster URL).
        ready: Whether the StatefulSet is ready.
    """

    url: str
    grpc_url: str
    ready: bool


# Polling configuration for StatefulSet readiness
_POLL_INTERVAL_SECONDS = 5
_MAX_POLL_ATTEMPTS = 60  # 60 * 5s = 5 minutes max wait


class Database(Resource[DatabaseConfig, DatabaseOutputs]):
    """Qdrant database deployed to GKE via Helm.

    Deploys the official Qdrant Helm chart to a GKE cluster and waits
    for the StatefulSet to become ready. Uses cluster credentials from
    the GKE dependency to authenticate with the Kubernetes API.

    Lifecycle:
        - on_create: Install Helm chart, wait for ready
        - on_update: Upgrade Helm release with new values
        - on_delete: Uninstall Helm release
    """

    provider: ClassVar[str] = "qdrant"
    resource: ClassVar[str] = "database"

    def _get_release_name(self) -> str:
        """Get Helm release name based on resource name."""
        return f"qdrant-{self.name}"

    def _get_namespace(self) -> str:
        """Get Kubernetes namespace."""
        return "default"

    async def _get_kubeconfig(self) -> str:
        """Build kubeconfig from GKE cluster credentials.

        Returns:
            Path to temporary kubeconfig file.
        """
        cluster = await self.config.cluster.resolve()
        outputs = cluster.outputs
        if outputs is None:
            msg = "GKE cluster outputs not available"
            raise RuntimeError(msg)

        kubeconfig = f"""
apiVersion: v1
kind: Config
clusters:
- name: gke-cluster
  cluster:
    server: https://{outputs.endpoint}
    certificate-authority-data: {outputs.cluster_ca_certificate}
contexts:
- name: gke-context
  context:
    cluster: gke-cluster
    user: gke-user
current-context: gke-context
users:
- name: gke-user
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1beta1
      command: gke-gcloud-auth-plugin
      installHint: Install gke-gcloud-auth-plugin for kubectl auth
      provideClusterInfo: true
"""
        return kubeconfig

    def _build_helm_values(self) -> dict:
        """Build Helm values from config."""
        values: dict = {
            "replicaCount": self.config.replicas,
        }

        if self.config.storage:
            values["persistence"] = {
                "size": self.config.storage.size,
                "storageClassName": self.config.storage.class_,
            }

        if self.config.resources:
            values["resources"] = {
                "limits": {
                    "memory": self.config.resources.memory,
                    "cpu": self.config.resources.cpu,
                },
                "requests": {
                    "memory": self.config.resources.memory,
                    "cpu": self.config.resources.cpu,
                },
            }

        return values

    async def _run_command(self, cmd: list[str], kubeconfig_path: str, name: str) -> subprocess.CompletedProcess:
        """Run a CLI command with kubeconfig.

        Args:
            cmd: Full command with arguments.
            kubeconfig_path: Path to kubeconfig file.
            name: Command name for error messages.

        Returns:
            Completed process result.

        Raises:
            RuntimeError: If command fails.
        """
        env = {"KUBECONFIG": kubeconfig_path}

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env={**os.environ, **env},
                check=False,
            ),
        )

        if result.returncode != 0:
            msg = f"{name} command failed: {result.stderr}"
            raise RuntimeError(msg)

        return result

    async def _run_helm(self, args: list[str], kubeconfig_path: str) -> subprocess.CompletedProcess:
        """Run helm command with kubeconfig."""
        return await self._run_command(["helm", *args], kubeconfig_path, "Helm")

    async def _wait_for_ready(self, kubeconfig_path: str) -> bool:
        """Wait for StatefulSet to be ready.

        Args:
            kubeconfig_path: Path to kubeconfig file.

        Returns:
            True when StatefulSet is ready.

        Raises:
            TimeoutError: If StatefulSet doesn't become ready in time.
        """
        namespace = self._get_namespace()
        release_name = self._get_release_name()

        for _ in range(_MAX_POLL_ATTEMPTS):
            try:
                result = await self._run_kubectl(
                    [
                        "get",
                        "statefulset",
                        release_name,
                        "-n",
                        namespace,
                        "-o",
                        "jsonpath={.status.readyReplicas}",
                    ],
                    kubeconfig_path,
                )
                ready_replicas = int(result.stdout.strip() or "0")
                if ready_replicas >= self.config.replicas:
                    return True
            except (RuntimeError, ValueError):
                pass  # StatefulSet may not exist yet or parsing failed

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        msg = f"StatefulSet {release_name} did not become ready within {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS}s"
        raise TimeoutError(msg)

    async def _run_kubectl(self, args: list[str], kubeconfig_path: str) -> subprocess.CompletedProcess:
        """Run kubectl command with kubeconfig."""
        return await self._run_command(["kubectl", *args], kubeconfig_path, "kubectl")

    async def _helm_install(self, kubeconfig_path: str, values_path: str) -> None:
        """Install or upgrade Qdrant Helm chart.

        Args:
            kubeconfig_path: Path to kubeconfig file.
            values_path: Path to Helm values file.
        """
        release_name = self._get_release_name()
        namespace = self._get_namespace()

        await self._run_helm(
            [
                "upgrade",
                "--install",
                release_name,
                "qdrant/qdrant",
                "--namespace",
                namespace,
                "--create-namespace",
                "--values",
                values_path,
                "--wait",
                "--timeout",
                "5m",
            ],
            kubeconfig_path,
        )

    async def _helm_uninstall(self, kubeconfig_path: str) -> None:
        """Uninstall Qdrant Helm chart.

        Args:
            kubeconfig_path: Path to kubeconfig file.
        """
        release_name = self._get_release_name()
        namespace = self._get_namespace()

        try:
            await self._run_helm(
                [
                    "uninstall",
                    release_name,
                    "--namespace",
                    namespace,
                ],
                kubeconfig_path,
            )
        except RuntimeError as e:
            # Ignore if release doesn't exist
            if "not found" not in str(e).lower():
                raise

    async def _ensure_helm_repo(self, kubeconfig_path: str) -> None:
        """Ensure Qdrant Helm repo is added.

        Args:
            kubeconfig_path: Path to kubeconfig file.
        """
        try:
            await self._run_helm(
                ["repo", "add", "qdrant", "https://qdrant.github.io/qdrant-helm"],
                kubeconfig_path,
            )
        except RuntimeError:
            pass  # Repo may already exist

        await self._run_helm(["repo", "update"], kubeconfig_path)

    def _build_outputs(self) -> DatabaseOutputs:
        """Build outputs with in-cluster URLs."""
        release_name = self._get_release_name()
        namespace = self._get_namespace()

        # Qdrant service follows release name convention
        service_name = release_name
        url = f"http://{service_name}.{namespace}.svc.cluster.local:6333"
        grpc_url = f"http://{service_name}.{namespace}.svc.cluster.local:6334"

        return DatabaseOutputs(
            url=url,
            grpc_url=grpc_url,
            ready=True,
        )

    async def on_create(self) -> DatabaseOutputs:
        """Deploy Qdrant to GKE via Helm.

        Idempotent: Uses helm upgrade --install which handles both
        initial install and upgrades.

        Returns:
            DatabaseOutputs with in-cluster URLs.
        """
        kubeconfig = await self._get_kubeconfig()
        values = self._build_helm_values()

        with tempfile.TemporaryDirectory() as tmpdir:
            kubeconfig_path = Path(tmpdir) / "kubeconfig"
            values_path = Path(tmpdir) / "values.json"

            kubeconfig_path.write_text(kubeconfig)
            values_path.write_text(json.dumps(values))

            await self._ensure_helm_repo(str(kubeconfig_path))
            await self._helm_install(str(kubeconfig_path), str(values_path))
            await self._wait_for_ready(str(kubeconfig_path))

        return self._build_outputs()

    async def on_update(self, previous_config: DatabaseConfig) -> DatabaseOutputs:
        """Update Qdrant deployment via Helm upgrade.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            DatabaseOutputs with in-cluster URLs.
        """
        # Check for immutable field changes
        # Cluster change would require migration which we don't support
        if previous_config.cluster.id != self.config.cluster.id:
            msg = "Cannot change cluster; delete and recreate resource"
            raise ValueError(msg)

        # If nothing changed, return existing outputs
        if self.outputs is not None:
            previous_dict = previous_config.model_dump(exclude={"cluster"})
            current_dict = self.config.model_dump(exclude={"cluster"})
            if previous_dict == current_dict:
                return self.outputs

        kubeconfig = await self._get_kubeconfig()
        values = self._build_helm_values()

        with tempfile.TemporaryDirectory() as tmpdir:
            kubeconfig_path = Path(tmpdir) / "kubeconfig"
            values_path = Path(tmpdir) / "values.json"

            kubeconfig_path.write_text(kubeconfig)
            values_path.write_text(json.dumps(values))

            await self._ensure_helm_repo(str(kubeconfig_path))
            await self._helm_install(str(kubeconfig_path), str(values_path))
            await self._wait_for_ready(str(kubeconfig_path))

        return self._build_outputs()

    async def on_delete(self) -> None:
        """Uninstall Qdrant Helm release.

        Idempotent: Succeeds if release doesn't exist.
        """
        kubeconfig = await self._get_kubeconfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            kubeconfig_path = Path(tmpdir) / "kubeconfig"
            kubeconfig_path.write_text(kubeconfig)

            await self._helm_uninstall(str(kubeconfig_path))
