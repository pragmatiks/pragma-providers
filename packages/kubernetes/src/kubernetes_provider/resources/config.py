"""Kubernetes config resource providing authenticated cluster access.

Decouples kubernetes workload resources from the specific cluster provider
(GKE, EKS, AKS, self-managed) by presenting a single unified dependency for
all downstream kubernetes resources. Supports three authentication modes:

- ``in_cluster``: Use the pod's ServiceAccount credentials. This is what
  production uses -- the platform API pod and its runner pods all run in
  the same cluster as the workloads they manage.
- ``gke_cluster``: Build a kubeconfig from a GKE cluster's endpoint, CA
  certificate, and a bearer token obtained from GCP service account
  credentials.
- ``kubeconfig_file``: Read a kubeconfig YAML file from disk.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from gcp_provider import GKE
from lightkube import AsyncClient, KubeConfig
from lightkube.core.exceptions import ConfigError
from pragma_sdk import (
    Config,
    HealthStatus,
    ImmutableDependency,
    ImmutableField,
    LogEntry,
    Outputs,
    Resource,
)
from pydantic import model_validator

from kubernetes_provider.client import build_kubeconfig_from_gke


_SERVICE_ACCOUNT_PATH = "/var/run/secrets/kubernetes.io/serviceaccount"
_KUBECONFIG_ALLOWED_ROOT = Path("/etc/pragma-kubeconfig")


def _validate_kubeconfig_path(raw_path: str) -> Path:
    """Validate a kubeconfig file path against the allowed root.

    Kubeconfig files must live under :data:`_KUBECONFIG_ALLOWED_ROOT` to
    prevent the config resource from reading arbitrary files on disk (for
    example, user-level kubeconfigs or tokens mounted for other workloads).
    Symlinks are rejected outright: they could traverse outside the
    allowed root even if :meth:`Path.resolve` maps them back inside.

    Args:
        raw_path: User-supplied kubeconfig path from the resource config.

    Returns:
        Resolved absolute path to the kubeconfig file.

    Raises:
        ValueError: If the path is relative, symlinked, or outside the
            allowed root.
        FileNotFoundError: If the resolved path does not exist.
    """
    path = Path(raw_path)

    if not path.is_absolute():
        msg = f"kubeconfig_path must be absolute and live under {_KUBECONFIG_ALLOWED_ROOT}: {raw_path}"
        raise ValueError(msg)

    if path.is_symlink():
        msg = f"kubeconfig_path must not be a symlink: {raw_path}"
        raise ValueError(msg)

    resolved = path.resolve()
    resolved_root = _KUBECONFIG_ALLOWED_ROOT.resolve()

    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        msg = f"kubeconfig_path must live under {_KUBECONFIG_ALLOWED_ROOT}: {raw_path}"
        raise ValueError(msg) from exc

    if not resolved.is_file():
        msg = f"kubeconfig file not found: {resolved}"
        raise FileNotFoundError(msg)

    return resolved


class ConfigConfig(Config):
    """Configuration for a Kubernetes config resource.

    Attributes:
        mode: Authentication mode. One of ``in_cluster``, ``gke_cluster``,
            or ``kubeconfig_file``. Immutable after creation.
        cluster: GKE cluster dependency. Required when ``mode=gke_cluster``,
            must be absent for other modes.
        kubeconfig_path: Absolute path to a kubeconfig YAML file under
            ``/etc/pragma-kubeconfig``. Required when ``mode=kubeconfig_file``,
            must be absent for other modes. Immutable after creation.
    """

    mode: ImmutableField[Literal["in_cluster", "gke_cluster", "kubeconfig_file"]]
    cluster: ImmutableDependency[GKE] | None = None
    kubeconfig_path: ImmutableField[str] | None = None

    @model_validator(mode="after")
    def validate_mode_fields(self) -> ConfigConfig:
        """Validate that only the mode's fields are present.

        Returns:
            Self after validation.

        Raises:
            ValueError: If a required field for the chosen mode is missing
                or an irrelevant field for the chosen mode is set.
        """
        if self.mode == "in_cluster":
            if self.cluster is not None:
                msg = "cluster must not be set when mode=in_cluster"
                raise ValueError(msg)

            if self.kubeconfig_path is not None:
                msg = "kubeconfig_path must not be set when mode=in_cluster"
                raise ValueError(msg)

        if self.mode == "gke_cluster":
            if self.cluster is None:
                msg = "cluster is required when mode=gke_cluster"
                raise ValueError(msg)

            if self.kubeconfig_path is not None:
                msg = "kubeconfig_path must not be set when mode=gke_cluster"
                raise ValueError(msg)

        if self.mode == "kubeconfig_file":
            if not self.kubeconfig_path:
                msg = "kubeconfig_path is required when mode=kubeconfig_file"
                raise ValueError(msg)

            if self.cluster is not None:
                msg = "cluster must not be set when mode=kubeconfig_file"
                raise ValueError(msg)

        return self


class ConfigOutputs(Outputs):
    """Outputs from Kubernetes config creation.

    Attributes:
        mode: Authentication mode in use.
    """

    mode: str


class KubernetesConfig(Resource[ConfigConfig, ConfigOutputs]):
    """Kubernetes config resource exposing an authenticated cluster client.

    Produces a lightkube ``AsyncClient`` that downstream kubernetes resources
    (namespace, deployment, service, etc.) consume via a dependency. By moving
    client construction into its own resource, workload resources no longer
    couple to GKE (or any other cluster provider) directly and can target
    existing clusters without owning their lifecycle.

    The handler validates that the configured mode can produce a client but
    does not make any external API calls -- it's pure validation / file IO.

    Lifecycle:
        - on_create: Validate the configured mode.
        - on_update: Re-validate (mode is immutable; other fields may change).
        - on_delete: No-op. Config resources own no external state.
    """

    async def _validate(self) -> None:
        """Validate that the configured mode can produce a client.

        Raises:
            RuntimeError: If GKE dependency outputs are unavailable or the
                pod is not running with a service account (in_cluster mode).
        """
        if self.config.mode == "in_cluster":
            try:
                KubeConfig.from_service_account()
            except ConfigError as exc:
                msg = (
                    "in_cluster mode requires a pod-mounted service account at "
                    f"{_SERVICE_ACCOUNT_PATH}; no credentials found"
                )
                raise RuntimeError(msg) from exc
            return

        if self.config.mode == "gke_cluster":
            assert self.config.cluster is not None
            cluster = await self.config.cluster.resolve()

            if cluster.outputs is None:
                msg = "GKE cluster outputs not available"
                raise RuntimeError(msg)
            return

        if self.config.mode == "kubeconfig_file":
            assert self.config.kubeconfig_path is not None
            resolved = _validate_kubeconfig_path(self.config.kubeconfig_path)

            KubeConfig.from_file(str(resolved))
            return

    async def create_client(self) -> AsyncClient:
        """Build a lightkube AsyncClient for the configured auth mode.

        Called by downstream kubernetes resources after resolving this
        resource via ``await self.config.config.resolve()`` and then invoking
        ``await resolved.create_client()``.

        Returns:
            Configured lightkube async client.

        Raises:
            RuntimeError: If the mode requires a dependency or file that
                is not available, or if in_cluster mode is selected but no
                pod service account credentials are mounted.
        """
        if self.config.mode == "in_cluster":
            try:
                kubeconfig = KubeConfig.from_service_account()
            except ConfigError as exc:
                msg = (
                    "in_cluster mode requires a pod-mounted service account at "
                    f"{_SERVICE_ACCOUNT_PATH}; no credentials found"
                )
                raise RuntimeError(msg) from exc
            return AsyncClient(config=kubeconfig)

        if self.config.mode == "gke_cluster":
            assert self.config.cluster is not None
            cluster = await self.config.cluster.resolve()

            if cluster.outputs is None:
                msg = "GKE cluster outputs not available"
                raise RuntimeError(msg)

            kubeconfig = build_kubeconfig_from_gke(cluster.outputs, cluster.config.credentials)
            return AsyncClient(config=kubeconfig)

        if self.config.mode == "kubeconfig_file":
            assert self.config.kubeconfig_path is not None
            resolved = _validate_kubeconfig_path(self.config.kubeconfig_path)
            kubeconfig = KubeConfig.from_file(str(resolved))
            return AsyncClient(config=kubeconfig)

        msg = f"Unknown mode: {self.config.mode}"
        raise RuntimeError(msg)

    @asynccontextmanager
    async def build_client(self) -> AsyncIterator[AsyncClient]:
        """Yield a lightkube AsyncClient for the configured auth mode.

        Downstream kubernetes resources should prefer this over
        :meth:`create_client` so the client is always closed at the end of
        the block. Resolves the caller from this resource via
        ``await self.config.config.resolve()``.

        Yields:
            Configured lightkube async client, closed on exit.
        """
        client = await self.create_client()

        try:
            yield client
        finally:
            await client.close()

    async def on_create(self) -> ConfigOutputs:
        """Validate the cluster config.

        Returns:
            ConfigOutputs with the mode.
        """
        await self._validate()

        return ConfigOutputs(mode=self.config.mode)

    async def on_update(self, previous_config: ConfigConfig) -> ConfigOutputs:
        """Re-validate the cluster config.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            ConfigOutputs with the mode.
        """
        await self._validate()

        return ConfigOutputs(mode=self.config.mode)

    async def on_delete(self) -> None:
        """No external state to clean up."""
        return

    @classmethod
    def upgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs

    @classmethod
    def downgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs

    async def health(self) -> HealthStatus:
        """Check config health by attempting to build a client.

        Returns:
            HealthStatus indicating whether the config can produce a client.
        """
        try:
            async with self.build_client():
                pass
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(
                status="unhealthy",
                message=f"Failed to build client: {exc}",
                details={"mode": self.config.mode},
            )

        return HealthStatus(
            status="healthy",
            message=f"Config ready (mode={self.config.mode})",
            details={"mode": self.config.mode},
        )

    async def logs(
        self,
        since: datetime | None = None,
        tail: int = 100,
    ) -> AsyncIterator[LogEntry]:
        """Config resources do not produce logs.

        Args:
            since: Ignored.
            tail: Ignored.

        Yields:
            A single informational LogEntry.
        """
        yield LogEntry(
            timestamp=datetime.now(UTC),
            level="info",
            message="Config resources do not produce logs",
        )
        return
