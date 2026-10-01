"""Rollout progress and replica health of Kubernetes Deployments and StatefulSets."""

from __future__ import annotations

from typing import Any, cast

from lightkube import AsyncClient
from lightkube.resources.apps_v1 import Deployment, StatefulSet
from pragma_sdk import HealthStatus

from kubernetes_provider.polling import poll_until


ROLLOUT_POLL_INTERVAL_SECONDS = 5
ROLLOUT_TIMEOUT_SECONDS = 300


def is_rollout_complete(workload: Deployment | StatefulSet) -> bool:
    """Tell whether every replica of a workload runs its current spec and is ready.

    Args:
        workload: Deployment or StatefulSet as read from the cluster.

    Returns:
        ``True`` when the controller has observed the latest generation and the updated,
        ready and total replica counts all equal the replica count in the workload's spec.
    """
    status = workload.status

    if status is None or workload.spec is None:
        return False

    generation = (workload.metadata.generation if workload.metadata else None) or 0
    desired = workload.spec.replicas or 0
    updated = status.updatedReplicas or 0
    ready = status.readyReplicas or 0
    total = status.replicas or 0

    return (status.observedGeneration or 0) >= generation and updated == ready == total == desired


def format_rollout_progress(workload: Deployment | StatefulSet) -> str:
    """Describe how far a workload's rollout has progressed.

    Args:
        workload: Deployment or StatefulSet as read from the cluster.

    Returns:
        The generation the controller observed and the latest one, the updated, ready, total
        and desired replica counts, and the reason of a Deployment's ``Progressing`` condition
        or a StatefulSet's current and update revisions. Values the cluster has not reported
        are left out.
    """
    status = workload.status
    desired = (workload.spec.replicas if workload.spec else None) or 0

    if status is None:
        return f"no status reported yet, desired replicas {desired}"

    facts: dict[str, object] = {
        "observed generation": status.observedGeneration,
        "generation": workload.metadata.generation if workload.metadata else None,
        "updated replicas": status.updatedReplicas or 0,
        "ready replicas": status.readyReplicas or 0,
        "replicas": status.replicas or 0,
        "desired replicas": desired,
    }

    if isinstance(workload, Deployment) and workload.status is not None:
        facts["Progressing reason"] = next(
            (condition.reason for condition in workload.status.conditions or [] if condition.type == "Progressing"),
            None,
        )

    if isinstance(workload, StatefulSet) and workload.status is not None:
        facts["current revision"] = workload.status.currentRevision
        facts["update revision"] = workload.status.updateRevision

    return ", ".join(f"{label} {value}" for label, value in facts.items() if value is not None)


async def wait_for_rollout[WorkloadT: (Deployment, StatefulSet)](
    client: AsyncClient,
    workload_type: type[WorkloadT],
    name: str,
    namespace: str,
) -> WorkloadT:
    """Poll until every replica of a workload runs its current spec and is ready.

    Args:
        client: Lightkube client for the target cluster.
        workload_type: ``Deployment`` or ``StatefulSet``.
        name: Workload name.
        namespace: Namespace of the workload.

    Returns:
        The workload as read once its rollout is complete.

    Raises:
        ApiError: If reading the workload fails, including when it no longer exists.
        TimeoutError: If the rollout is not complete within ``ROLLOUT_TIMEOUT_SECONDS``; the
            message names the workload and its rollout progress.
    """

    def describe_timeout(workload: WorkloadT) -> str:
        return (
            f"{workload_type.__name__} {namespace}/{name} did not become ready within {ROLLOUT_TIMEOUT_SECONDS}s: "
            f"{format_rollout_progress(workload)}"
        )

    return await poll_until(
        lambda: client.get(cast(Any, workload_type), name, namespace=namespace),
        is_rollout_complete,
        describe_timeout,
        ROLLOUT_TIMEOUT_SECONDS,
        ROLLOUT_POLL_INTERVAL_SECONDS,
    )


def build_replica_health(workload: Deployment | StatefulSet) -> HealthStatus:
    """Classify a workload's health by its ready replicas against the replicas in its spec.

    Args:
        workload: Deployment or StatefulSet as read from the cluster.

    Returns:
        Healthy when every desired replica is ready, degraded when only some are, and
        unhealthy when none is or no replica is desired.
    """
    ready = (workload.status.readyReplicas if workload.status else None) or 0
    desired = (workload.spec.replicas if workload.spec else None) or 0
    details = {"ready_replicas": ready, "desired_replicas": desired}

    if ready >= desired and desired > 0:
        return HealthStatus(status="healthy", message=f"All {ready} replicas ready", details=details)

    if ready > 0:
        return HealthStatus(status="degraded", message=f"{ready}/{desired} replicas ready", details=details)

    return HealthStatus(status="unhealthy", message=f"No replicas ready (desired: {desired})", details=details)
