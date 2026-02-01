"""Tests for Kubernetes StatefulSet resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from lightkube import ApiError
from pragma_sdk import Dependency, LifecycleState

from kubernetes_provider import StatefulSet, StatefulSetConfig, StatefulSetOutputs
from kubernetes_provider.resources.statefulset import (
    ContainerConfig,
    VolumeClaimTemplateConfig,
    VolumeMountConfig,
)


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_statefulset_with_mocked_dependency(
    name: str,
    namespace: str = "default",
    replicas: int = 1,
    service_name: str = "test-svc",
    containers: list[ContainerConfig] | None = None,
    mock_gke_cluster: Any = None,
    outputs: StatefulSetOutputs | None = None,
) -> StatefulSet:
    dep = Dependency(provider="gcp", resource="gke", name="test-cluster")

    config = StatefulSetConfig(
        cluster=dep,
        namespace=namespace,
        replicas=replicas,
        service_name=service_name,
        containers=containers or [ContainerConfig(name="app", image="nginx:latest")],
    )

    if mock_gke_cluster:
        config.cluster._resolved = mock_gke_cluster

    return StatefulSet(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


async def test_create_statefulset_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_create applies statefulset and waits for ready."""
    mock_sts = mocker.MagicMock()
    mock_sts.metadata.name = "test-sts"
    mock_sts.metadata.namespace = "default"
    mock_sts.spec.replicas = 1
    mock_sts.spec.serviceName = "test-svc"
    mock_sts.status.readyReplicas = 1
    mock_lightkube_client.get.return_value = mock_sts

    sts = create_statefulset_with_mocked_dependency(
        name="test-sts",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await sts.on_create()

    assert result.name == "test-sts"
    assert result.ready_replicas == 1
    mock_lightkube_client.apply.assert_called_once()


async def test_create_statefulset_with_pvc(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_create handles volume claim templates."""
    mock_sts = mocker.MagicMock()
    mock_sts.metadata.name = "db"
    mock_sts.metadata.namespace = "default"
    mock_sts.spec.replicas = 1
    mock_sts.spec.serviceName = "db-svc"
    mock_sts.status.readyReplicas = 1
    mock_lightkube_client.get.return_value = mock_sts

    dep = Dependency(provider="gcp", resource="gke", name="test-cluster")
    config = StatefulSetConfig(
        cluster=dep,
        namespace="default",
        replicas=1,
        service_name="db-svc",
        containers=[
            ContainerConfig(
                name="db",
                image="postgres:15",
                volume_mounts=[VolumeMountConfig(name="data", mount_path="/var/lib/postgresql/data")],
            )
        ],
        volume_claim_templates=[
            VolumeClaimTemplateConfig(
                name="data",
                storage_class="standard-rwo",
                storage="10Gi",
            )
        ],
    )
    config.cluster._resolved = mock_gke_cluster

    sts = StatefulSet(
        name="db",
        config=config,
        lifecycle_state=LifecycleState.PROCESSING,
    )

    result = await sts.on_create()

    assert result.name == "db"


async def test_create_statefulset_waits_for_ready(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_create polls until replicas are ready."""
    mock_sts_pending = mocker.MagicMock()
    mock_sts_pending.metadata.name = "test-sts"
    mock_sts_pending.metadata.namespace = "default"
    mock_sts_pending.spec.replicas = 2
    mock_sts_pending.spec.serviceName = "test-svc"
    mock_sts_pending.status.readyReplicas = 1

    mock_sts_ready = mocker.MagicMock()
    mock_sts_ready.metadata.name = "test-sts"
    mock_sts_ready.metadata.namespace = "default"
    mock_sts_ready.spec.replicas = 2
    mock_sts_ready.spec.serviceName = "test-svc"
    mock_sts_ready.status.readyReplicas = 2

    mock_lightkube_client.get.side_effect = [mock_sts_pending, mock_sts_ready]

    sts = create_statefulset_with_mocked_dependency(
        name="test-sts",
        replicas=2,
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await sts.on_create()

    assert result.ready_replicas == 2
    assert mock_lightkube_client.get.call_count == 2


async def test_update_statefulset_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_update applies updated statefulset."""
    mock_sts = mocker.MagicMock()
    mock_sts.metadata.name = "test-sts"
    mock_sts.metadata.namespace = "default"
    mock_sts.spec.replicas = 2
    mock_sts.spec.serviceName = "test-svc"
    mock_sts.status.readyReplicas = 2
    mock_lightkube_client.get.return_value = mock_sts

    sts = create_statefulset_with_mocked_dependency(
        name="test-sts",
        replicas=2,
        mock_gke_cluster=mock_gke_cluster,
    )

    previous = StatefulSetConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        namespace="default",
        replicas=1,
        service_name="test-svc",
        containers=[ContainerConfig(name="app", image="nginx:1.0")],
    )

    result = await sts.on_update(previous)

    assert result.replicas == 2


async def test_update_rejects_service_name_change(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_update rejects service_name changes."""
    sts = create_statefulset_with_mocked_dependency(
        name="test-sts",
        service_name="svc-b",
        mock_gke_cluster=mock_gke_cluster,
    )

    previous = StatefulSetConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        namespace="default",
        replicas=1,
        service_name="svc-a",
        containers=[ContainerConfig(name="app", image="nginx")],
    )

    with pytest.raises(ValueError, match="service_name"):
        await sts.on_update(previous)


async def test_delete_statefulset_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_delete removes statefulset."""
    sts = create_statefulset_with_mocked_dependency(
        name="test-sts",
        mock_gke_cluster=mock_gke_cluster,
    )

    await sts.on_delete()

    mock_lightkube_client.delete.assert_called_once()


async def test_delete_statefulset_idempotent(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_delete succeeds when statefulset doesn't exist."""
    error = ApiError(response=mocker.MagicMock())
    error.status = mocker.MagicMock(code=404)
    mock_lightkube_client.delete.side_effect = error

    sts = create_statefulset_with_mocked_dependency(
        name="gone",
        mock_gke_cluster=mock_gke_cluster,
    )

    await sts.on_delete()


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert StatefulSet.provider == "kubernetes"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert StatefulSet.resource == "statefulset"


async def test_health_all_replicas_ready(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns healthy when all replicas ready."""
    mock_sts = mocker.MagicMock()
    mock_sts.spec.replicas = 3
    mock_sts.status.readyReplicas = 3
    mock_lightkube_client.get.return_value = mock_sts

    sts = create_statefulset_with_mocked_dependency(
        name="test-sts",
        replicas=3,
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await sts.health()

    assert result.status == "healthy"
    assert result.details["ready_replicas"] == 3


async def test_health_partial_replicas(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns degraded when some replicas ready."""
    mock_sts = mocker.MagicMock()
    mock_sts.spec.replicas = 3
    mock_sts.status.readyReplicas = 1
    mock_lightkube_client.get.return_value = mock_sts

    sts = create_statefulset_with_mocked_dependency(
        name="test-sts",
        replicas=3,
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await sts.health()

    assert result.status == "degraded"


async def test_health_no_replicas(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns unhealthy when no replicas ready."""
    mock_sts = mocker.MagicMock()
    mock_sts.spec.replicas = 3
    mock_sts.status.readyReplicas = 0
    mock_lightkube_client.get.return_value = mock_sts

    sts = create_statefulset_with_mocked_dependency(
        name="test-sts",
        replicas=3,
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await sts.health()

    assert result.status == "unhealthy"


async def test_health_not_found(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns unhealthy when statefulset not found."""
    error = ApiError(response=mocker.MagicMock())
    error.status = mocker.MagicMock(code=404)
    mock_lightkube_client.get.side_effect = error

    sts = create_statefulset_with_mocked_dependency(
        name="missing",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await sts.health()

    assert result.status == "unhealthy"
    assert "not found" in result.message
