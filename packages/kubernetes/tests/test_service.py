"""Tests for Kubernetes Service resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from lightkube import ApiError
from pragma_sdk import Dependency, LifecycleState

from kubernetes_provider import Service, ServiceConfig, ServiceOutputs
from kubernetes_provider.resources.service import PortConfig


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_service_with_mocked_dependency(
    name: str,
    namespace: str = "default",
    service_type: str = "ClusterIP",
    selector: dict | None = None,
    ports: list[PortConfig] | None = None,
    mock_gke_cluster: Any = None,
    outputs: ServiceOutputs | None = None,
) -> Service:
    dep = Dependency(provider="gcp", resource="gke", name="test-cluster")

    config = ServiceConfig(
        cluster=dep,
        namespace=namespace,
        type=service_type,
        selector=selector or {"app": "test"},
        ports=ports or [PortConfig(port=80, target_port=8080)],
    )

    if mock_gke_cluster:
        config.cluster._resolved = mock_gke_cluster

    return Service(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


async def test_create_service_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_create applies service and returns outputs."""
    mock_service = mocker.MagicMock()
    mock_service.metadata.name = "test-service"
    mock_service.metadata.namespace = "default"
    mock_service.spec.clusterIP = "10.0.0.100"
    mock_service.spec.type = "ClusterIP"
    mock_lightkube_client.get.return_value = mock_service

    svc = create_service_with_mocked_dependency(
        name="test-service",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await svc.on_create()

    assert result.name == "test-service"
    assert result.cluster_ip == "10.0.0.100"
    mock_lightkube_client.apply.assert_called_once()


async def test_create_headless_service(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_create handles headless service type."""
    mock_service = mocker.MagicMock()
    mock_service.metadata.name = "headless-svc"
    mock_service.metadata.namespace = "default"
    mock_service.spec.clusterIP = "None"
    mock_service.spec.type = "ClusterIP"
    mock_lightkube_client.get.return_value = mock_service

    svc = create_service_with_mocked_dependency(
        name="headless-svc",
        service_type="Headless",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await svc.on_create()

    assert result.cluster_ip == "None"


async def test_update_service_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_update applies updated service."""
    mock_service = mocker.MagicMock()
    mock_service.metadata.name = "test-service"
    mock_service.metadata.namespace = "default"
    mock_service.spec.clusterIP = "10.0.0.100"
    mock_service.spec.type = "ClusterIP"
    mock_lightkube_client.get.return_value = mock_service

    svc = create_service_with_mocked_dependency(
        name="test-service",
        ports=[PortConfig(port=80), PortConfig(port=443)],
        mock_gke_cluster=mock_gke_cluster,
    )

    previous = ServiceConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        namespace="default",
        type="ClusterIP",
        selector={"app": "test"},
        ports=[PortConfig(port=80)],
    )

    result = await svc.on_update(previous)

    assert result.name == "test-service"
    mock_lightkube_client.apply.assert_called_once()


async def test_update_rejects_namespace_change(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_update rejects namespace changes."""
    svc = create_service_with_mocked_dependency(
        name="test-service",
        namespace="ns-b",
        mock_gke_cluster=mock_gke_cluster,
    )

    previous = ServiceConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        namespace="ns-a",
        type="ClusterIP",
        selector={"app": "test"},
        ports=[PortConfig(port=80)],
    )

    with pytest.raises(ValueError, match="namespace"):
        await svc.on_update(previous)


async def test_delete_service_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_delete removes service."""
    svc = create_service_with_mocked_dependency(
        name="test-service",
        mock_gke_cluster=mock_gke_cluster,
    )

    await svc.on_delete()

    mock_lightkube_client.delete.assert_called_once()


async def test_delete_service_idempotent(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_delete succeeds when service doesn't exist."""
    error = ApiError(response=mocker.MagicMock())
    error.status = mocker.MagicMock(code=404)
    mock_lightkube_client.delete.side_effect = error

    svc = create_service_with_mocked_dependency(
        name="gone",
        mock_gke_cluster=mock_gke_cluster,
    )

    await svc.on_delete()


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert Service.provider == "kubernetes"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Service.resource == "service"


async def test_health_with_endpoints(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns healthy when service has endpoints."""
    mock_svc = mocker.MagicMock()
    mock_svc.metadata.name = "test-svc"
    mock_svc.metadata.namespace = "default"

    mock_endpoints = mocker.MagicMock()
    mock_subset = mocker.MagicMock()
    mock_subset.addresses = [mocker.MagicMock(), mocker.MagicMock()]
    mock_endpoints.subsets = [mock_subset]

    mock_lightkube_client.get.side_effect = [mock_svc, mock_endpoints]

    svc = create_service_with_mocked_dependency(
        name="test-svc",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await svc.health()

    assert result.status == "healthy"
    assert result.details["endpoint_count"] == 2


async def test_health_no_endpoints(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns degraded when service has no endpoints."""
    mock_svc = mocker.MagicMock()
    mock_svc.metadata.name = "test-svc"
    mock_svc.metadata.namespace = "default"

    mock_endpoints = mocker.MagicMock()
    mock_endpoints.subsets = None

    mock_lightkube_client.get.side_effect = [mock_svc, mock_endpoints]

    svc = create_service_with_mocked_dependency(
        name="test-svc",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await svc.health()

    assert result.status == "degraded"


async def test_health_service_not_found(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns unhealthy when service not found."""
    error = ApiError(response=mocker.MagicMock())
    error.status = mocker.MagicMock(code=404)
    mock_lightkube_client.get.side_effect = error

    svc = create_service_with_mocked_dependency(
        name="missing",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await svc.health()

    assert result.status == "unhealthy"
    assert "not found" in result.message


async def test_logs_returns_message(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """logs() yields info message about services not having logs."""
    svc = create_service_with_mocked_dependency(
        name="test-svc",
        mock_gke_cluster=mock_gke_cluster,
    )

    entries = []
    async for entry in svc.logs():
        entries.append(entry)

    assert len(entries) == 1
    assert entries[0].level == "info"
    assert "do not produce logs" in entries[0].message
