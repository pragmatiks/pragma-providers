"""Tests for Kubernetes ConfigMap resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from lightkube import ApiError
from pragma_sdk import Dependency, LifecycleState

from kubernetes_provider import ConfigMap, ConfigMapConfig, ConfigMapOutputs


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_configmap_with_mocked_dependency(
    name: str,
    namespace: str = "default",
    data: dict | None = None,
    mock_gke_cluster: Any = None,
    outputs: ConfigMapOutputs | None = None,
) -> ConfigMap:
    dep = Dependency(provider="gcp", resource="gke", name="test-cluster")

    config = ConfigMapConfig(
        cluster=dep,
        namespace=namespace,
        data=data or {"key": "value"},
    )

    if mock_gke_cluster:
        config.cluster._resolved = mock_gke_cluster

    return ConfigMap(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


async def test_create_configmap_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_create applies configmap and returns outputs."""
    cm = create_configmap_with_mocked_dependency(
        name="test-cm",
        data={"key": "value"},
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await cm.on_create()

    assert result.name == "test-cm"
    assert result.namespace == "default"
    assert result.data == {"key": "value"}
    mock_lightkube_client.apply.assert_called_once()


async def test_update_configmap_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_update applies updated configmap."""
    cm = create_configmap_with_mocked_dependency(
        name="test-cm",
        data={"key": "new"},
        mock_gke_cluster=mock_gke_cluster,
    )

    previous = ConfigMapConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        namespace="default",
        data={"key": "old"},
    )

    result = await cm.on_update(previous)

    assert result.name == "test-cm"
    mock_lightkube_client.apply.assert_called_once()


async def test_update_rejects_namespace_change(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_update rejects namespace changes."""
    cm = create_configmap_with_mocked_dependency(
        name="test-cm",
        namespace="ns-b",
        mock_gke_cluster=mock_gke_cluster,
    )

    previous = ConfigMapConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        namespace="ns-a",
        data={"key": "value"},
    )

    with pytest.raises(ValueError, match="namespace"):
        await cm.on_update(previous)


async def test_delete_configmap_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_delete removes configmap."""
    cm = create_configmap_with_mocked_dependency(
        name="test-cm",
        mock_gke_cluster=mock_gke_cluster,
    )

    await cm.on_delete()

    mock_lightkube_client.delete.assert_called_once()


async def test_delete_configmap_idempotent(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_delete succeeds when configmap doesn't exist."""
    error = ApiError(response=mocker.MagicMock())
    error.status = mocker.MagicMock(code=404)
    mock_lightkube_client.delete.side_effect = error

    cm = create_configmap_with_mocked_dependency(
        name="gone",
        mock_gke_cluster=mock_gke_cluster,
    )

    await cm.on_delete()


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert ConfigMap.provider == "kubernetes"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert ConfigMap.resource == "configmap"


async def test_health_exists(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns healthy when configmap exists."""
    mock_cm = mocker.MagicMock()
    mock_cm.data = {"key1": "value1", "key2": "value2"}
    mock_lightkube_client.get.return_value = mock_cm

    cm = create_configmap_with_mocked_dependency(
        name="test-cm",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await cm.health()

    assert result.status == "healthy"
    assert result.details["key_count"] == 2


async def test_health_not_found(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns unhealthy when configmap not found."""
    error = ApiError(response=mocker.MagicMock())
    error.status = mocker.MagicMock(code=404)
    mock_lightkube_client.get.side_effect = error

    cm = create_configmap_with_mocked_dependency(
        name="missing",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await cm.health()

    assert result.status == "unhealthy"
    assert "not found" in result.message


async def test_logs_returns_message(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """logs() yields info message about configmaps not having logs."""
    cm = create_configmap_with_mocked_dependency(
        name="test-cm",
        mock_gke_cluster=mock_gke_cluster,
    )

    entries = []
    async for entry in cm.logs():
        entries.append(entry)

    assert len(entries) == 1
    assert entries[0].level == "info"
    assert "do not produce logs" in entries[0].message
