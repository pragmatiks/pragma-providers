"""Tests for Kubernetes Namespace resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightkube import ApiError
from pragma_sdk import Dependency, LifecycleState

from kubernetes_provider import Namespace, NamespaceConfig, NamespaceOutputs


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_namespace_with_mocked_dependency(
    name: str,
    labels: dict[str, str] | None = None,
    mock_gke_cluster: Any = None,
    outputs: NamespaceOutputs | None = None,
) -> Namespace:
    dep = Dependency(provider="gcp", resource="gke", name="test-cluster")

    config = NamespaceConfig(
        cluster=dep,
        labels=labels,
    )

    if mock_gke_cluster:
        config.cluster._resolved = mock_gke_cluster

    return Namespace(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


async def test_create_namespace_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_create applies namespace and returns outputs."""
    ns = create_namespace_with_mocked_dependency(
        name="agents",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await ns.on_create()

    assert result.name == "agents"
    mock_lightkube_client.apply.assert_called_once()


async def test_create_namespace_with_labels(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_create applies namespace with labels."""
    ns = create_namespace_with_mocked_dependency(
        name="agents",
        labels={"team": "ai", "env": "prod"},
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await ns.on_create()

    assert result.name == "agents"
    mock_lightkube_client.apply.assert_called_once()

    applied_ns = mock_lightkube_client.apply.call_args[0][0]
    assert applied_ns.metadata.labels == {"team": "ai", "env": "prod"}


async def test_update_namespace_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_update applies updated namespace."""
    ns = create_namespace_with_mocked_dependency(
        name="agents",
        labels={"team": "ai"},
        mock_gke_cluster=mock_gke_cluster,
    )

    previous = NamespaceConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        labels=None,
    )

    result = await ns.on_update(previous)

    assert result.name == "agents"
    mock_lightkube_client.apply.assert_called_once()


async def test_delete_namespace_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_delete removes namespace."""
    ns = create_namespace_with_mocked_dependency(
        name="agents",
        mock_gke_cluster=mock_gke_cluster,
    )

    await ns.on_delete()

    mock_lightkube_client.delete.assert_called_once()


async def test_delete_namespace_idempotent(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_delete succeeds when namespace doesn't exist."""
    error = ApiError(response=mocker.MagicMock())
    error.status = mocker.MagicMock(code=404)
    mock_lightkube_client.delete.side_effect = error

    ns = create_namespace_with_mocked_dependency(
        name="gone",
        mock_gke_cluster=mock_gke_cluster,
    )

    await ns.on_delete()


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Namespace.resource == "namespace"


async def test_health_active(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns healthy when namespace is active."""
    mock_ns = mocker.MagicMock()
    mock_ns.status.phase = "Active"
    mock_lightkube_client.get.return_value = mock_ns

    ns = create_namespace_with_mocked_dependency(
        name="agents",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await ns.health()

    assert result.status == "healthy"
    assert result.details["phase"] == "Active"


async def test_health_terminating(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns degraded when namespace is terminating."""
    mock_ns = mocker.MagicMock()
    mock_ns.status.phase = "Terminating"
    mock_lightkube_client.get.return_value = mock_ns

    ns = create_namespace_with_mocked_dependency(
        name="agents",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await ns.health()

    assert result.status == "degraded"


async def test_health_not_found(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns unhealthy when namespace not found."""
    error = ApiError(response=mocker.MagicMock())
    error.status = mocker.MagicMock(code=404)
    mock_lightkube_client.get.side_effect = error

    ns = create_namespace_with_mocked_dependency(
        name="missing",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await ns.health()

    assert result.status == "unhealthy"
    assert "not found" in result.message


async def test_logs_returns_message(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """logs() yields info message about namespaces not having logs."""
    ns = create_namespace_with_mocked_dependency(
        name="agents",
        mock_gke_cluster=mock_gke_cluster,
    )

    entries = []
    async for entry in ns.logs():
        entries.append(entry)

    assert len(entries) == 1
    assert entries[0].level == "info"
    assert "do not produce logs" in entries[0].message
