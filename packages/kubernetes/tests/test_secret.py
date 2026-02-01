"""Tests for Kubernetes Secret resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from lightkube import ApiError
from pragma_sdk import Dependency, LifecycleState

from kubernetes_provider import Secret, SecretConfig, SecretOutputs


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_secret_with_mocked_dependency(
    name: str,
    namespace: str = "default",
    secret_type: str = "Opaque",
    data: dict | None = None,
    string_data: dict | None = None,
    mock_gke_cluster: Any = None,
    outputs: SecretOutputs | None = None,
) -> Secret:
    dep = Dependency(provider="gcp", resource="gke", name="test-cluster")

    config = SecretConfig(
        cluster=dep,
        namespace=namespace,
        type=secret_type,
        data=data,
        string_data=string_data,
    )

    if mock_gke_cluster:
        config.cluster._resolved = mock_gke_cluster

    return Secret(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


async def test_create_secret_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_create applies secret and returns outputs."""
    secret = create_secret_with_mocked_dependency(
        name="test-secret",
        data={"password": "secret123"},
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await secret.on_create()

    assert result.name == "test-secret"
    assert result.type == "Opaque"
    assert result.data == {"password": "secret123"}
    mock_lightkube_client.apply.assert_called_once()


async def test_create_secret_with_string_data(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_create handles string_data (plain text)."""
    secret = create_secret_with_mocked_dependency(
        name="api-secret",
        string_data={"api-key": "my-api-key"},
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await secret.on_create()

    assert result.name == "api-secret"
    assert result.data == {"api-key": "my-api-key"}


async def test_update_secret_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_update applies updated secret."""
    secret = create_secret_with_mocked_dependency(
        name="test-secret",
        data={"password": "new"},
        mock_gke_cluster=mock_gke_cluster,
    )

    previous = SecretConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        namespace="default",
        data={"password": "old"},
    )

    result = await secret.on_update(previous)

    assert result.name == "test-secret"
    mock_lightkube_client.apply.assert_called_once()


async def test_update_rejects_namespace_change(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_update rejects namespace changes."""
    secret = create_secret_with_mocked_dependency(
        name="test-secret",
        namespace="ns-b",
        mock_gke_cluster=mock_gke_cluster,
    )

    previous = SecretConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        namespace="ns-a",
        data={"key": "value"},
    )

    with pytest.raises(ValueError, match="namespace"):
        await secret.on_update(previous)


async def test_delete_secret_success(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """on_delete removes secret."""
    secret = create_secret_with_mocked_dependency(
        name="test-secret",
        mock_gke_cluster=mock_gke_cluster,
    )

    await secret.on_delete()

    mock_lightkube_client.delete.assert_called_once()


async def test_delete_secret_idempotent(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """on_delete succeeds when secret doesn't exist."""
    error = ApiError(response=mocker.MagicMock())
    error.status = mocker.MagicMock(code=404)
    mock_lightkube_client.delete.side_effect = error

    secret = create_secret_with_mocked_dependency(
        name="gone",
        mock_gke_cluster=mock_gke_cluster,
    )

    await secret.on_delete()


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert Secret.provider == "kubernetes"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Secret.resource == "secret"


async def test_health_exists(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns healthy when secret exists."""
    mock_secret = mocker.MagicMock()
    mock_secret.data = {"username": "dXNlcg==", "password": "cGFzcw=="}
    mock_secret.type = "Opaque"
    mock_lightkube_client.get.return_value = mock_secret

    secret = create_secret_with_mocked_dependency(
        name="test-secret",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await secret.health()

    assert result.status == "healthy"
    assert result.details["key_count"] == 2
    assert result.details["type"] == "Opaque"


async def test_health_not_found(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
    mocker: MockerFixture,
) -> None:
    """health() returns unhealthy when secret not found."""
    error = ApiError(response=mocker.MagicMock())
    error.status = mocker.MagicMock(code=404)
    mock_lightkube_client.get.side_effect = error

    secret = create_secret_with_mocked_dependency(
        name="missing",
        mock_gke_cluster=mock_gke_cluster,
    )

    result = await secret.health()

    assert result.status == "unhealthy"
    assert "not found" in result.message


async def test_logs_returns_message(
    mock_lightkube_client: Any,
    mock_gke_cluster: Any,
) -> None:
    """logs() yields info message about secrets not having logs."""
    secret = create_secret_with_mocked_dependency(
        name="test-secret",
        mock_gke_cluster=mock_gke_cluster,
    )

    entries = []
    async for entry in secret.logs():
        entries.append(entry)

    assert len(entries) == 1
    assert entries[0].level == "info"
    assert "do not produce logs" in entries[0].message
