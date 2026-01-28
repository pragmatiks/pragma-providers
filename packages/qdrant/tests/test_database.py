"""Tests for Qdrant Database resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pragma_sdk import Dependency, LifecycleState

from qdrant_provider import (
    Database,
    DatabaseConfig,
    DatabaseOutputs,
    ResourceConfig,
    StorageConfig,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_gke_outputs(mocker: MockerFixture) -> MagicMock:
    """Mock GKE outputs with cluster credentials."""
    outputs = mocker.MagicMock()
    outputs.endpoint = "10.0.0.1"
    outputs.cluster_ca_certificate = "Y2VydGlmaWNhdGU="  # base64 of "certificate"
    outputs.name = "test-cluster"
    outputs.location = "europe-west4"
    outputs.status = "RUNNING"
    return outputs


@pytest.fixture
def mock_gke_resource(mocker: MockerFixture, mock_gke_outputs: MagicMock) -> MagicMock:
    """Mock GKE resource with outputs."""
    resource = mocker.MagicMock()
    resource.outputs = mock_gke_outputs
    return resource


def create_database_with_mocked_dependency(
    name: str,
    replicas: int = 1,
    storage: StorageConfig | None = None,
    resources: ResourceConfig | None = None,
    mock_gke_resource: MagicMock | None = None,
    outputs: DatabaseOutputs | None = None,
) -> Database:
    """Create a Database instance with mocked dependency resolution.

    This bypasses Pydantic validation which loses private attributes.
    """
    # Create the dependency
    dep = Dependency(provider="gcp", resource="gke", name="test-cluster")

    # Create config
    config = DatabaseConfig(
        cluster=dep,
        replicas=replicas,
        storage=storage,
        resources=resources,
    )

    # Now inject the resolved resource after config creation
    # by directly setting it on the config's cluster attribute
    if mock_gke_resource:
        config.cluster._resolved = mock_gke_resource

    # Create resource
    return Database(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


@pytest.fixture
def mock_subprocess(mocker: "MockerFixture") -> MagicMock:
    """Mock subprocess.run for helm and kubectl commands."""
    mock_run = mocker.patch("subprocess.run")

    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        # Handle kubectl get statefulset (readiness check)
        if cmd[0] == "kubectl" and "statefulset" in cmd:
            result.stdout = "1"  # 1 ready replica

        return result

    mock_run.side_effect = run_side_effect
    return mock_run


@pytest.fixture
def mock_asyncio_sleep(mocker: "MockerFixture") -> MagicMock:
    """Mock asyncio.sleep to avoid actual waits."""
    return mocker.patch("qdrant_provider.resources.database.asyncio.sleep", return_value=None)


async def test_create_database_success(
    mock_gke_resource: MagicMock,
    mock_subprocess: MagicMock,
    mock_asyncio_sleep: MagicMock,
) -> None:
    """on_create deploys Qdrant via Helm and returns outputs."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        mock_gke_resource=mock_gke_resource,
    )

    result = await db.on_create()

    assert result.url == "http://qdrant-test-db.default.svc.cluster.local:6333"
    assert result.grpc_url == "http://qdrant-test-db.default.svc.cluster.local:6334"
    assert result.ready is True


async def test_create_database_with_storage(
    mock_gke_resource: MagicMock,
    mock_subprocess: MagicMock,
    mock_asyncio_sleep: MagicMock,
) -> None:
    """on_create includes storage configuration in Helm values."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        storage=StorageConfig(size="20Gi", class_="premium-rwo"),
        mock_gke_resource=mock_gke_resource,
    )

    result = await db.on_create()

    assert result.ready is True

    # Check that helm was called with values containing persistence config
    helm_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == "helm"]
    upgrade_call = next((c for c in helm_calls if "upgrade" in c[0][0]), None)
    assert upgrade_call is not None


async def test_create_database_with_resources(
    mock_gke_resource: MagicMock,
    mocker: "MockerFixture",
) -> None:
    """on_create includes resource limits in Helm values."""

    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if cmd[0] == "kubectl" and "statefulset" in cmd:
            result.stdout = "3"  # 3 ready replicas
        return result

    mocker.patch("subprocess.run", side_effect=run_side_effect)
    mocker.patch("qdrant_provider.resources.database.asyncio.sleep", return_value=None)

    db = create_database_with_mocked_dependency(
        name="test-db",
        replicas=3,
        resources=ResourceConfig(memory="4Gi", cpu="2"),
        mock_gke_resource=mock_gke_resource,
    )

    result = await db.on_create()

    assert result.ready is True


async def test_create_database_helm_repo_added(
    mock_gke_resource: MagicMock,
    mock_subprocess: MagicMock,
    mock_asyncio_sleep: MagicMock,
) -> None:
    """on_create adds Qdrant Helm repo before install."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        mock_gke_resource=mock_gke_resource,
    )

    await db.on_create()

    # Verify helm repo add was called
    helm_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == "helm"]
    repo_add_call = next(
        (c for c in helm_calls if "repo" in c[0][0] and "add" in c[0][0]),
        None,
    )
    assert repo_add_call is not None
    assert "qdrant" in repo_add_call[0][0]
    assert "https://qdrant.github.io/qdrant-helm" in repo_add_call[0][0]


async def test_create_database_helm_upgrade_install(
    mock_gke_resource: MagicMock,
    mock_subprocess: MagicMock,
    mock_asyncio_sleep: MagicMock,
) -> None:
    """on_create uses helm upgrade --install for idempotency."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        mock_gke_resource=mock_gke_resource,
    )

    await db.on_create()

    # Verify helm upgrade --install was called
    helm_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == "helm"]
    upgrade_call = next(
        (c for c in helm_calls if "upgrade" in c[0][0] and "--install" in c[0][0]),
        None,
    )
    assert upgrade_call is not None
    assert "qdrant-test-db" in upgrade_call[0][0]  # release name
    assert "qdrant/qdrant" in upgrade_call[0][0]  # chart name


async def test_create_database_waits_for_ready(
    mock_gke_resource: MagicMock,
    mocker: "MockerFixture",
) -> None:
    """on_create polls StatefulSet until ready."""
    call_count = 0

    def run_side_effect(cmd, **kwargs):
        nonlocal call_count
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        if cmd[0] == "kubectl" and "statefulset" in cmd:
            call_count += 1
            # First call returns 0 replicas, second returns 1
            result.stdout = "0" if call_count == 1 else "1"

        return result

    mocker.patch("subprocess.run", side_effect=run_side_effect)
    mocker.patch("qdrant_provider.resources.database.asyncio.sleep", return_value=None)

    db = create_database_with_mocked_dependency(
        name="test-db",
        mock_gke_resource=mock_gke_resource,
    )

    result = await db.on_create()

    assert result.ready is True
    assert call_count >= 2  # At least two kubectl calls to check readiness


async def test_create_database_helm_failure(
    mock_gke_resource: MagicMock,
    mocker: "MockerFixture",
) -> None:
    """on_create fails when helm command fails."""

    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        if cmd[0] == "helm" and "upgrade" in cmd:
            result.returncode = 1
            result.stderr = "Error: chart not found"
        else:
            result.returncode = 0
            result.stderr = ""
        result.stdout = ""
        return result

    mocker.patch("subprocess.run", side_effect=run_side_effect)

    db = create_database_with_mocked_dependency(
        name="test-db",
        mock_gke_resource=mock_gke_resource,
    )

    with pytest.raises(RuntimeError, match="Helm command failed"):
        await db.on_create()


async def test_update_unchanged_returns_existing(
    mock_gke_resource: MagicMock,
    mock_subprocess: MagicMock,
    mock_asyncio_sleep: MagicMock,
) -> None:
    """on_update returns existing outputs when config unchanged."""
    existing_outputs = DatabaseOutputs(
        url="http://qdrant-test-db.default.svc.cluster.local:6333",
        grpc_url="http://qdrant-test-db.default.svc.cluster.local:6334",
        ready=True,
    )

    db = create_database_with_mocked_dependency(
        name="test-db",
        mock_gke_resource=mock_gke_resource,
        outputs=existing_outputs,
    )

    # Reset mock to track calls
    mock_subprocess.reset_mock()

    # Create identical previous config
    previous_config = DatabaseConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        replicas=1,
    )

    result = await db.on_update(previous_config)

    assert result == existing_outputs
    # Verify no helm calls were made (short-circuited)
    helm_upgrade_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == "helm" and "upgrade" in c[0][0]]
    assert len(helm_upgrade_calls) == 0


async def test_update_replicas_triggers_helm_upgrade(
    mock_gke_resource: MagicMock,
    mocker: "MockerFixture",
) -> None:
    """on_update runs helm upgrade when replicas change."""

    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if cmd[0] == "kubectl" and "statefulset" in cmd:
            result.stdout = "3"  # 3 ready replicas
        return result

    mocker.patch("subprocess.run", side_effect=run_side_effect)
    mocker.patch("qdrant_provider.resources.database.asyncio.sleep", return_value=None)

    db = create_database_with_mocked_dependency(
        name="test-db",
        replicas=3,
        mock_gke_resource=mock_gke_resource,
        outputs=DatabaseOutputs(
            url="http://qdrant-test-db.default.svc.cluster.local:6333",
            grpc_url="http://qdrant-test-db.default.svc.cluster.local:6334",
            ready=True,
        ),
    )

    # Previous config had different replicas
    previous_config = DatabaseConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        replicas=1,
    )

    result = await db.on_update(previous_config)

    assert result.ready is True


async def test_update_rejects_cluster_change(
    mock_gke_resource: MagicMock,
) -> None:
    """on_update rejects cluster changes."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        mock_gke_resource=mock_gke_resource,
    )

    # Previous config had different cluster
    previous_config = DatabaseConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="other-cluster"),
        replicas=1,
    )

    with pytest.raises(ValueError, match="cluster"):
        await db.on_update(previous_config)


async def test_delete_success(
    mock_gke_resource: MagicMock,
    mock_subprocess: MagicMock,
) -> None:
    """on_delete uninstalls the Helm release."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        mock_gke_resource=mock_gke_resource,
    )

    await db.on_delete()

    # Verify helm uninstall was called
    helm_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == "helm"]
    uninstall_call = next(
        (c for c in helm_calls if "uninstall" in c[0][0]),
        None,
    )
    assert uninstall_call is not None
    assert "qdrant-test-db" in uninstall_call[0][0]


async def test_delete_idempotent(
    mock_gke_resource: MagicMock,
    mocker: "MockerFixture",
) -> None:
    """on_delete succeeds when release doesn't exist."""

    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.stdout = ""
        if cmd[0] == "helm" and "uninstall" in cmd:
            result.returncode = 1
            result.stderr = "Error: release not found"
        else:
            result.returncode = 0
            result.stderr = ""
        return result

    mocker.patch("subprocess.run", side_effect=run_side_effect)

    db = create_database_with_mocked_dependency(
        name="test-db",
        mock_gke_resource=mock_gke_resource,
    )

    # Should not raise
    await db.on_delete()


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert Database.provider == "qdrant"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Database.resource == "database"


def test_storage_config_alias() -> None:
    """StorageConfig uses 'class' alias for class_ field."""
    storage = StorageConfig(size="10Gi")
    assert storage.class_ == "standard-rwo"

    # Can use alias in input
    storage = StorageConfig.model_validate({"size": "20Gi", "class": "premium-rwo"})
    assert storage.class_ == "premium-rwo"


def test_resource_config_defaults() -> None:
    """ResourceConfig has sensible defaults."""
    resources = ResourceConfig()
    assert resources.memory == "2Gi"
    assert resources.cpu == "1"


def test_build_helm_values() -> None:
    """_build_helm_values correctly structures values for Helm."""
    db = create_database_with_mocked_dependency(
        name="test-db",
        replicas=3,
        storage=StorageConfig(size="20Gi", class_="premium-rwo"),
        resources=ResourceConfig(memory="4Gi", cpu="2"),
    )

    values = db._build_helm_values()

    assert values["replicaCount"] == 3
    assert values["persistence"]["size"] == "20Gi"
    assert values["persistence"]["storageClassName"] == "premium-rwo"
    assert values["resources"]["limits"]["memory"] == "4Gi"
    assert values["resources"]["limits"]["cpu"] == "2"


def test_build_outputs() -> None:
    """_build_outputs creates correct in-cluster URLs."""
    db = create_database_with_mocked_dependency(name="my-qdrant")

    outputs = db._build_outputs()

    assert outputs.url == "http://qdrant-my-qdrant.default.svc.cluster.local:6333"
    assert outputs.grpc_url == "http://qdrant-my-qdrant.default.svc.cluster.local:6334"
    assert outputs.ready is True


def test_release_name() -> None:
    """_get_release_name prefixes with qdrant-."""
    db = create_database_with_mocked_dependency(name="test-db")
    assert db._get_release_name() == "qdrant-test-db"
