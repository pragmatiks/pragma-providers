"""Tests for GCP GKE resource."""

from unittest.mock import MagicMock

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud.container_v1.types import Cluster
from pragma_sdk.provider import ProviderHarness

from gcp_provider import GKE, GKEConfig, GKEOutputs


async def test_create_cluster_success(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_create creates cluster and waits for RUNNING state."""
    config = GKEConfig(
        project_id="test-project",
        credentials=sample_credentials,
        region="europe-west4",
        name="test-cluster",
        autopilot=True,
        network="default",
        release_channel="REGULAR",
    )

    result = await harness.invoke_create(GKE, name="test-cluster", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.name == "test-cluster"
    assert result.outputs.endpoint == "https://10.0.0.1"
    assert result.outputs.cluster_ca_certificate == "Y2VydGlmaWNhdGU="
    assert result.outputs.location == "europe-west4"
    assert result.outputs.status == "RUNNING"

    mock_container_client.create_cluster.assert_called_once()
    mock_container_client.get_cluster.assert_called()


async def test_create_cluster_idempotent(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_create handles AlreadyExists (idempotent retry)."""
    mock_container_client.create_cluster.side_effect = AlreadyExists("exists")

    config = GKEConfig(
        project_id="test-project",
        credentials=sample_credentials,
        region="europe-west4",
        name="existing-cluster",
    )

    result = await harness.invoke_create(GKE, name="existing", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.status == "RUNNING"


async def test_create_cluster_with_subnetwork(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_create includes subnetwork when specified."""
    config = GKEConfig(
        project_id="test-project",
        credentials=sample_credentials,
        region="europe-west4",
        name="test-cluster",
        network="custom-vpc",
        subnetwork="custom-subnet",
    )

    result = await harness.invoke_create(GKE, name="test-cluster", config=config)

    assert result.success
    call_args = mock_container_client.create_cluster.call_args
    cluster_config = call_args.kwargs["request"].cluster
    assert cluster_config.network == "custom-vpc"
    assert cluster_config.subnetwork == "custom-subnet"


async def test_create_cluster_standard_mode(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_create supports non-autopilot clusters."""
    config = GKEConfig(
        project_id="test-project",
        credentials=sample_credentials,
        region="europe-west4",
        name="standard-cluster",
        autopilot=False,
    )

    result = await harness.invoke_create(GKE, name="standard-cluster", config=config)

    assert result.success
    call_args = mock_container_client.create_cluster.call_args
    cluster_config = call_args.kwargs["request"].cluster
    # When autopilot is False, the autopilot field should not be set to enabled
    assert not cluster_config.autopilot.enabled


async def test_create_cluster_error_state(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_create fails when cluster enters ERROR state."""
    error_cluster = MagicMock()
    error_cluster.status = Cluster.Status.ERROR
    error_cluster.status_message = "Cluster creation failed"
    mock_container_client.get_cluster.return_value = error_cluster

    config = GKEConfig(
        project_id="test-project",
        credentials=sample_credentials,
        region="europe-west4",
        name="failed-cluster",
    )

    result = await harness.invoke_create(GKE, name="failed-cluster", config=config)

    assert result.failed
    assert result.error is not None
    assert "ERROR state" in str(result.error)


async def test_update_unchanged_returns_existing(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update returns existing outputs when config unchanged."""
    previous = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster",
        network="default",
        autopilot=True,
    )
    current = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster",
        network="default",
        autopilot=True,
    )
    existing_outputs = GKEOutputs(
        name="cluster",
        endpoint="https://10.0.0.1",
        cluster_ca_certificate="Y2VydA==",
        location="europe-west4",
        status="RUNNING",
    )

    result = await harness.invoke_update(
        GKE,
        name="cluster",
        config=current,
        previous_config=previous,
        current_outputs=existing_outputs,
    )

    assert result.success
    assert result.outputs == existing_outputs


async def test_update_rejects_project_change(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update rejects project_id changes."""
    previous = GKEConfig(
        project_id="proj-a",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster",
    )
    current = GKEConfig(
        project_id="proj-b",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster",
    )

    result = await harness.invoke_update(
        GKE,
        name="cluster",
        config=current,
        previous_config=previous,
    )

    assert result.failed
    assert "project_id" in str(result.error)


async def test_update_rejects_region_change(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update rejects region changes."""
    previous = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster",
    )
    current = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="us-central1",
        name="cluster",
    )

    result = await harness.invoke_update(
        GKE,
        name="cluster",
        config=current,
        previous_config=previous,
    )

    assert result.failed
    assert "region" in str(result.error)


async def test_update_rejects_name_change(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update rejects name changes."""
    previous = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster-a",
    )
    current = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster-b",
    )

    result = await harness.invoke_update(
        GKE,
        name="cluster",
        config=current,
        previous_config=previous,
    )

    assert result.failed
    assert "name" in str(result.error)


async def test_update_rejects_network_change(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update rejects network changes."""
    previous = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster",
        network="vpc-a",
    )
    current = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster",
        network="vpc-b",
    )

    result = await harness.invoke_update(
        GKE,
        name="cluster",
        config=current,
        previous_config=previous,
    )

    assert result.failed
    assert "network" in str(result.error)


async def test_update_rejects_autopilot_change(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update rejects autopilot mode changes."""
    previous = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster",
        autopilot=True,
    )
    current = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster",
        autopilot=False,
    )

    result = await harness.invoke_update(
        GKE,
        name="cluster",
        config=current,
        previous_config=previous,
    )

    assert result.failed
    assert "autopilot" in str(result.error)


async def test_delete_success(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_delete removes cluster."""
    # After delete, get_cluster should raise NotFound
    mock_container_client.get_cluster.side_effect = NotFound("deleted")

    config = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster",
    )

    result = await harness.invoke_delete(GKE, name="cluster", config=config)

    assert result.success
    mock_container_client.delete_cluster.assert_called_once()


async def test_delete_idempotent(
    harness: ProviderHarness,
    mock_container_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_delete succeeds when cluster doesn't exist."""
    mock_container_client.delete_cluster.side_effect = NotFound("gone")

    config = GKEConfig(
        project_id="proj",
        credentials=sample_credentials,
        region="europe-west4",
        name="cluster",
    )

    result = await harness.invoke_delete(GKE, name="cluster", config=config)

    assert result.success
