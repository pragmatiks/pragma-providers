"""Pytest configuration for gcp provider tests."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from google.cloud.container_v1.types import Cluster
from pragma_sdk.provider import ProviderHarness


if TYPE_CHECKING:
    from pytest_mock import MockerFixture

SAMPLE_CREDENTIALS = {
    "type": "service_account",
    "project_id": "test-project",
    "private_key_id": "key123",
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGy0AHB7MJ7EH7M7FV8PLVP5\nfake-key-for-testing-only\n-----END RSA PRIVATE KEY-----\n",
    "client_email": "test@test-project.iam.gserviceaccount.com",
    "client_id": "123456789",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com",
}


@pytest.fixture
def harness() -> ProviderHarness:
    """Test harness for invoking lifecycle methods."""
    return ProviderHarness()


@pytest.fixture
def sample_credentials() -> dict:
    """Sample GCP service account credentials for testing."""
    return SAMPLE_CREDENTIALS.copy()


@pytest.fixture
def mock_secretmanager_client(mocker: "MockerFixture") -> MagicMock:
    """Mock GCP Secret Manager async client with credentials support."""
    mock_client = MagicMock()

    mock_secret = MagicMock()
    mock_secret.name = "projects/test-project/secrets/test-secret"
    mock_client.create_secret = mocker.AsyncMock(return_value=mock_secret)
    mock_client.get_secret = mocker.AsyncMock(return_value=mock_secret)

    mock_version = MagicMock()
    mock_version.name = "projects/test-project/secrets/test-secret/versions/1"
    mock_client.add_secret_version = mocker.AsyncMock(return_value=mock_version)
    mock_client.delete_secret = mocker.AsyncMock()

    mocker.patch(
        "gcp_provider.resources.secret.SecretManagerServiceAsyncClient",
        return_value=mock_client,
    )

    mock_credentials = MagicMock()
    mocker.patch(
        "gcp_provider.resources.secret.service_account.Credentials.from_service_account_info",
        return_value=mock_credentials,
    )

    return mock_client


@pytest.fixture
def mock_container_client(mocker: "MockerFixture") -> MagicMock:
    """Mock GCP Container (GKE) async client with credentials support."""
    mock_client = MagicMock()

    # Mock cluster in RUNNING state
    mock_cluster = MagicMock()
    mock_cluster.name = "test-cluster"
    mock_cluster.endpoint = "https://10.0.0.1"
    mock_cluster.location = "europe-west4"
    mock_cluster.status = Cluster.Status.RUNNING
    mock_cluster.master_auth.cluster_ca_certificate = "Y2VydGlmaWNhdGU="

    mock_client.create_cluster = mocker.AsyncMock(return_value=MagicMock())
    mock_client.get_cluster = mocker.AsyncMock(return_value=mock_cluster)
    mock_client.delete_cluster = mocker.AsyncMock(return_value=MagicMock())

    mocker.patch(
        "gcp_provider.resources.gke.ClusterManagerAsyncClient",
        return_value=mock_client,
    )

    mock_credentials = MagicMock()
    mocker.patch(
        "gcp_provider.resources.gke.service_account.Credentials.from_service_account_info",
        return_value=mock_credentials,
    )

    # Patch asyncio.sleep to avoid actual waits in tests
    mocker.patch("gcp_provider.resources.gke.asyncio.sleep", return_value=None)

    return mock_client
