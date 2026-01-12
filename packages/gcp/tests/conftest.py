"""Pytest configuration for gcp provider tests."""

from unittest.mock import MagicMock

import pytest
from pragma_sdk.provider import ProviderHarness


@pytest.fixture
def harness() -> ProviderHarness:
    """Test harness for invoking lifecycle methods."""
    return ProviderHarness()


@pytest.fixture
def mock_secretmanager_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock GCP Secret Manager client."""
    mock_client = MagicMock()

    # Mock create_secret response
    mock_secret = MagicMock()
    mock_secret.name = "projects/test-project/secrets/test-secret"
    mock_client.create_secret.return_value = mock_secret
    mock_client.get_secret.return_value = mock_secret

    # Mock add_secret_version response
    mock_version = MagicMock()
    mock_version.name = "projects/test-project/secrets/test-secret/versions/1"
    mock_client.add_secret_version.return_value = mock_version

    # Patch the client constructor
    monkeypatch.setattr(
        "gcp_provider.resources.secret.secretmanager.SecretManagerServiceClient",
        lambda: mock_client,
    )

    return mock_client
