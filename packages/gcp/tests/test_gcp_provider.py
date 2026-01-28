"""Tests for GCP Secret Manager resource."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from google.api_core.exceptions import AlreadyExists, NotFound
from pragma_sdk.provider import ProviderHarness

from gcp_provider import Secret, SecretConfig, SecretOutputs

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# Sample GCP service account credentials for testing
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


async def test_create_secret_success(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_create creates secret and version, returns outputs."""
    config = SecretConfig(
        project_id="test-project",
        secret_id="my-secret",
        data="super-secret-value",
        credentials=sample_credentials,
    )

    result = await harness.invoke_create(Secret, name="my-secret", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.resource_name == "projects/test-project/secrets/test-secret"
    assert result.outputs.version_name == "projects/test-project/secrets/test-secret/versions/1"
    assert result.outputs.version_id == "1"

    mock_secretmanager_client.create_secret.assert_called_once()
    mock_secretmanager_client.add_secret_version.assert_called_once()


async def test_create_secret_idempotent(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_create handles AlreadyExists (idempotent retry)."""
    mock_secretmanager_client.create_secret.side_effect = AlreadyExists("exists")

    config = SecretConfig(
        project_id="test-project",
        secret_id="existing-secret",
        data="value",
        credentials=sample_credentials,
    )

    result = await harness.invoke_create(Secret, name="existing", config=config)

    assert result.success
    mock_secretmanager_client.get_secret.assert_called_once()
    mock_secretmanager_client.add_secret_version.assert_called_once()


async def test_update_adds_new_version(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
    sample_credentials: dict,
    mocker: MockerFixture,
) -> None:
    """on_update creates new version when data changes."""
    # Update mock for version 2
    mock_version = mocker.MagicMock()
    mock_version.name = "projects/proj/secrets/sec/versions/2"
    mock_secretmanager_client.add_secret_version.return_value = mock_version

    previous = SecretConfig(project_id="proj", secret_id="sec", data="old", credentials=sample_credentials)
    current = SecretConfig(project_id="proj", secret_id="sec", data="new", credentials=sample_credentials)

    result = await harness.invoke_update(
        Secret,
        name="test",
        config=current,
        previous_config=previous,
        current_outputs=SecretOutputs(
            resource_name="projects/proj/secrets/sec",
            version_name="projects/proj/secrets/sec/versions/1",
            version_id="1",
        ),
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.version_id == "2"
    mock_secretmanager_client.add_secret_version.assert_called_once()


async def test_update_no_change_returns_existing(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update returns existing outputs when data unchanged."""
    previous = SecretConfig(project_id="proj", secret_id="sec", data="same", credentials=sample_credentials)
    current = SecretConfig(project_id="proj", secret_id="sec", data="same", credentials=sample_credentials)
    existing_outputs = SecretOutputs(
        resource_name="projects/proj/secrets/sec",
        version_name="projects/proj/secrets/sec/versions/1",
        version_id="1",
    )

    result = await harness.invoke_update(
        Secret,
        name="test",
        config=current,
        previous_config=previous,
        current_outputs=existing_outputs,
    )

    assert result.success
    assert result.outputs == existing_outputs
    mock_secretmanager_client.add_secret_version.assert_not_called()


async def test_update_rejects_project_change(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update rejects project_id changes."""
    previous = SecretConfig(project_id="proj-a", secret_id="sec", data="val", credentials=sample_credentials)
    current = SecretConfig(project_id="proj-b", secret_id="sec", data="val", credentials=sample_credentials)

    result = await harness.invoke_update(
        Secret,
        name="test",
        config=current,
        previous_config=previous,
    )

    assert result.failed
    assert result.error is not None
    assert "project_id" in str(result.error)


async def test_update_rejects_secret_id_change(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_update rejects secret_id changes."""
    previous = SecretConfig(project_id="proj", secret_id="sec-a", data="val", credentials=sample_credentials)
    current = SecretConfig(project_id="proj", secret_id="sec-b", data="val", credentials=sample_credentials)

    result = await harness.invoke_update(
        Secret,
        name="test",
        config=current,
        previous_config=previous,
    )

    assert result.failed
    assert result.error is not None
    assert "secret_id" in str(result.error)


async def test_delete_success(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_delete removes secret."""
    config = SecretConfig(project_id="proj", secret_id="sec", data="val", credentials=sample_credentials)

    result = await harness.invoke_delete(Secret, name="test", config=config)

    assert result.success
    mock_secretmanager_client.delete_secret.assert_called_once_with(name="projects/proj/secrets/sec")


async def test_delete_idempotent(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
    sample_credentials: dict,
) -> None:
    """on_delete succeeds when secret doesn't exist."""
    mock_secretmanager_client.delete_secret.side_effect = NotFound("gone")

    config = SecretConfig(project_id="proj", secret_id="sec", data="val", credentials=sample_credentials)

    result = await harness.invoke_delete(Secret, name="test", config=config)

    assert result.success


async def test_create_with_string_credentials(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
) -> None:
    """on_create accepts JSON-encoded string credentials."""
    # Credentials as a JSON string (common when passing through env vars or refs)
    string_credentials = json.dumps(SAMPLE_CREDENTIALS)

    config = SecretConfig(
        project_id="test-project",
        secret_id="my-secret",
        data="secret-value",
        credentials=string_credentials,
    )

    result = await harness.invoke_create(Secret, name="my-secret", config=config)

    assert result.success
    assert result.outputs is not None
    mock_secretmanager_client.create_secret.assert_called_once()
