"""Tests for GCP Secret Manager resource."""

from unittest.mock import MagicMock

from google.api_core.exceptions import AlreadyExists, NotFound
from pragma_sdk.provider import ProviderHarness

from gcp_provider import Secret, SecretConfig, SecretOutputs


async def test_create_secret_success(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
) -> None:
    """on_create creates secret and version, returns outputs."""
    config = SecretConfig(
        project_id="test-project",
        secret_id="my-secret",
        data="super-secret-value",
    )

    result = await harness.invoke_create(Secret, name="my-secret", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.resource_name == "projects/test-project/secrets/test-secret"
    assert (
        result.outputs.version_name
        == "projects/test-project/secrets/test-secret/versions/1"
    )
    assert result.outputs.version_id == "1"

    mock_secretmanager_client.create_secret.assert_called_once()
    mock_secretmanager_client.add_secret_version.assert_called_once()


async def test_create_secret_idempotent(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
) -> None:
    """on_create handles AlreadyExists (idempotent retry)."""
    mock_secretmanager_client.create_secret.side_effect = AlreadyExists("exists")

    config = SecretConfig(
        project_id="test-project",
        secret_id="existing-secret",
        data="value",
    )

    result = await harness.invoke_create(Secret, name="existing", config=config)

    assert result.success
    mock_secretmanager_client.get_secret.assert_called_once()
    mock_secretmanager_client.add_secret_version.assert_called_once()


async def test_update_adds_new_version(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
) -> None:
    """on_update creates new version when data changes."""
    # Update mock for version 2
    mock_version = MagicMock()
    mock_version.name = "projects/proj/secrets/sec/versions/2"
    mock_secretmanager_client.add_secret_version.return_value = mock_version

    previous = SecretConfig(project_id="proj", secret_id="sec", data="old")
    current = SecretConfig(project_id="proj", secret_id="sec", data="new")

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
) -> None:
    """on_update returns existing outputs when data unchanged."""
    previous = SecretConfig(project_id="proj", secret_id="sec", data="same")
    current = SecretConfig(project_id="proj", secret_id="sec", data="same")
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
) -> None:
    """on_update rejects project_id changes."""
    previous = SecretConfig(project_id="proj-a", secret_id="sec", data="val")
    current = SecretConfig(project_id="proj-b", secret_id="sec", data="val")

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
) -> None:
    """on_update rejects secret_id changes."""
    previous = SecretConfig(project_id="proj", secret_id="sec-a", data="val")
    current = SecretConfig(project_id="proj", secret_id="sec-b", data="val")

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
) -> None:
    """on_delete removes secret."""
    config = SecretConfig(project_id="proj", secret_id="sec", data="val")

    result = await harness.invoke_delete(Secret, name="test", config=config)

    assert result.success
    mock_secretmanager_client.delete_secret.assert_called_once_with(
        name="projects/proj/secrets/sec"
    )


async def test_delete_idempotent(
    harness: ProviderHarness,
    mock_secretmanager_client: MagicMock,
) -> None:
    """on_delete succeeds when secret doesn't exist."""
    mock_secretmanager_client.delete_secret.side_effect = NotFound("gone")

    config = SecretConfig(project_id="proj", secret_id="sec", data="val")

    result = await harness.invoke_delete(Secret, name="test", config=config)

    assert result.success
