"""Tests for GitHub Secret resource."""

from __future__ import annotations

import json
from typing import Any

import respx
from pragma_sdk.provider import ProviderHarness

from github_provider import Secret, SecretConfig, SecretOutputs


_BASE_URL = "https://api.github.com"


async def test_create_repository_secret_success(
    harness: ProviderHarness,
    sample_public_key_data: dict[str, Any],
    sample_secret_metadata: dict[str, Any],
) -> None:
    """on_create encrypts and creates a repository-level secret."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.get("/repos/my-org/my-repo/actions/secrets/public-key").respond(200, json=sample_public_key_data)
        put_route = mock.put("/repos/my-org/my-repo/actions/secrets/DEPLOY_KEY").respond(201, json={})
        mock.get("/repos/my-org/my-repo/actions/secrets/DEPLOY_KEY").respond(200, json=sample_secret_metadata)

        config = SecretConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            secret_name="DEPLOY_KEY",
            secret_value="super-secret-value",
        )

        result = await harness.invoke_create(Secret, name="deploy-key", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.secret_name == "DEPLOY_KEY"
    assert result.outputs.scope == "repository"
    assert result.outputs.environment_name == ""
    assert result.outputs.created_at == "2026-03-27T12:00:00Z"

    request_body = json.loads(put_route.calls[0].request.content)
    assert request_body["key_id"] == "568250167242549743"
    assert request_body["encrypted_value"] != ""
    assert request_body["encrypted_value"] != "super-secret-value"


async def test_create_environment_secret_success(
    harness: ProviderHarness,
    sample_public_key_data: dict[str, Any],
    sample_secret_metadata: dict[str, Any],
) -> None:
    """on_create creates an environment-level secret."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.get("/repos/my-org/my-repo/environments/production/secrets/public-key").respond(
            200, json=sample_public_key_data
        )
        mock.put("/repos/my-org/my-repo/environments/production/secrets/API_KEY").respond(201, json={})
        mock.get("/repos/my-org/my-repo/environments/production/secrets/API_KEY").respond(
            200, json={**sample_secret_metadata, "name": "API_KEY"}
        )

        config = SecretConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            secret_name="API_KEY",
            secret_value="api-key-value",
            environment_name="production",
        )

        result = await harness.invoke_create(Secret, name="api-key", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.secret_name == "API_KEY"
    assert result.outputs.scope == "environment"
    assert result.outputs.environment_name == "production"


async def test_create_secret_api_error_on_public_key(
    harness: ProviderHarness,
) -> None:
    """on_create propagates errors when fetching public key fails."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.get("/repos/my-org/my-repo/actions/secrets/public-key").respond(404, json={"message": "Not Found"})

        config = SecretConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            secret_name="MY_SECRET",
            secret_value="value",
        )

        result = await harness.invoke_create(Secret, name="my-secret", config=config)

    assert result.failed
    assert "Not Found" in str(result.error)


async def test_create_secret_api_error_on_put(
    harness: ProviderHarness,
    sample_public_key_data: dict[str, Any],
) -> None:
    """on_create propagates errors when creating the secret fails."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.get("/repos/my-org/my-repo/actions/secrets/public-key").respond(200, json=sample_public_key_data)
        mock.put("/repos/my-org/my-repo/actions/secrets/MY_SECRET").respond(422, json={"message": "Validation Failed"})

        config = SecretConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            secret_name="MY_SECRET",
            secret_value="value",
        )

        result = await harness.invoke_create(Secret, name="my-secret", config=config)

    assert result.failed
    assert "Validation Failed" in str(result.error)


async def test_update_secret_replaces_value(
    harness: ProviderHarness,
    sample_public_key_data: dict[str, Any],
    sample_secret_metadata: dict[str, Any],
) -> None:
    """on_update re-encrypts and replaces the secret."""
    updated_metadata = {**sample_secret_metadata, "updated_at": "2026-03-27T14:00:00Z"}

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.get("/repos/my-org/my-repo/actions/secrets/public-key").respond(200, json=sample_public_key_data)
        mock.put("/repos/my-org/my-repo/actions/secrets/DEPLOY_KEY").respond(204, json={})
        mock.get("/repos/my-org/my-repo/actions/secrets/DEPLOY_KEY").respond(200, json=updated_metadata)

        previous = SecretConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            secret_name="DEPLOY_KEY",
            secret_value="old-value",
        )
        current = SecretConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            secret_name="DEPLOY_KEY",
            secret_value="new-value",
        )
        existing_outputs = SecretOutputs(
            secret_name="DEPLOY_KEY",
            scope="repository",
            environment_name="",
            created_at="2026-03-27T12:00:00Z",
            updated_at="2026-03-27T12:00:00Z",
        )

        result = await harness.invoke_update(
            Secret,
            name="deploy-key",
            config=current,
            previous_config=previous,
            current_outputs=existing_outputs,
        )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.updated_at == "2026-03-27T14:00:00Z"


async def test_delete_repository_secret_success(
    harness: ProviderHarness,
) -> None:
    """on_delete removes a repository-level secret."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/repos/my-org/my-repo/actions/secrets/DEPLOY_KEY").respond(204)

        config = SecretConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            secret_name="DEPLOY_KEY",
            secret_value="value",
        )
        existing_outputs = SecretOutputs(
            secret_name="DEPLOY_KEY",
            scope="repository",
            environment_name="",
            created_at="2026-03-27T12:00:00Z",
            updated_at="2026-03-27T12:00:00Z",
        )

        result = await harness.invoke_delete(
            Secret,
            name="deploy-key",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_environment_secret_success(
    harness: ProviderHarness,
) -> None:
    """on_delete removes an environment-level secret."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/repos/my-org/my-repo/environments/production/secrets/API_KEY").respond(204)

        config = SecretConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            secret_name="API_KEY",
            secret_value="value",
            environment_name="production",
        )
        existing_outputs = SecretOutputs(
            secret_name="API_KEY",
            scope="environment",
            environment_name="production",
            created_at="2026-03-27T12:00:00Z",
            updated_at="2026-03-27T12:00:00Z",
        )

        result = await harness.invoke_delete(
            Secret,
            name="api-key",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_secret_already_gone(
    harness: ProviderHarness,
) -> None:
    """on_delete is idempotent when secret does not exist."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/repos/my-org/my-repo/actions/secrets/DEPLOY_KEY").respond(404, json={"message": "Not Found"})

        config = SecretConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            secret_name="DEPLOY_KEY",
            secret_value="value",
        )
        existing_outputs = SecretOutputs(
            secret_name="DEPLOY_KEY",
            scope="repository",
            environment_name="",
            created_at="2026-03-27T12:00:00Z",
            updated_at="2026-03-27T12:00:00Z",
        )

        result = await harness.invoke_delete(
            Secret,
            name="deploy-key",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_secret_no_outputs(
    harness: ProviderHarness,
) -> None:
    """on_delete succeeds silently when no outputs exist -- secret has no prior state."""
    config = SecretConfig(
        access_token="ghp_test_token",
        owner="my-org",
        repository="my-repo",
        secret_name="DEPLOY_KEY",
        secret_value="value",
    )

    with respx.mock(base_url=_BASE_URL):
        result = await harness.invoke_delete(
            Secret,
            name="deploy-key",
            config=config,
        )

    assert result.success


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Secret.resource == "secret"
