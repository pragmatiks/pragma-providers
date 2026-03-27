"""Tests for Supabase Auth configuration resource."""

from __future__ import annotations

import json
from typing import Any

import respx
from pragma_sdk.provider import ProviderHarness

from supabase_provider import Auth, AuthConfig, AuthOutputs, ExternalProviderConfig


_BASE_URL = "https://api.supabase.com/v1"


async def test_create_auth_config_success(
    harness: ProviderHarness,
    sample_auth_config_data: dict[str, Any],
) -> None:
    """on_create applies auth configuration and returns outputs."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.patch("/projects/testref/config/auth").respond(200, json=sample_auth_config_data)
        mock.get("/projects/testref/config/auth").respond(200, json=sample_auth_config_data)

        config = AuthConfig(
            access_token="sbp_test_token",
            project_ref="testref",
            site_url="https://myapp.example.com",
            disable_signup=False,
            jwt_expiry=3600,
            external_email_enabled=True,
            external_google=ExternalProviderConfig(
                enabled=True,
                client_id="google-client-id",
                secret="google-secret",
            ),
        )

        result = await harness.invoke_create(Auth, name="auth-config", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.project_ref == "testref"
    assert result.outputs.site_url == "https://myapp.example.com"
    assert result.outputs.disable_signup is False
    assert result.outputs.jwt_expiry == 3600
    assert result.outputs.external_email_enabled is True
    assert "google" in result.outputs.external_providers_enabled


async def test_create_auth_config_with_github(
    harness: ProviderHarness,
) -> None:
    """on_create includes GitHub provider configuration."""
    response_data = {
        "SITE_URL": "https://myapp.example.com",
        "DISABLE_SIGNUP": False,
        "JWT_EXP": 3600,
        "EXTERNAL_EMAIL_ENABLED": True,
        "EXTERNAL_PHONE_ENABLED": False,
        "MAILER_AUTOCONFIRM": False,
        "EXTERNAL_GOOGLE_ENABLED": False,
        "EXTERNAL_GITHUB_ENABLED": True,
        "EXTERNAL_APPLE_ENABLED": False,
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        patch_route = mock.patch("/projects/testref/config/auth").respond(200, json=response_data)
        mock.get("/projects/testref/config/auth").respond(200, json=response_data)

        config = AuthConfig(
            access_token="sbp_test_token",
            project_ref="testref",
            external_github=ExternalProviderConfig(
                enabled=True,
                client_id="gh-client-id",
                secret="gh-secret",
            ),
        )

        result = await harness.invoke_create(Auth, name="auth-config", config=config)

    assert result.success
    assert result.outputs is not None
    assert "github" in result.outputs.external_providers_enabled

    patch_request = patch_route.calls[0].request
    body = json.loads(patch_request.content)
    assert body["EXTERNAL_GITHUB_ENABLED"] is True
    assert body["EXTERNAL_GITHUB_CLIENT_ID"] == "gh-client-id"
    assert body["EXTERNAL_GITHUB_SECRET"] == "gh-secret"


async def test_create_auth_config_api_error(
    harness: ProviderHarness,
) -> None:
    """on_create propagates API errors."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.patch("/projects/testref/config/auth").respond(404, json={"message": "Project not found"})

        config = AuthConfig(
            access_token="sbp_test_token",
            project_ref="testref",
        )

        result = await harness.invoke_create(Auth, name="auth-config", config=config)

    assert result.failed
    assert "Project not found" in str(result.error)


async def test_update_auth_config_success(
    harness: ProviderHarness,
    sample_auth_config_data: dict[str, Any],
) -> None:
    """on_update patches auth configuration."""
    updated_data = {**sample_auth_config_data, "DISABLE_SIGNUP": True}

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.patch("/projects/testref/config/auth").respond(200, json=updated_data)
        mock.get("/projects/testref/config/auth").respond(200, json=updated_data)

        previous = AuthConfig(
            access_token="sbp_test_token",
            project_ref="testref",
            disable_signup=False,
        )
        current = AuthConfig(
            access_token="sbp_test_token",
            project_ref="testref",
            disable_signup=True,
        )
        existing_outputs = AuthOutputs(
            project_ref="testref",
            site_url="https://myapp.example.com",
            disable_signup=False,
            jwt_expiry=3600,
            external_email_enabled=True,
            external_phone_enabled=False,
            mailer_autoconfirm=False,
            external_providers_enabled=["google"],
        )

        result = await harness.invoke_update(
            Auth,
            name="auth-config",
            config=current,
            previous_config=previous,
            current_outputs=existing_outputs,
        )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.disable_signup is True


async def test_delete_auth_config_resets_to_defaults(
    harness: ProviderHarness,
) -> None:
    """on_delete resets auth configuration to defaults."""
    with respx.mock(base_url=_BASE_URL) as mock:
        patch_route = mock.patch("/projects/testref/config/auth").respond(200, json={})

        config = AuthConfig(
            access_token="sbp_test_token",
            project_ref="testref",
            disable_signup=True,
            external_github=ExternalProviderConfig(
                enabled=True,
                client_id="gh-client-id",
                secret="gh-secret",
            ),
        )

        result = await harness.invoke_delete(Auth, name="auth-config", config=config)

    assert result.success

    patch_request = patch_route.calls[0].request
    body = json.loads(patch_request.content)
    assert body["DISABLE_SIGNUP"] is False
    assert body["EXTERNAL_GOOGLE_ENABLED"] is False
    assert body["EXTERNAL_GITHUB_ENABLED"] is False
    assert body["EXTERNAL_APPLE_ENABLED"] is False
    assert body["EXTERNAL_EMAIL_ENABLED"] is True


async def test_create_auth_config_with_redirect_urls(
    harness: ProviderHarness,
    sample_auth_config_data: dict[str, Any],
) -> None:
    """on_create includes redirect URLs in the patch."""
    with respx.mock(base_url=_BASE_URL) as mock:
        patch_route = mock.patch("/projects/testref/config/auth").respond(200, json=sample_auth_config_data)
        mock.get("/projects/testref/config/auth").respond(200, json=sample_auth_config_data)

        config = AuthConfig(
            access_token="sbp_test_token",
            project_ref="testref",
            site_url="https://myapp.example.com",
            additional_redirect_urls=["https://localhost:3000", "https://staging.myapp.example.com"],
        )

        result = await harness.invoke_create(Auth, name="auth-config", config=config)

    assert result.success

    patch_request = patch_route.calls[0].request
    body = json.loads(patch_request.content)
    assert body["URI_ALLOW_LIST"] == "https://localhost:3000,https://staging.myapp.example.com"


async def test_create_auth_config_minimal(
    harness: ProviderHarness,
) -> None:
    """on_create works with minimal configuration (just access_token and project_ref)."""
    response_data = {
        "SITE_URL": "",
        "DISABLE_SIGNUP": False,
        "JWT_EXP": 3600,
        "EXTERNAL_EMAIL_ENABLED": True,
        "EXTERNAL_PHONE_ENABLED": False,
        "MAILER_AUTOCONFIRM": False,
        "EXTERNAL_GOOGLE_ENABLED": False,
        "EXTERNAL_GITHUB_ENABLED": False,
        "EXTERNAL_APPLE_ENABLED": False,
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.patch("/projects/testref/config/auth").respond(200, json=response_data)
        mock.get("/projects/testref/config/auth").respond(200, json=response_data)

        config = AuthConfig(
            access_token="sbp_test_token",
            project_ref="testref",
        )

        result = await harness.invoke_create(Auth, name="auth-config", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.external_providers_enabled == []


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Auth.resource == "auth"
