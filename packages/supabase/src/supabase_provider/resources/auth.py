"""Supabase Auth configuration resource."""

from __future__ import annotations

from typing import Any

from pragma_sdk import Config, Field, Outputs, Resource, SensitiveField
from pydantic import BaseModel
from pydantic import Field as PydanticField

from supabase_provider.client import create_management_client, raise_for_status


class ExternalProviderConfig(BaseModel):
    """Configuration for an external OAuth provider.

    Attributes:
        enabled: Whether this provider is enabled.
        client_id: OAuth client ID for the provider.
        secret: OAuth client secret for the provider.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = True
    client_id: str = ""
    secret: str = ""


class AuthConfig(Config):
    """Configuration for Supabase Auth settings.

    Attributes:
        access_token: Supabase personal access token for the Management API.
        project_ref: Supabase project reference ID.
        site_url: The base URL of the site where users are redirected after
            authentication.
        additional_redirect_urls: Allowed redirect URLs beyond the site URL.
        disable_signup: Whether to disable new user signups.
        jwt_expiry: JWT token expiry time in seconds. Defaults to 3600.
        external_email_enabled: Whether email/password authentication is enabled.
        external_phone_enabled: Whether phone/OTP authentication is enabled.
        mailer_autoconfirm: Whether to auto-confirm email addresses on signup.
        external_google: Google OAuth provider configuration.
        external_github: GitHub OAuth provider configuration.
        external_apple: Apple OAuth provider configuration.
    """

    access_token: SensitiveField[str]
    project_ref: Field[str]
    site_url: Field[str] | None = None
    additional_redirect_urls: Field[list[str]] = PydanticField(default_factory=list)
    disable_signup: Field[bool] = False
    jwt_expiry: Field[int] = 3600
    external_email_enabled: Field[bool] = True
    external_phone_enabled: Field[bool] = False
    mailer_autoconfirm: Field[bool] = False
    external_google: Field[ExternalProviderConfig] | None = None
    external_github: Field[ExternalProviderConfig] | None = None
    external_apple: Field[ExternalProviderConfig] | None = None


class AuthOutputs(Outputs):
    """Outputs from Supabase Auth configuration.

    Attributes:
        project_ref: Supabase project reference ID.
        site_url: Configured site URL.
        disable_signup: Whether signups are disabled.
        jwt_expiry: JWT token expiry in seconds.
        external_email_enabled: Whether email auth is enabled.
        external_phone_enabled: Whether phone auth is enabled.
        mailer_autoconfirm: Whether email auto-confirm is on.
        external_providers_enabled: List of enabled external OAuth providers.
    """

    project_ref: str
    site_url: str
    disable_signup: bool
    jwt_expiry: int
    external_email_enabled: bool
    external_phone_enabled: bool
    mailer_autoconfirm: bool
    external_providers_enabled: list[str]


def _build_auth_patch(config: AuthConfig) -> dict[str, Any]:
    """Build the API request body from auth configuration.

    Args:
        config: Auth configuration to convert to API payload.

    Returns:
        Dictionary suitable for PATCH /config/auth request body.
    """
    patch: dict[str, Any] = {
        "DISABLE_SIGNUP": config.disable_signup,
        "JWT_EXP": config.jwt_expiry,
        "EXTERNAL_EMAIL_ENABLED": config.external_email_enabled,
        "EXTERNAL_PHONE_ENABLED": config.external_phone_enabled,
        "MAILER_AUTOCONFIRM": config.mailer_autoconfirm,
    }

    if config.site_url is not None:
        patch["SITE_URL"] = config.site_url

    if config.additional_redirect_urls:
        patch["URI_ALLOW_LIST"] = ",".join(config.additional_redirect_urls)

    if config.external_google is not None:
        patch["EXTERNAL_GOOGLE_ENABLED"] = config.external_google.enabled
        patch["EXTERNAL_GOOGLE_CLIENT_ID"] = config.external_google.client_id
        patch["EXTERNAL_GOOGLE_SECRET"] = config.external_google.secret

    if config.external_github is not None:
        patch["EXTERNAL_GITHUB_ENABLED"] = config.external_github.enabled
        patch["EXTERNAL_GITHUB_CLIENT_ID"] = config.external_github.client_id
        patch["EXTERNAL_GITHUB_SECRET"] = config.external_github.secret

    if config.external_apple is not None:
        patch["EXTERNAL_APPLE_ENABLED"] = config.external_apple.enabled
        patch["EXTERNAL_APPLE_CLIENT_ID"] = config.external_apple.client_id
        patch["EXTERNAL_APPLE_SECRET"] = config.external_apple.secret

    return patch


def _build_outputs_from_api(project_ref: str, data: dict[str, Any]) -> AuthOutputs:
    """Build AuthOutputs from the Supabase API response.

    Args:
        project_ref: Supabase project reference ID.
        data: Raw auth config response from GET /config/auth.

    Returns:
        AuthOutputs reflecting the current auth configuration state.
    """
    enabled_providers: list[str] = []
    for provider in ("google", "github", "apple"):
        if data.get(f"EXTERNAL_{provider.upper()}_ENABLED", False):
            enabled_providers.append(provider)

    return AuthOutputs(
        project_ref=project_ref,
        site_url=data.get("SITE_URL", ""),
        disable_signup=data.get("DISABLE_SIGNUP", False),
        jwt_expiry=data.get("JWT_EXP", 3600),
        external_email_enabled=data.get("EXTERNAL_EMAIL_ENABLED", True),
        external_phone_enabled=data.get("EXTERNAL_PHONE_ENABLED", False),
        mailer_autoconfirm=data.get("MAILER_AUTOCONFIRM", False),
        external_providers_enabled=enabled_providers,
    )


_AUTH_DEFAULTS: dict[str, Any] = {
    "DISABLE_SIGNUP": False,
    "JWT_EXP": 3600,
    "EXTERNAL_EMAIL_ENABLED": True,
    "EXTERNAL_PHONE_ENABLED": False,
    "MAILER_AUTOCONFIRM": False,
    "EXTERNAL_GOOGLE_ENABLED": False,
    "EXTERNAL_GITHUB_ENABLED": False,
    "EXTERNAL_APPLE_ENABLED": False,
}


class Auth(Resource[AuthConfig, AuthOutputs]):
    """Supabase Auth configuration resource.

    Manages authentication settings for a Supabase project via the
    Management API. This resource treats auth configuration as a
    declarative resource: on_create applies settings, on_update patches
    changes, and on_delete resets to Supabase defaults.

    Lifecycle:
        - on_create: Applies the configured auth settings to the project.
        - on_update: Patches changed settings. Only modified fields are sent.
        - on_delete: Resets auth settings to Supabase defaults (disables
          external providers, re-enables signups, etc.).

    Example::

        resources:
          - name: auth-config
            provider: supabase
            type: auth
            config:
              access_token:
                provider: pragma
                resource: secret
                name: supabase-token
                field: outputs.value
              project_ref:
                provider: supabase
                resource: project
                name: my-app
                field: outputs.project_ref
              site_url: "https://myapp.example.com"
              disable_signup: false
              jwt_expiry: 3600
              external_email_enabled: true
              mailer_autoconfirm: false
              external_github:
                enabled: true
                client_id: "gh-client-id"
                secret: "gh-client-secret"
    """

    async def _apply_config(self) -> AuthOutputs:
        """Patch auth configuration and return current state.

        Returns:
            AuthOutputs reflecting the applied configuration.
        """
        client = create_management_client(self.config.access_token)

        try:
            patch_body = _build_auth_patch(self.config)
            response = await client.patch(
                f"/projects/{self.config.project_ref}/config/auth",
                json=patch_body,
            )
            await raise_for_status(response)

            get_response = await client.get(f"/projects/{self.config.project_ref}/config/auth")
            await raise_for_status(get_response)

            return _build_outputs_from_api(self.config.project_ref, get_response.json())
        finally:
            await client.aclose()

    async def on_create(self) -> AuthOutputs:
        """Apply auth configuration to the Supabase project.

        Returns:
            AuthOutputs reflecting the applied configuration.
        """
        return await self._apply_config()

    async def on_update(self, previous_config: AuthConfig) -> AuthOutputs:
        """Patch auth configuration with changed settings.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            AuthOutputs reflecting the updated configuration.
        """
        return await self._apply_config()

    async def on_delete(self) -> None:
        """Reset auth configuration to Supabase defaults."""
        client = create_management_client(self.config.access_token)

        try:
            response = await client.patch(
                f"/projects/{self.config.project_ref}/config/auth",
                json=_AUTH_DEFAULTS,
            )
            await raise_for_status(response)
        finally:
            await client.aclose()

    @classmethod
    def upgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs

    @classmethod
    def downgrade(cls, config: dict, outputs: dict) -> tuple[dict, dict]:  # noqa: D102
        return config, outputs
