"""Pytest configuration for supabase provider tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest
from pragma_sdk.context import reset_provider_name, set_provider_name
from pragma_sdk.provider import ProviderHarness


if TYPE_CHECKING:
    from pytest_mock import MockerFixture, MockType


@pytest.fixture(autouse=True)
def provider_context() -> Iterator[None]:
    """Set provider name context for tests that call lifecycle methods directly."""
    token = set_provider_name("supabase")
    yield
    reset_provider_name(token)


@pytest.fixture
def harness() -> ProviderHarness:
    """Test harness for invoking lifecycle methods."""
    return ProviderHarness()


@pytest.fixture
def mock_sleep(mocker: MockerFixture) -> MockType:
    """Mock asyncio.sleep to avoid real delays in polling tests.

    Returns:
        AsyncMock that replaces asyncio.sleep.
    """
    return mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)


@pytest.fixture
def sample_project_data() -> dict[str, Any]:
    """Sample Supabase project API response data."""
    return {
        "id": "abcdefghijklmnop",
        "name": "test-project",
        "organization_id": "org_test123",
        "region": "eu-west-1",
        "status": "ACTIVE_HEALTHY",
        "created_at": "2026-03-27T12:00:00Z",
    }


@pytest.fixture
def sample_api_keys() -> list[dict[str, str]]:
    """Sample Supabase API keys response data."""
    return [
        {"name": "anon", "api_key": "eyJ-anon-key-test"},
        {"name": "service_role", "api_key": "eyJ-service-role-key-test"},
    ]


@pytest.fixture
def sample_health_data() -> list[dict[str, str]]:
    """Sample Supabase project health response data."""
    return [
        {"name": "auth", "status": "ACTIVE_HEALTHY"},
        {"name": "rest", "status": "ACTIVE_HEALTHY"},
        {"name": "realtime", "status": "ACTIVE_HEALTHY"},
        {"name": "storage", "status": "ACTIVE_HEALTHY"},
        {"name": "db", "status": "ACTIVE_HEALTHY"},
    ]


@pytest.fixture
def sample_auth_config_data() -> dict[str, Any]:
    """Sample Supabase auth config API response data."""
    return {
        "SITE_URL": "https://myapp.example.com",
        "DISABLE_SIGNUP": False,
        "JWT_EXP": 3600,
        "EXTERNAL_EMAIL_ENABLED": True,
        "EXTERNAL_PHONE_ENABLED": False,
        "MAILER_AUTOCONFIRM": False,
        "EXTERNAL_GOOGLE_ENABLED": True,
        "EXTERNAL_GOOGLE_CLIENT_ID": "google-client-id",
        "EXTERNAL_GITHUB_ENABLED": False,
        "EXTERNAL_APPLE_ENABLED": False,
        "URI_ALLOW_LIST": "",
    }
