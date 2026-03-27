"""Pytest configuration for Vercel provider tests."""

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
    token = set_provider_name("vercel")
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
    """Sample Vercel project API response data."""
    return {
        "id": "prj_abc123def456",
        "name": "my-app",
        "accountId": "team_xyz789",
        "framework": "nextjs",
        "createdAt": 1711540800000,
        "updatedAt": 1711540800000,
    }


@pytest.fixture
def sample_deployment_data() -> dict[str, Any]:
    """Sample Vercel deployment API response data."""
    return {
        "id": "dpl_abc123def456",
        "url": "my-app-abc123.vercel.app",
        "state": "READY",
        "readyState": "READY",
        "projectId": "prj_abc123def456",
        "createdAt": 1711540800000,
    }


@pytest.fixture
def sample_domain_data() -> dict[str, Any]:
    """Sample Vercel domain API response data."""
    return {
        "name": "myapp.example.com",
        "verified": True,
        "redirect": None,
        "gitBranch": None,
        "createdAt": 1711540800000,
        "updatedAt": 1711540800000,
    }


@pytest.fixture
def sample_env_list_data() -> dict[str, Any]:
    """Sample Vercel environment variables list API response data."""
    return {
        "envs": [
            {
                "id": "env_abc123",
                "key": "DATABASE_URL",
                "value": "postgres://localhost",
                "target": ["production"],
                "type": "encrypted",
            },
        ],
    }
