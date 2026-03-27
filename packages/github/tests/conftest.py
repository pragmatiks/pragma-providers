"""Pytest configuration for GitHub provider tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from pragma_sdk.context import reset_provider_name, set_provider_name
from pragma_sdk.provider import ProviderHarness


@pytest.fixture(autouse=True)
def provider_context() -> Iterator[None]:
    """Set provider name context for tests that call lifecycle methods directly."""
    token = set_provider_name("github")
    yield
    reset_provider_name(token)


@pytest.fixture
def harness() -> ProviderHarness:
    """Test harness for invoking lifecycle methods."""
    return ProviderHarness()


@pytest.fixture
def sample_repository_data() -> dict[str, Any]:
    """Sample GitHub repository API response data."""
    return {
        "id": 123456789,
        "name": "my-repo",
        "full_name": "my-org/my-repo",
        "html_url": "https://github.com/my-org/my-repo",
        "clone_url": "https://github.com/my-org/my-repo.git",
        "ssh_url": "git@github.com:my-org/my-repo.git",
        "default_branch": "main",
        "visibility": "private",
        "description": "My application repository",
        "private": True,
        "has_issues": True,
        "has_wiki": False,
        "has_projects": False,
    }


@pytest.fixture
def sample_environment_data() -> dict[str, Any]:
    """Sample GitHub environment API response data."""
    return {
        "id": 987654,
        "name": "production",
        "html_url": "https://github.com/my-org/my-repo/settings/environments/987654",
        "created_at": "2026-03-27T12:00:00Z",
        "updated_at": "2026-03-27T12:00:00Z",
        "protection_rules": [],
    }


@pytest.fixture
def sample_environment_data_with_rules() -> dict[str, Any]:
    """Sample GitHub environment API response data with protection rules."""
    return {
        "id": 987654,
        "name": "production",
        "html_url": "https://github.com/my-org/my-repo/settings/environments/987654",
        "created_at": "2026-03-27T12:00:00Z",
        "updated_at": "2026-03-27T12:00:00Z",
        "protection_rules": [
            {
                "id": 1,
                "type": "wait_timer",
                "wait_timer": 30,
            },
            {
                "id": 2,
                "type": "required_reviewers",
                "reviewers": [
                    {"type": "User", "reviewer": {"id": 12345, "login": "octocat"}},
                ],
            },
        ],
    }


@pytest.fixture
def sample_public_key_data() -> dict[str, Any]:
    """Sample GitHub public key API response data.

    Uses a valid 32-byte NaCl public key encoded in base64.
    """
    return {
        "key_id": "568250167242549743",
        "key": "Hkp5sEoLsgcGxfCyKo5rJhHaM+2mhn3ULQC2olDB2LY=",
    }


@pytest.fixture
def sample_secret_metadata() -> dict[str, Any]:
    """Sample GitHub secret metadata API response data."""
    return {
        "name": "DEPLOY_KEY",
        "created_at": "2026-03-27T12:00:00Z",
        "updated_at": "2026-03-27T12:00:00Z",
    }
