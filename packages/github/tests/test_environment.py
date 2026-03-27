"""Tests for GitHub Environment resource."""

from __future__ import annotations

import json
from typing import Any

import respx
from pragma_sdk.provider import ProviderHarness

from github_provider import (
    Environment,
    EnvironmentConfig,
    EnvironmentOutputs,
    ProtectionRulesConfig,
    ReviewerConfig,
)


_BASE_URL = "https://api.github.com"


async def test_create_environment_success(
    harness: ProviderHarness,
    sample_environment_data: dict[str, Any],
) -> None:
    """on_create creates environment and returns outputs."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.put("/repos/my-org/my-repo/environments/production").respond(200, json=sample_environment_data)

        config = EnvironmentConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            environment_name="production",
        )

        result = await harness.invoke_create(Environment, name="production-env", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.environment_id == 987654
    assert result.outputs.environment_name == "production"
    assert result.outputs.wait_timer == 0
    assert result.outputs.reviewers_count == 0


async def test_create_environment_with_protection_rules(
    harness: ProviderHarness,
    sample_environment_data_with_rules: dict[str, Any],
) -> None:
    """on_create includes protection rules in the request body."""
    with respx.mock(base_url=_BASE_URL) as mock:
        create_route = mock.put("/repos/my-org/my-repo/environments/production").respond(
            200, json=sample_environment_data_with_rules
        )

        config = EnvironmentConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            environment_name="production",
            protection_rules=ProtectionRulesConfig(
                wait_timer=30,
                reviewers=[ReviewerConfig(reviewer_type="User", reviewer_id=12345)],
            ),
        )

        result = await harness.invoke_create(Environment, name="production-env", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.wait_timer == 30
    assert result.outputs.reviewers_count == 1

    request_body = json.loads(create_route.calls[0].request.content)
    assert request_body["wait_timer"] == 30
    assert request_body["reviewers"] == [{"type": "User", "id": 12345}]


async def test_create_environment_api_error(
    harness: ProviderHarness,
) -> None:
    """on_create propagates API errors."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.put("/repos/my-org/my-repo/environments/production").respond(422, json={"message": "Validation Failed"})

        config = EnvironmentConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            environment_name="production",
        )

        result = await harness.invoke_create(Environment, name="production-env", config=config)

    assert result.failed
    assert "Validation Failed" in str(result.error)


async def test_update_environment_reapplies_configuration(
    harness: ProviderHarness,
    sample_environment_data_with_rules: dict[str, Any],
) -> None:
    """on_update re-applies environment configuration via PUT."""
    with respx.mock(base_url=_BASE_URL) as mock:
        put_route = mock.put("/repos/my-org/my-repo/environments/production").respond(
            200, json=sample_environment_data_with_rules
        )

        previous = EnvironmentConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            environment_name="production",
        )
        current = EnvironmentConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            environment_name="production",
            protection_rules=ProtectionRulesConfig(wait_timer=30),
        )
        existing_outputs = EnvironmentOutputs(
            environment_id=987654,
            environment_name="production",
            html_url="https://github.com/my-org/my-repo/settings/environments/987654",
            wait_timer=0,
            reviewers_count=0,
        )

        result = await harness.invoke_update(
            Environment,
            name="production-env",
            config=current,
            previous_config=previous,
            current_outputs=existing_outputs,
        )

    assert result.success
    assert result.outputs is not None
    assert put_route.called


async def test_delete_environment_success(
    harness: ProviderHarness,
) -> None:
    """on_delete removes the environment."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/repos/my-org/my-repo/environments/production").respond(204)

        config = EnvironmentConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            environment_name="production",
        )

        result = await harness.invoke_delete(Environment, name="production-env", config=config)

    assert result.success


async def test_delete_environment_already_gone(
    harness: ProviderHarness,
) -> None:
    """on_delete is idempotent when environment does not exist."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/repos/my-org/my-repo/environments/production").respond(404, json={"message": "Not Found"})

        config = EnvironmentConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            environment_name="production",
        )

        result = await harness.invoke_delete(Environment, name="production-env", config=config)

    assert result.success


async def test_create_environment_no_protection_rules_sends_empty_body(
    harness: ProviderHarness,
    sample_environment_data: dict[str, Any],
) -> None:
    """on_create sends empty body when no protection rules are configured."""
    with respx.mock(base_url=_BASE_URL) as mock:
        create_route = mock.put("/repos/my-org/my-repo/environments/staging").respond(
            200, json={**sample_environment_data, "name": "staging"}
        )

        config = EnvironmentConfig(
            access_token="ghp_test_token",
            owner="my-org",
            repository="my-repo",
            environment_name="staging",
        )

        result = await harness.invoke_create(Environment, name="staging-env", config=config)

    assert result.success

    request_body = json.loads(create_route.calls[0].request.content)
    assert request_body == {}


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Environment.resource == "environment"
