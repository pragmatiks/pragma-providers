"""Tests for Vercel Domain resource."""

from __future__ import annotations

import json
from typing import Any

import respx
from pragma_sdk.provider import ProviderHarness

from vercel_provider import Domain, DomainConfig, DomainOutputs


_BASE_URL = "https://api.vercel.com"


async def test_create_domain_success(
    harness: ProviderHarness,
    sample_domain_data: dict[str, Any],
) -> None:
    """on_create adds domain to the project and returns outputs."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v10/projects/prj_abc123def456/domains").respond(200, json=sample_domain_data)

        config = DomainConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            domain="myapp.example.com",
        )

        result = await harness.invoke_create(Domain, name="my-domain", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.domain == "myapp.example.com"
    assert result.outputs.project_id == "prj_abc123def456"
    assert result.outputs.verified is True


async def test_create_domain_with_redirect(
    harness: ProviderHarness,
    sample_domain_data: dict[str, Any],
) -> None:
    """on_create includes redirect configuration in request body."""
    redirect_data = {**sample_domain_data, "redirect": "www.example.com"}

    with respx.mock(base_url=_BASE_URL) as mock:
        create_route = mock.post("/v10/projects/prj_abc123def456/domains").respond(200, json=redirect_data)

        config = DomainConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            domain="example.com",
            redirect="www.example.com",
            redirect_status_code=301,
        )

        result = await harness.invoke_create(Domain, name="my-domain", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.redirect == "www.example.com"

    request_body = json.loads(create_route.calls[0].request.content)
    assert request_body["redirect"] == "www.example.com"
    assert request_body["redirectStatusCode"] == 301


async def test_create_domain_with_git_branch(
    harness: ProviderHarness,
    sample_domain_data: dict[str, Any],
) -> None:
    """on_create includes git branch in request body."""
    branch_data = {**sample_domain_data, "gitBranch": "staging"}

    with respx.mock(base_url=_BASE_URL) as mock:
        create_route = mock.post("/v10/projects/prj_abc123def456/domains").respond(200, json=branch_data)

        config = DomainConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            domain="staging.example.com",
            git_branch="staging",
        )

        result = await harness.invoke_create(Domain, name="my-domain", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.git_branch == "staging"

    request_body = json.loads(create_route.calls[0].request.content)
    assert request_body["gitBranch"] == "staging"


async def test_create_domain_unverified(
    harness: ProviderHarness,
) -> None:
    """on_create returns verified=False when domain needs verification."""
    unverified_data = {
        "name": "custom.example.com",
        "verified": False,
        "verification": [
            {"type": "TXT", "domain": "_vercel.custom.example.com", "value": "vc-domain-verify=abc123"},
        ],
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v10/projects/prj_abc123def456/domains").respond(200, json=unverified_data)

        config = DomainConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            domain="custom.example.com",
        )

        result = await harness.invoke_create(Domain, name="my-domain", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.verified is False


async def test_create_domain_api_error(
    harness: ProviderHarness,
) -> None:
    """on_create propagates API errors."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v10/projects/prj_abc123def456/domains").respond(
            400, json={"error": {"message": "Domain already exists"}}
        )

        config = DomainConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            domain="existing.example.com",
        )

        result = await harness.invoke_create(Domain, name="my-domain", config=config)

    assert result.failed
    assert "Domain already exists" in str(result.error)


async def test_create_domain_with_team_id(
    harness: ProviderHarness,
    sample_domain_data: dict[str, Any],
) -> None:
    """on_create includes team_id as query parameter."""
    with respx.mock(base_url=_BASE_URL) as mock:
        create_route = mock.post("/v10/projects/prj_abc123def456/domains").respond(200, json=sample_domain_data)

        config = DomainConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            domain="myapp.example.com",
            team_id="team_xyz789",
        )

        result = await harness.invoke_create(Domain, name="my-domain", config=config)

    assert result.success
    assert "teamId=team_xyz789" in str(create_route.calls[0].request.url)


async def test_update_domain_success(
    harness: ProviderHarness,
    sample_domain_data: dict[str, Any],
) -> None:
    """on_update removes and re-adds the domain."""
    updated_data = {**sample_domain_data, "gitBranch": "develop"}

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/v9/projects/prj_abc123def456/domains/myapp.example.com").respond(200, json={})
        mock.post("/v10/projects/prj_abc123def456/domains").respond(200, json=updated_data)

        previous = DomainConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            domain="myapp.example.com",
        )
        current = DomainConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            domain="myapp.example.com",
            git_branch="develop",
        )
        existing_outputs = DomainOutputs(
            domain="myapp.example.com",
            project_id="prj_abc123def456",
            verified=True,
            redirect="",
            git_branch="",
        )

        result = await harness.invoke_update(
            Domain,
            name="my-domain",
            config=current,
            previous_config=previous,
            current_outputs=existing_outputs,
        )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.git_branch == "develop"


async def test_delete_domain_success(
    harness: ProviderHarness,
) -> None:
    """on_delete removes the domain from the project."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/v9/projects/prj_abc123def456/domains/myapp.example.com").respond(200, json={})

        config = DomainConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            domain="myapp.example.com",
        )
        existing_outputs = DomainOutputs(
            domain="myapp.example.com",
            project_id="prj_abc123def456",
            verified=True,
            redirect="",
            git_branch="",
        )

        result = await harness.invoke_delete(
            Domain,
            name="my-domain",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_domain_already_gone(
    harness: ProviderHarness,
) -> None:
    """on_delete is idempotent when domain is not attached."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/v9/projects/prj_abc123def456/domains/myapp.example.com").respond(
            404, json={"error": {"message": "Domain not found"}}
        )

        config = DomainConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            domain="myapp.example.com",
        )
        existing_outputs = DomainOutputs(
            domain="myapp.example.com",
            project_id="prj_abc123def456",
            verified=True,
            redirect="",
            git_branch="",
        )

        result = await harness.invoke_delete(
            Domain,
            name="my-domain",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_domain_no_outputs(
    harness: ProviderHarness,
) -> None:
    """on_delete succeeds silently when no outputs exist."""
    config = DomainConfig(
        access_token="test-token",
        project_id="prj_abc123def456",
        domain="myapp.example.com",
    )

    with respx.mock(base_url=_BASE_URL):
        result = await harness.invoke_delete(
            Domain,
            name="my-domain",
            config=config,
        )

    assert result.success


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Domain.resource == "domain"
