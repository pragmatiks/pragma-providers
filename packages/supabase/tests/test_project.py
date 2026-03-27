"""Tests for Supabase Project resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import respx
from pragma_sdk.provider import ProviderHarness

from supabase_provider import Project, ProjectConfig, ProjectOutputs


if TYPE_CHECKING:
    pass


_BASE_URL = "https://api.supabase.com/v1"


async def test_create_project_success(
    harness: ProviderHarness,
    sample_project_data: dict[str, Any],
    sample_api_keys: list[dict[str, str]],
    sample_health_data: list[dict[str, str]],
) -> None:
    """on_create creates project and returns outputs after healthy."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/projects").respond(200, json=sample_project_data)
        mock.get("/projects/abcdefghijklmnop/health").respond(200, json=sample_health_data)
        mock.get("/projects/abcdefghijklmnop").respond(200, json=sample_project_data)
        mock.get("/projects/abcdefghijklmnop/api-keys").respond(200, json=sample_api_keys)

        config = ProjectConfig(
            access_token="sbp_test_token",
            organization_id="org_test123",
            name="test-project",
            region="eu-west-1",
            database_password="secure-password-123",
        )

        result = await harness.invoke_create(Project, name="test-project", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.project_ref == "abcdefghijklmnop"
    assert result.outputs.name == "test-project"
    assert result.outputs.region == "eu-west-1"
    assert result.outputs.endpoint == "https://abcdefghijklmnop.supabase.co"
    assert result.outputs.anon_key == "eyJ-anon-key-test"
    assert result.outputs.service_role_key == "eyJ-service-role-key-test"


async def test_create_project_api_error(
    harness: ProviderHarness,
) -> None:
    """on_create propagates API errors."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/projects").respond(400, json={"message": "Invalid region"})

        config = ProjectConfig(
            access_token="sbp_test_token",
            organization_id="org_test123",
            name="test-project",
            region="invalid-region",
            database_password="secure-password-123",
        )

        result = await harness.invoke_create(Project, name="test-project", config=config)

    assert result.failed
    assert "Invalid region" in str(result.error)


async def test_update_project_name_changed(
    harness: ProviderHarness,
    sample_project_data: dict[str, Any],
    sample_api_keys: list[dict[str, str]],
) -> None:
    """on_update patches project when name changes."""
    updated_data = {**sample_project_data, "name": "renamed-project"}

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.patch("/projects/abcdefghijklmnop").respond(200, json=updated_data)
        mock.get("/projects/abcdefghijklmnop").respond(200, json=updated_data)
        mock.get("/projects/abcdefghijklmnop/api-keys").respond(200, json=sample_api_keys)

        previous = ProjectConfig(
            access_token="sbp_test_token",
            organization_id="org_test123",
            name="test-project",
            region="eu-west-1",
            database_password="secure-password-123",
        )
        current = ProjectConfig(
            access_token="sbp_test_token",
            organization_id="org_test123",
            name="renamed-project",
            region="eu-west-1",
            database_password="secure-password-123",
        )
        existing_outputs = ProjectOutputs(
            project_ref="abcdefghijklmnop",
            name="test-project",
            organization_id="org_test123",
            region="eu-west-1",
            status="ACTIVE_HEALTHY",
            endpoint="https://abcdefghijklmnop.supabase.co",
            anon_key="eyJ-anon-key-test",
            service_role_key="eyJ-service-role-key-test",
        )

        result = await harness.invoke_update(
            Project,
            name="test-project",
            config=current,
            previous_config=previous,
            current_outputs=existing_outputs,
        )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.name == "renamed-project"


async def test_update_project_no_change(
    harness: ProviderHarness,
) -> None:
    """on_update returns existing outputs when nothing changed."""
    config = ProjectConfig(
        access_token="sbp_test_token",
        organization_id="org_test123",
        name="test-project",
        region="eu-west-1",
        database_password="secure-password-123",
    )
    existing_outputs = ProjectOutputs(
        project_ref="abcdefghijklmnop",
        name="test-project",
        organization_id="org_test123",
        region="eu-west-1",
        status="ACTIVE_HEALTHY",
        endpoint="https://abcdefghijklmnop.supabase.co",
        anon_key="eyJ-anon-key-test",
        service_role_key="eyJ-service-role-key-test",
    )

    with respx.mock(base_url=_BASE_URL):
        result = await harness.invoke_update(
            Project,
            name="test-project",
            config=config,
            previous_config=config,
            current_outputs=existing_outputs,
        )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.project_ref == "abcdefghijklmnop"


async def test_delete_project_success(
    harness: ProviderHarness,
) -> None:
    """on_delete removes the project."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/projects/abcdefghijklmnop").respond(200, json={})

        config = ProjectConfig(
            access_token="sbp_test_token",
            organization_id="org_test123",
            name="test-project",
            region="eu-west-1",
            database_password="secure-password-123",
        )
        existing_outputs = ProjectOutputs(
            project_ref="abcdefghijklmnop",
            name="test-project",
            organization_id="org_test123",
            region="eu-west-1",
            status="ACTIVE_HEALTHY",
            endpoint="https://abcdefghijklmnop.supabase.co",
            anon_key="eyJ-anon-key-test",
            service_role_key="eyJ-service-role-key-test",
        )

        result = await harness.invoke_delete(
            Project,
            name="test-project",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_project_already_gone(
    harness: ProviderHarness,
) -> None:
    """on_delete is idempotent when project does not exist."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/projects/abcdefghijklmnop").respond(404, json={"message": "Project not found"})

        config = ProjectConfig(
            access_token="sbp_test_token",
            organization_id="org_test123",
            name="test-project",
            region="eu-west-1",
            database_password="secure-password-123",
        )
        existing_outputs = ProjectOutputs(
            project_ref="abcdefghijklmnop",
            name="test-project",
            organization_id="org_test123",
            region="eu-west-1",
            status="ACTIVE_HEALTHY",
            endpoint="https://abcdefghijklmnop.supabase.co",
            anon_key="eyJ-anon-key-test",
            service_role_key="eyJ-service-role-key-test",
        )

        result = await harness.invoke_delete(
            Project,
            name="test-project",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_project_no_outputs(
    harness: ProviderHarness,
) -> None:
    """on_delete succeeds silently when no outputs exist."""
    config = ProjectConfig(
        access_token="sbp_test_token",
        organization_id="org_test123",
        name="test-project",
        region="eu-west-1",
        database_password="secure-password-123",
    )

    with respx.mock(base_url=_BASE_URL):
        result = await harness.invoke_delete(
            Project,
            name="test-project",
            config=config,
        )

    assert result.success


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Project.resource == "project"
