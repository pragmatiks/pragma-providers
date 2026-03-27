"""Tests for Vercel Deployment resource."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import respx
from pragma_sdk.provider import ProviderHarness

from vercel_provider import Deployment, DeploymentConfig, DeploymentOutputs


if TYPE_CHECKING:
    from pytest_mock import MockType


_BASE_URL = "https://api.vercel.com"


async def test_create_deployment_success(
    harness: ProviderHarness,
    mock_sleep: MockType,
    sample_deployment_data: dict[str, Any],
) -> None:
    """on_create triggers deployment and returns outputs after ready."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v13/deployments").respond(200, json=sample_deployment_data)
        mock.get("/v13/deployments/dpl_abc123def456").respond(200, json=sample_deployment_data)

        config = DeploymentConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            project_name="my-app",
            target="production",
        )

        result = await harness.invoke_create(Deployment, name="my-app-deploy", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.deployment_id == "dpl_abc123def456"
    assert result.outputs.url == "my-app-abc123.vercel.app"
    assert result.outputs.state == "READY"
    assert result.outputs.ready_state == "READY"


async def test_create_deployment_with_git_ref(
    harness: ProviderHarness,
    mock_sleep: MockType,
    sample_deployment_data: dict[str, Any],
) -> None:
    """on_create includes git source when git_ref is provided."""
    with respx.mock(base_url=_BASE_URL) as mock:
        create_route = mock.post("/v13/deployments").respond(200, json=sample_deployment_data)
        mock.get("/v13/deployments/dpl_abc123def456").respond(200, json=sample_deployment_data)

        config = DeploymentConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            project_name="my-app",
            git_ref="main",
            target="production",
        )

        result = await harness.invoke_create(Deployment, name="my-app-deploy", config=config)

    assert result.success

    request_body = json.loads(create_route.calls[0].request.content)
    assert request_body["gitSource"] == {"ref": "main", "type": "github"}


async def test_create_deployment_polls_until_ready(
    harness: ProviderHarness,
    mock_sleep: MockType,
) -> None:
    """on_create polls deployment status until READY."""
    building_data = {
        "id": "dpl_abc123def456",
        "url": "my-app-abc123.vercel.app",
        "state": "BUILDING",
        "readyState": "BUILDING",
        "projectId": "prj_abc123def456",
    }
    ready_data = {
        "id": "dpl_abc123def456",
        "url": "my-app-abc123.vercel.app",
        "state": "READY",
        "readyState": "READY",
        "projectId": "prj_abc123def456",
    }

    call_count = 0

    def get_side_effect(request: httpx.Request) -> respx.MockResponse:
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            return respx.MockResponse(200, json=building_data)
        return respx.MockResponse(200, json=ready_data)

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v13/deployments").respond(200, json=building_data)
        mock.get("/v13/deployments/dpl_abc123def456").mock(side_effect=get_side_effect)

        config = DeploymentConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            project_name="my-app",
        )

        result = await harness.invoke_create(Deployment, name="my-app-deploy", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.ready_state == "READY"
    assert mock_sleep.call_count == 2


async def test_create_deployment_timeout(
    harness: ProviderHarness,
    mock_sleep: MockType,
) -> None:
    """on_create raises TimeoutError when deployment never completes."""
    building_data = {
        "id": "dpl_abc123def456",
        "url": "my-app-abc123.vercel.app",
        "state": "BUILDING",
        "readyState": "BUILDING",
        "projectId": "prj_abc123def456",
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v13/deployments").respond(200, json=building_data)
        mock.get("/v13/deployments/dpl_abc123def456").respond(200, json=building_data)

        config = DeploymentConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            project_name="my-app",
        )

        result = await harness.invoke_create(Deployment, name="my-app-deploy", config=config)

    assert result.failed
    assert "did not complete" in str(result.error)


async def test_create_deployment_api_error(
    harness: ProviderHarness,
) -> None:
    """on_create propagates API errors."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v13/deployments").respond(400, json={"error": {"message": "Invalid project"}})

        config = DeploymentConfig(
            access_token="test-token",
            project_id="prj_nonexistent",
            project_name="bad-project",
        )

        result = await harness.invoke_create(Deployment, name="my-app-deploy", config=config)

    assert result.failed
    assert "Invalid project" in str(result.error)


async def test_create_deployment_poll_fails_fast_on_401(
    harness: ProviderHarness,
    mock_sleep: MockType,
) -> None:
    """on_create fails fast when poll returns 401."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v13/deployments").respond(
            200,
            json={
                "id": "dpl_abc123def456",
                "url": "my-app-abc123.vercel.app",
                "state": "BUILDING",
                "readyState": "BUILDING",
                "projectId": "prj_abc123def456",
            },
        )
        mock.get("/v13/deployments/dpl_abc123def456").respond(401, json={"error": {"message": "Invalid token"}})

        config = DeploymentConfig(
            access_token="bad-token",
            project_id="prj_abc123def456",
            project_name="my-app",
        )

        result = await harness.invoke_create(Deployment, name="my-app-deploy", config=config)

    assert result.failed
    assert "Invalid token" in str(result.error)
    assert mock_sleep.call_count == 0


async def test_create_deployment_error_state(
    harness: ProviderHarness,
    mock_sleep: MockType,
) -> None:
    """on_create fails when deployment reaches ERROR state."""
    error_data = {
        "id": "dpl_abc123def456",
        "url": "my-app-abc123.vercel.app",
        "state": "ERROR",
        "readyState": "ERROR",
        "projectId": "prj_abc123def456",
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v13/deployments").respond(200, json=error_data)
        mock.get("/v13/deployments/dpl_abc123def456").respond(200, json=error_data)

        config = DeploymentConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            project_name="my-app",
        )

        result = await harness.invoke_create(Deployment, name="my-app-deploy", config=config)

    assert result.failed
    assert "reached terminal state ERROR" in str(result.error)


async def test_create_deployment_canceled_state(
    harness: ProviderHarness,
    mock_sleep: MockType,
) -> None:
    """on_create fails when deployment reaches CANCELED state."""
    canceled_data = {
        "id": "dpl_abc123def456",
        "url": "my-app-abc123.vercel.app",
        "state": "CANCELED",
        "readyState": "CANCELED",
        "projectId": "prj_abc123def456",
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v13/deployments").respond(200, json=canceled_data)
        mock.get("/v13/deployments/dpl_abc123def456").respond(200, json=canceled_data)

        config = DeploymentConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            project_name="my-app",
        )

        result = await harness.invoke_create(Deployment, name="my-app-deploy", config=config)

    assert result.failed
    assert "reached terminal state CANCELED" in str(result.error)


async def test_update_deployment_triggers_new(
    harness: ProviderHarness,
    mock_sleep: MockType,
    sample_deployment_data: dict[str, Any],
) -> None:
    """on_update triggers a new deployment."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v13/deployments").respond(200, json=sample_deployment_data)
        mock.get("/v13/deployments/dpl_abc123def456").respond(200, json=sample_deployment_data)

        previous = DeploymentConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            project_name="my-app",
            git_ref="v1.0.0",
        )
        current = DeploymentConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            project_name="my-app",
            git_ref="v2.0.0",
        )
        existing_outputs = DeploymentOutputs(
            deployment_id="dpl_old",
            url="my-app-old.vercel.app",
            state="READY",
            ready_state="READY",
            project_id="prj_abc123def456",
        )

        result = await harness.invoke_update(
            Deployment,
            name="my-app-deploy",
            config=current,
            previous_config=previous,
            current_outputs=existing_outputs,
        )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.deployment_id == "dpl_abc123def456"


async def test_delete_deployment_success(
    harness: ProviderHarness,
) -> None:
    """on_delete removes the deployment."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/v13/deployments/dpl_abc123def456").respond(200, json={})

        config = DeploymentConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            project_name="my-app",
        )
        existing_outputs = DeploymentOutputs(
            deployment_id="dpl_abc123def456",
            url="my-app-abc123.vercel.app",
            state="READY",
            ready_state="READY",
            project_id="prj_abc123def456",
        )

        result = await harness.invoke_delete(
            Deployment,
            name="my-app-deploy",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_deployment_already_gone(
    harness: ProviderHarness,
) -> None:
    """on_delete is idempotent when deployment does not exist."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/v13/deployments/dpl_abc123def456").respond(
            404, json={"error": {"message": "Deployment not found"}}
        )

        config = DeploymentConfig(
            access_token="test-token",
            project_id="prj_abc123def456",
            project_name="my-app",
        )
        existing_outputs = DeploymentOutputs(
            deployment_id="dpl_abc123def456",
            url="my-app-abc123.vercel.app",
            state="READY",
            ready_state="READY",
            project_id="prj_abc123def456",
        )

        result = await harness.invoke_delete(
            Deployment,
            name="my-app-deploy",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_deployment_no_outputs(
    harness: ProviderHarness,
) -> None:
    """on_delete succeeds silently when no outputs exist."""
    config = DeploymentConfig(
        access_token="test-token",
        project_id="prj_abc123def456",
        project_name="my-app",
    )

    with respx.mock(base_url=_BASE_URL):
        result = await harness.invoke_delete(
            Deployment,
            name="my-app-deploy",
            config=config,
        )

    assert result.success


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Deployment.resource == "deployment"
