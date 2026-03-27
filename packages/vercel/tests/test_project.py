"""Tests for Vercel Project resource."""

from __future__ import annotations

import json
from typing import Any

import respx
from pragma_sdk.provider import ProviderHarness

from vercel_provider import EnvironmentVariableConfig, GitRepositoryConfig, Project, ProjectConfig, ProjectOutputs


_BASE_URL = "https://api.vercel.com"


async def test_create_project_success(
    harness: ProviderHarness,
    sample_project_data: dict[str, Any],
) -> None:
    """on_create creates project and returns outputs."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v10/projects").respond(200, json=sample_project_data)

        config = ProjectConfig(
            access_token="test-token",
            name="my-app",
            framework="nextjs",
        )

        result = await harness.invoke_create(Project, name="my-app", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.project_id == "prj_abc123def456"
    assert result.outputs.name == "my-app"
    assert result.outputs.framework == "nextjs"
    assert result.outputs.url == "https://my-app.vercel.app"


async def test_create_project_with_git_repository(
    harness: ProviderHarness,
    sample_project_data: dict[str, Any],
) -> None:
    """on_create includes git repository in request body."""
    with respx.mock(base_url=_BASE_URL) as mock:
        create_route = mock.post("/v10/projects").respond(200, json=sample_project_data)

        config = ProjectConfig(
            access_token="test-token",
            name="my-app",
            framework="nextjs",
            git_repository=GitRepositoryConfig(repo="my-org/my-app", repo_type="github"),
        )

        result = await harness.invoke_create(Project, name="my-app", config=config)

    assert result.success

    request_body = json.loads(create_route.calls[0].request.content)
    assert request_body["gitRepository"] == {"repo": "my-org/my-app", "type": "github"}


async def test_create_project_with_environment_variables(
    harness: ProviderHarness,
    sample_project_data: dict[str, Any],
) -> None:
    """on_create syncs environment variables after project creation."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v10/projects").respond(200, json=sample_project_data)
        mock.get("/v9/projects/prj_abc123def456/env").respond(200, json={"envs": []})
        env_route = mock.post("/v10/projects/prj_abc123def456/env").respond(200, json={"created": []})

        config = ProjectConfig(
            access_token="test-token",
            name="my-app",
            environment_variables=[
                EnvironmentVariableConfig(key="DATABASE_URL", value="postgres://localhost", target=["production"]),
            ],
        )

        result = await harness.invoke_create(Project, name="my-app", config=config)

    assert result.success

    env_request_body = json.loads(env_route.calls[0].request.content)
    assert len(env_request_body) == 1
    assert env_request_body[0]["key"] == "DATABASE_URL"
    assert env_request_body[0]["value"] == "postgres://localhost"
    assert env_request_body[0]["target"] == ["production"]


async def test_create_project_with_team_id(
    harness: ProviderHarness,
    sample_project_data: dict[str, Any],
) -> None:
    """on_create includes team_id as query parameter."""
    with respx.mock(base_url=_BASE_URL) as mock:
        create_route = mock.post("/v10/projects").respond(200, json=sample_project_data)

        config = ProjectConfig(
            access_token="test-token",
            name="my-app",
            team_id="team_xyz789",
        )

        result = await harness.invoke_create(Project, name="my-app", config=config)

    assert result.success
    assert "teamId=team_xyz789" in str(create_route.calls[0].request.url)


async def test_create_project_api_error(
    harness: ProviderHarness,
) -> None:
    """on_create propagates API errors."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/v10/projects").respond(400, json={"error": {"message": "Name already exists"}})

        config = ProjectConfig(
            access_token="test-token",
            name="existing-project",
        )

        result = await harness.invoke_create(Project, name="existing-project", config=config)

    assert result.failed
    assert "Name already exists" in str(result.error)


async def test_update_project_settings(
    harness: ProviderHarness,
    sample_project_data: dict[str, Any],
    sample_env_list_data: dict[str, Any],
) -> None:
    """on_update patches project when settings change."""
    updated_data = {**sample_project_data, "framework": "vite"}

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.patch("/v9/projects/prj_abc123def456").respond(200, json=updated_data)
        mock.get("/v9/projects/prj_abc123def456/env").respond(200, json={"envs": []})
        mock.get("/v9/projects/prj_abc123def456").respond(200, json=updated_data)

        previous = ProjectConfig(
            access_token="test-token",
            name="my-app",
            framework="nextjs",
        )
        current = ProjectConfig(
            access_token="test-token",
            name="my-app",
            framework="vite",
        )
        existing_outputs = ProjectOutputs(
            project_id="prj_abc123def456",
            name="my-app",
            account_id="team_xyz789",
            framework="nextjs",
            url="https://my-app.vercel.app",
        )

        result = await harness.invoke_update(
            Project,
            name="my-app",
            config=current,
            previous_config=previous,
            current_outputs=existing_outputs,
        )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.framework == "vite"


async def test_update_project_syncs_env_variables(
    harness: ProviderHarness,
    sample_project_data: dict[str, Any],
    sample_env_list_data: dict[str, Any],
) -> None:
    """on_update removes old env vars and creates new ones."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.get("/v9/projects/prj_abc123def456/env").respond(200, json=sample_env_list_data)
        mock.delete("/v9/projects/prj_abc123def456/env/env_abc123").respond(200, json={})
        env_route = mock.post("/v10/projects/prj_abc123def456/env").respond(200, json={"created": []})
        mock.get("/v9/projects/prj_abc123def456").respond(200, json=sample_project_data)

        previous = ProjectConfig(
            access_token="test-token",
            name="my-app",
        )
        current = ProjectConfig(
            access_token="test-token",
            name="my-app",
            environment_variables=[
                EnvironmentVariableConfig(key="NEW_VAR", value="new-value"),
            ],
        )
        existing_outputs = ProjectOutputs(
            project_id="prj_abc123def456",
            name="my-app",
            account_id="team_xyz789",
            framework="nextjs",
            url="https://my-app.vercel.app",
        )

        result = await harness.invoke_update(
            Project,
            name="my-app",
            config=current,
            previous_config=previous,
            current_outputs=existing_outputs,
        )

    assert result.success

    env_request_body = json.loads(env_route.calls[0].request.content)
    assert env_request_body[0]["key"] == "NEW_VAR"


async def test_delete_project_success(
    harness: ProviderHarness,
) -> None:
    """on_delete removes the project."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/v9/projects/prj_abc123def456").respond(200, json={})

        config = ProjectConfig(
            access_token="test-token",
            name="my-app",
        )
        existing_outputs = ProjectOutputs(
            project_id="prj_abc123def456",
            name="my-app",
            account_id="team_xyz789",
            framework="nextjs",
            url="https://my-app.vercel.app",
        )

        result = await harness.invoke_delete(
            Project,
            name="my-app",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_project_already_gone(
    harness: ProviderHarness,
) -> None:
    """on_delete is idempotent when project does not exist."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/v9/projects/prj_abc123def456").respond(404, json={"error": {"message": "Not found"}})

        config = ProjectConfig(
            access_token="test-token",
            name="my-app",
        )
        existing_outputs = ProjectOutputs(
            project_id="prj_abc123def456",
            name="my-app",
            account_id="team_xyz789",
            framework="nextjs",
            url="https://my-app.vercel.app",
        )

        result = await harness.invoke_delete(
            Project,
            name="my-app",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_project_no_outputs(
    harness: ProviderHarness,
) -> None:
    """on_delete succeeds silently when no outputs exist."""
    config = ProjectConfig(
        access_token="test-token",
        name="my-app",
    )

    with respx.mock(base_url=_BASE_URL):
        result = await harness.invoke_delete(
            Project,
            name="my-app",
            config=config,
        )

    assert result.success


async def test_create_project_with_build_settings(
    harness: ProviderHarness,
    sample_project_data: dict[str, Any],
) -> None:
    """on_create includes build settings in request body."""
    with respx.mock(base_url=_BASE_URL) as mock:
        create_route = mock.post("/v10/projects").respond(200, json=sample_project_data)

        config = ProjectConfig(
            access_token="test-token",
            name="my-app",
            build_command="npm run build",
            output_directory="dist",
            install_command="npm ci",
            root_directory="packages/web",
        )

        result = await harness.invoke_create(Project, name="my-app", config=config)

    assert result.success

    request_body = json.loads(create_route.calls[0].request.content)
    assert request_body["buildCommand"] == "npm run build"
    assert request_body["outputDirectory"] == "dist"
    assert request_body["installCommand"] == "npm ci"
    assert request_body["rootDirectory"] == "packages/web"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Project.resource == "project"
