"""Tests for GitHub Repository resource."""

from __future__ import annotations

import json
from typing import Any

import respx
from pragma_sdk.provider import ProviderHarness

from github_provider import Repository, RepositoryConfig, RepositoryOutputs


_BASE_URL = "https://api.github.com"


async def test_create_repository_success(
    harness: ProviderHarness,
    sample_repository_data: dict[str, Any],
) -> None:
    """on_create creates repository via organization endpoint and returns outputs."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/orgs/my-org/repos").respond(200, json=sample_repository_data)

        config = RepositoryConfig(
            access_token="ghp_test_token",
            owner="my-org",
            name="my-repo",
            description="My application repository",
            visibility="private",
        )

        result = await harness.invoke_create(Repository, name="my-repo", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.repository_id == 123456789
    assert result.outputs.full_name == "my-org/my-repo"
    assert result.outputs.html_url == "https://github.com/my-org/my-repo"
    assert result.outputs.clone_url == "https://github.com/my-org/my-repo.git"
    assert result.outputs.ssh_url == "git@github.com:my-org/my-repo.git"
    assert result.outputs.default_branch == "main"
    assert result.outputs.visibility == "private"


async def test_create_repository_falls_back_to_user_endpoint(
    harness: ProviderHarness,
    sample_repository_data: dict[str, Any],
) -> None:
    """on_create falls back to user endpoint when org endpoint returns 404."""
    user_data = {**sample_repository_data, "full_name": "my-user/my-repo"}

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/orgs/my-user/repos").respond(404, json={"message": "Not Found"})
        mock.post("/user/repos").respond(200, json=user_data)

        config = RepositoryConfig(
            access_token="ghp_test_token",
            owner="my-user",
            name="my-repo",
        )

        result = await harness.invoke_create(Repository, name="my-repo", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.full_name == "my-user/my-repo"


async def test_create_repository_includes_all_settings(
    harness: ProviderHarness,
    sample_repository_data: dict[str, Any],
) -> None:
    """on_create includes all configuration settings in the request body."""
    with respx.mock(base_url=_BASE_URL) as mock:
        create_route = mock.post("/orgs/my-org/repos").respond(200, json=sample_repository_data)

        config = RepositoryConfig(
            access_token="ghp_test_token",
            owner="my-org",
            name="my-repo",
            description="Test repo",
            visibility="public",
            auto_init=True,
            has_issues=True,
            has_wiki=True,
            has_projects=True,
            delete_branch_on_merge=False,
            allow_squash_merge=True,
            allow_merge_commit=False,
            allow_rebase_merge=False,
        )

        result = await harness.invoke_create(Repository, name="my-repo", config=config)

    assert result.success

    request_body = json.loads(create_route.calls[0].request.content)
    assert request_body["name"] == "my-repo"
    assert request_body["description"] == "Test repo"
    assert request_body["private"] is False
    assert request_body["auto_init"] is True
    assert request_body["has_issues"] is True
    assert request_body["has_wiki"] is True
    assert request_body["has_projects"] is True
    assert request_body["delete_branch_on_merge"] is False
    assert request_body["allow_squash_merge"] is True
    assert request_body["allow_merge_commit"] is False
    assert request_body["allow_rebase_merge"] is False


async def test_create_repository_api_error(
    harness: ProviderHarness,
) -> None:
    """on_create propagates API errors."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/orgs/my-org/repos").respond(422, json={"message": "Repository creation failed"})

        config = RepositoryConfig(
            access_token="ghp_test_token",
            owner="my-org",
            name="existing-repo",
        )

        result = await harness.invoke_create(Repository, name="existing-repo", config=config)

    assert result.failed
    assert "Repository creation failed" in str(result.error)


async def test_update_repository_settings(
    harness: ProviderHarness,
    sample_repository_data: dict[str, Any],
) -> None:
    """on_update patches repository settings."""
    updated_data = {**sample_repository_data, "description": "Updated description", "visibility": "public"}

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.patch("/repos/my-org/my-repo").respond(200, json=updated_data)

        previous = RepositoryConfig(
            access_token="ghp_test_token",
            owner="my-org",
            name="my-repo",
            description="Original description",
        )
        current = RepositoryConfig(
            access_token="ghp_test_token",
            owner="my-org",
            name="my-repo",
            description="Updated description",
            visibility="public",
        )
        existing_outputs = RepositoryOutputs(
            repository_id=123456789,
            full_name="my-org/my-repo",
            html_url="https://github.com/my-org/my-repo",
            clone_url="https://github.com/my-org/my-repo.git",
            ssh_url="git@github.com:my-org/my-repo.git",
            default_branch="main",
            visibility="private",
        )

        result = await harness.invoke_update(
            Repository,
            name="my-repo",
            config=current,
            previous_config=previous,
            current_outputs=existing_outputs,
        )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.visibility == "public"


async def test_update_repository_sends_correct_body(
    harness: ProviderHarness,
    sample_repository_data: dict[str, Any],
) -> None:
    """on_update sends only mutable fields in the request body."""
    with respx.mock(base_url=_BASE_URL) as mock:
        patch_route = mock.patch("/repos/my-org/my-repo").respond(200, json=sample_repository_data)

        config = RepositoryConfig(
            access_token="ghp_test_token",
            owner="my-org",
            name="my-repo",
            description="New description",
            has_wiki=True,
        )
        existing_outputs = RepositoryOutputs(
            repository_id=123456789,
            full_name="my-org/my-repo",
            html_url="https://github.com/my-org/my-repo",
            clone_url="https://github.com/my-org/my-repo.git",
            ssh_url="git@github.com:my-org/my-repo.git",
            default_branch="main",
            visibility="private",
        )

        result = await harness.invoke_update(
            Repository,
            name="my-repo",
            config=config,
            previous_config=config,
            current_outputs=existing_outputs,
        )

    assert result.success

    request_body = json.loads(patch_route.calls[0].request.content)
    assert request_body["description"] == "New description"
    assert request_body["has_wiki"] is True
    assert "name" not in request_body
    assert "auto_init" not in request_body


async def test_delete_repository_success(
    harness: ProviderHarness,
) -> None:
    """on_delete removes the repository."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/repos/my-org/my-repo").respond(204)

        config = RepositoryConfig(
            access_token="ghp_test_token",
            owner="my-org",
            name="my-repo",
        )
        existing_outputs = RepositoryOutputs(
            repository_id=123456789,
            full_name="my-org/my-repo",
            html_url="https://github.com/my-org/my-repo",
            clone_url="https://github.com/my-org/my-repo.git",
            ssh_url="git@github.com:my-org/my-repo.git",
            default_branch="main",
            visibility="private",
        )

        result = await harness.invoke_delete(
            Repository,
            name="my-repo",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_repository_already_gone(
    harness: ProviderHarness,
) -> None:
    """on_delete is idempotent when repository does not exist."""
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.delete("/repos/my-org/my-repo").respond(404, json={"message": "Not Found"})

        config = RepositoryConfig(
            access_token="ghp_test_token",
            owner="my-org",
            name="my-repo",
        )
        existing_outputs = RepositoryOutputs(
            repository_id=123456789,
            full_name="my-org/my-repo",
            html_url="https://github.com/my-org/my-repo",
            clone_url="https://github.com/my-org/my-repo.git",
            ssh_url="git@github.com:my-org/my-repo.git",
            default_branch="main",
            visibility="private",
        )

        result = await harness.invoke_delete(
            Repository,
            name="my-repo",
            config=config,
            current_outputs=existing_outputs,
        )

    assert result.success


async def test_delete_repository_no_outputs(
    harness: ProviderHarness,
) -> None:
    """on_delete succeeds silently when no outputs exist."""
    config = RepositoryConfig(
        access_token="ghp_test_token",
        owner="my-org",
        name="my-repo",
    )

    with respx.mock(base_url=_BASE_URL):
        result = await harness.invoke_delete(
            Repository,
            name="my-repo",
            config=config,
        )

    assert result.success


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Repository.resource == "repository"
