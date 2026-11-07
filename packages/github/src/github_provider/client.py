"""GitHub REST API HTTP client."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pragma_sdk import AlreadyExistsError


_BASE_URL = "https://api.github.com"

logger = logging.getLogger(__name__)


def create_github_client(access_token: str) -> httpx.AsyncClient:
    """Create an authenticated httpx client for the GitHub REST API.

    Args:
        access_token: GitHub personal access token (classic or fine-grained).

    Returns:
        Configured async HTTP client with authorization headers and base URL.
        It does not follow redirects, so a request to a renamed or transferred
        repository never reaches the repository at its new location.
    """
    return httpx.AsyncClient(
        base_url=_BASE_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


def parse_error_message(response: httpx.Response) -> str:
    """Parse the failure message GitHub returned with an unsuccessful response.

    Args:
        response: Unsuccessful HTTP response.

    Returns:
        For a redirect, the message of ``build_moved_message``. Otherwise the
        API error message, falling back to the body or status code.
    """
    if response.is_redirect:
        return build_moved_message(response.request.url.path)

    try:
        body: dict[str, Any] = response.json()
        return body.get("message") or str(body)
    except Exception:
        return response.text or f"HTTP {response.status_code}"


def build_moved_message(path: str) -> str:
    """Build the failure message for a request GitHub redirected.

    Args:
        path: REST path GitHub redirected.

    Returns:
        A message saying the repository in ``path`` was renamed or transferred
        and that Pragmatiks does not follow it to its new location.
    """
    return (
        f"{path} redirects because its repository was renamed or transferred; "
        "Pragmatiks does not follow it to its new location"
    )


async def raise_for_status(response: httpx.Response) -> None:
    """Raise a descriptive error if the response indicates failure.

    Args:
        response: HTTP response to check.

    Raises:
        httpx.HTTPStatusError: If the response is an error status or a redirect,
            with the message from ``parse_error_message``.
    """
    if response.is_success:
        return

    raise httpx.HTTPStatusError(
        message=f"GitHub API error: {parse_error_message(response)}",
        request=response.request,
        response=response,
    )


async def fetch_optional_json(client: httpx.AsyncClient, path: str) -> dict[str, Any] | None:
    """Fetch the JSON object at ``path``, or ``None`` when nothing exists there.

    Args:
        client: Authenticated GitHub API client.
        path: REST path of the object to read.

    Returns:
        The response body, or ``None`` when GitHub answers 404.

    Raises:
        AlreadyExistsError: If GitHub redirects because the repository was
            renamed or transferred away from ``path``, so a create there never
            makes a new repository at the old name.
        httpx.HTTPStatusError: If GitHub answers with any other error status.
    """
    response = await client.get(path)

    if response.status_code == 404:
        return None

    if response.is_redirect:
        logger.warning("github_repository_moved path=%s operation=read", path)
        raise AlreadyExistsError(build_moved_message(path))

    await raise_for_status(response)

    return response.json()


async def ensure_absent(client: httpx.AsyncClient, path: str, description: str) -> None:
    """Refuse to create an object GitHub already holds at ``path``.

    GitHub creates environments and secrets with a create-or-replace PUT, so a
    create calls this first to never take over an object Pragmatiks did not create.

    Args:
        client: Authenticated GitHub API client.
        path: REST path of the object about to be created.
        description: What the object is, named in the error message.

    Raises:
        AlreadyExistsError: If GitHub already holds an object at ``path``, or
            redirects because its repository moved.
        httpx.HTTPStatusError: If GitHub answers the read with an error status other than 404.
    """
    if await fetch_optional_json(client, path) is not None:
        msg = f"{description} already exists"
        raise AlreadyExistsError(msg)


async def delete_if_present(client: httpx.AsyncClient, path: str) -> None:
    """Delete the object at ``path``, succeeding when nothing exists there.

    A repository renamed or transferred away from ``path`` counts as absent: it
    is left untouched at its new location and the old path is logged.

    Args:
        client: Authenticated GitHub API client.
        path: REST path of the object to delete.

    Raises:
        httpx.HTTPStatusError: If GitHub answers with an error status other than 404.
    """
    response = await client.delete(path)

    if response.status_code == 404:
        return

    if response.is_redirect:
        logger.warning("github_repository_moved path=%s operation=delete", path)
        return

    await raise_for_status(response)
