"""GitHub REST API HTTP client."""

from __future__ import annotations

from typing import Any

import httpx


_BASE_URL = "https://api.github.com"


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
        For a redirect, a message saying the repository in the request path moved
        and that ``owner`` and the repository name must point at its new location.
        Otherwise the API error message, falling back to the body or status code.
    """
    if response.is_redirect:
        return (
            f"{response.request.url.path} moved because the repository was renamed or transferred; "
            "set owner and the repository name to its new location"
        )

    try:
        body: dict[str, Any] = response.json()
        return body.get("message") or str(body)
    except Exception:
        return response.text or f"HTTP {response.status_code}"


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
        The response body, or ``None`` when GitHub answers 404 or redirects
        because the repository was renamed or transferred away from ``path``.

    Raises:
        httpx.HTTPStatusError: If GitHub answers with any other error status.
    """
    response = await client.get(path)

    if response.status_code == 404 or response.is_redirect:
        return None

    await raise_for_status(response)

    return response.json()


async def delete_if_present(client: httpx.AsyncClient, path: str) -> None:
    """Delete the object at ``path``, succeeding when nothing exists there.

    A repository renamed or transferred away from ``path`` counts as absent and
    is left untouched at its new location.

    Args:
        client: Authenticated GitHub API client.
        path: REST path of the object to delete.

    Raises:
        httpx.HTTPStatusError: If GitHub answers with an error status other than 404.
    """
    response = await client.delete(path)

    if response.status_code == 404 or response.is_redirect:
        return

    await raise_for_status(response)
