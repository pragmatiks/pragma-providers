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


async def raise_for_status(response: httpx.Response) -> None:
    """Raise a descriptive error if the response indicates failure.

    Extracts the error message from the GitHub API response body when
    available, falling back to the raw status code otherwise.

    Args:
        response: HTTP response to check.

    Raises:
        httpx.HTTPStatusError: If the response status code indicates an error,
            with the API error message included.
    """
    if response.is_success:
        return

    try:
        body: dict[str, Any] = response.json()
        message = body.get("message") or str(body)
    except Exception:
        message = response.text or f"HTTP {response.status_code}"

    raise httpx.HTTPStatusError(
        message=f"GitHub API error: {message}",
        request=response.request,
        response=response,
    )
