"""Vercel REST API HTTP client."""

from __future__ import annotations

from typing import Any

import httpx
from pragma_sdk import AlreadyExistsError


_BASE_URL = "https://api.vercel.com"


def create_vercel_client(access_token: str) -> httpx.AsyncClient:
    """Create an authenticated httpx client for the Vercel REST API.

    Args:
        access_token: Vercel API token.

    Returns:
        Configured async HTTP client with authorization headers and base URL.
    """
    return httpx.AsyncClient(
        base_url=_BASE_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


def parse_error_message(response: httpx.Response) -> str:
    """Extract Vercel's error message from a failed response.

    Args:
        response: Failed HTTP response from the Vercel API.

    Returns:
        The message from the response body, or the raw body or status code when it carries none.
    """
    try:
        body: dict[str, Any] = response.json()
        error = body.get("error", {})

        if isinstance(error, dict):
            return error.get("message") or error.get("code") or str(body)

        return body.get("message") or str(body)
    except Exception:
        return response.text or f"HTTP {response.status_code}"


async def raise_for_status(response: httpx.Response) -> None:
    """Raise a descriptive error if the response indicates failure.

    Args:
        response: HTTP response to check.

    Raises:
        httpx.HTTPStatusError: If the response status code indicates an error,
            with the API error message included.
    """
    if response.is_success:
        return

    raise httpx.HTTPStatusError(
        message=f"Vercel API error: {parse_error_message(response)}",
        request=response.request,
        response=response,
    )


async def raise_for_create_status(response: httpx.Response, description: str) -> None:
    """Raise a descriptive error if a create request failed, refusing an object that already exists.

    Args:
        response: HTTP response to a Vercel create request.
        description: What was being created, such as ``Vercel project my-app``.

    Raises:
        AlreadyExistsError: If Vercel answers 409 Conflict; the message carries Vercel's error message.
        httpx.HTTPStatusError: If Vercel answers with any other error status.
    """
    if response.status_code == httpx.codes.CONFLICT:
        msg = f"{description} already exists: {parse_error_message(response)}"
        raise AlreadyExistsError(msg)

    await raise_for_status(response)


async def fetch_optional_json(client: httpx.AsyncClient, path: str, params: dict[str, str]) -> dict[str, Any] | None:
    """Fetch the JSON object at ``path``, or ``None`` when Vercel reports it does not exist.

    Args:
        client: Authenticated Vercel API client.
        path: REST path of the object to read.
        params: Query parameters, such as the team ID.

    Returns:
        The response body, or ``None`` when Vercel answers 404.

    Raises:
        httpx.HTTPStatusError: If Vercel answers with any other error status.
    """
    response = await client.get(path, params=params)

    if response.status_code == 404:
        return None

    await raise_for_status(response)

    return response.json()
