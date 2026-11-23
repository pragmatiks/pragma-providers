"""The time budget a Vercel project handler shares across its requests and waits."""

from __future__ import annotations

import asyncio
import time

from pragma_sdk import RECOVERY_GUIDANCE


HANDLER_BUDGET_SECONDS = 1080
"""Time one handler may spend, leaving the rest of the 1200-second operation deadline to the host."""
REQUEST_TIMEOUT_SECONDS = 30.0


def compute_handler_deadline() -> float:
    """Compute the time by which every request and wait of one handler must finish.

    Returns:
        The deadline on the ``time.monotonic`` clock.
    """
    return time.monotonic() + HANDLER_BUDGET_SECONDS


def compute_request_timeout(deadline: float, subject: str) -> float:
    """Compute the HTTP timeout of one request so that it ends by the handler deadline.

    Args:
        deadline: Handler deadline on the ``time.monotonic`` clock.
        subject: What the request works on, such as ``Vercel deployment dpl_1``, named in the timeout.

    Returns:
        Seconds until the deadline, at most ``REQUEST_TIMEOUT_SECONDS``.

    Raises:
        TimeoutError: If the deadline has passed.
    """
    remaining = deadline - time.monotonic()

    if remaining <= 0:
        message = f"The operation ran out of time while working on {subject}. {RECOVERY_GUIDANCE}"
        raise TimeoutError(message)

    return min(REQUEST_TIMEOUT_SECONDS, remaining)


async def sleep_until_next_poll(interval: float, deadline: float) -> bool:
    """Sleep until the next poll, never past the deadline.

    Args:
        interval: Seconds between polls.
        deadline: Deadline on the ``time.monotonic`` clock.

    Returns:
        False when the deadline has passed and no poll is left.
    """
    remaining = deadline - time.monotonic()

    if remaining <= 0:
        return False

    await asyncio.sleep(min(interval, remaining))

    return True
