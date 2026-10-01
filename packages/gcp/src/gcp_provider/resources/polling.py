"""Polling bounded by one deadline shared across the waits of a lifecycle handler."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator


OPERATION_BUDGET_SECONDS = 1140
POLL_INTERVAL_SECONDS = 30


def compute_operation_deadline() -> float:
    """Compute the time by which every wait of one lifecycle handler must finish.

    The budget stays under the 1200-second operation deadline the host enforces, so a wait that runs
    out raises its own timeout, naming what it waited for, before the host cancels the handler.

    Returns:
        The deadline on the ``time.monotonic`` clock.
    """
    return time.monotonic() + OPERATION_BUDGET_SECONDS


async def poll_until(deadline: float) -> AsyncIterator[None]:
    """Yield once per poll, sleeping ``POLL_INTERVAL_SECONDS`` between polls, until ``deadline`` passes.

    The last poll happens at the deadline, so a caller that exhausts the iterator has seen the state
    at the deadline and should raise its timeout.

    Args:
        deadline: Deadline on the ``time.monotonic`` clock, from ``compute_operation_deadline``.

    Yields:
        None, once per poll.
    """
    while True:
        yield

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            return

        await asyncio.sleep(min(POLL_INTERVAL_SECONDS, remaining))
