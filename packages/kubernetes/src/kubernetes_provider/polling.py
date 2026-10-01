"""Polling a cluster read until its result satisfies a condition."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


async def poll_until[ResultT](
    read: Callable[[], Awaitable[ResultT]],
    is_done: Callable[[ResultT], bool],
    describe_timeout: Callable[[ResultT], str],
    timeout_seconds: int,
    interval_seconds: int,
) -> ResultT:
    """Repeat a read until its result satisfies a condition or the timeout passes.

    Args:
        read: Reads the current state.
        is_done: Tells whether a read result satisfies the condition.
        describe_timeout: Builds the timeout message from the last read result.
        timeout_seconds: Seconds after which polling stops.
        interval_seconds: Seconds between reads.

    Returns:
        The first read result that satisfies the condition.

    Raises:
        TimeoutError: If no read result satisfies the condition within the timeout; the
            message comes from ``describe_timeout``.
    """
    deadline = time.monotonic() + timeout_seconds

    while True:
        result = await read()

        if is_done(result):
            return result

        if time.monotonic() >= deadline:
            raise TimeoutError(describe_timeout(result))

        await asyncio.sleep(interval_seconds)
