"""Pod log reading and following shared by the workload resources."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime

from lightkube import ApiError, AsyncClient
from lightkube.core.client import LabelSelector
from lightkube.resources.core_v1 import Pod
from pragma_sdk import LogEntry


@dataclass(frozen=True)
class PodFollowEnded:
    """Marker a pod's follow puts on the queue last, once its part has ended."""


type PodFollowItem = LogEntry | Exception | PodFollowEnded


def compute_since_seconds(since: datetime | None) -> int | None:
    """Convert an absolute start time into the Kubernetes ``sinceSeconds`` value.

    Args:
        since: Only logs at or after this timestamp; None for no lower bound.

    Returns:
        Whole seconds between ``since`` and now, rounded up and at least 1, or None when ``since``
        is None. Rounding up makes the window cover ``since``; ``is_entry_since`` drops the
        surplus.
    """
    if since is None:
        return None

    return max(1, math.ceil((datetime.now(UTC) - since).total_seconds()))


def build_pod_log_entry(pod_name: str, line: str) -> LogEntry:
    """Build the log entry for one timestamped line a pod wrote.

    Args:
        pod_name: Name of the pod the line came from.
        line: The log line as Kubernetes sends it with ``timestamps=True``: an RFC 3339 timestamp,
            a space, then the message, without its trailing newline.

    Returns:
        An info entry stamped with the time the container wrote the line, carrying the pod name in
        its metadata.
    """
    timestamp, _, message = line.partition(" ")

    return LogEntry(
        timestamp=datetime.fromisoformat(timestamp),
        level="info",
        message=message,
        metadata={"pod": pod_name},
    )


def is_entry_since(entry: LogEntry, since: datetime | None) -> bool:
    """Decide whether an entry falls in the ``since`` window.

    Args:
        entry: The entry.
        since: Lower bound, inclusive; None for no lower bound.

    Returns:
        True when ``since`` is None or the entry's timestamp is at or after it.
    """
    return since is None or entry.timestamp >= since


def build_pod_failure_entry(pod_name: str) -> LogEntry:
    """Build the warning entry for a pod whose logs could not be read.

    Args:
        pod_name: Name of the pod.

    Returns:
        A warning entry stamped now, carrying the pod name in its metadata.
    """
    return LogEntry(
        timestamp=datetime.now(UTC),
        level="warn",
        message=f"Failed to fetch logs from pod {pod_name}",
        metadata={"pod": pod_name},
    )


def is_pod_log_followable(pod: Pod) -> bool:
    """Decide whether a pod has a log to follow.

    Args:
        pod: The pod.

    Returns:
        False for a pod still pending, whose containers have not started; True otherwise.
    """
    return pod.status is None or pod.status.phase != "Pending"


async def list_selected_pods(
    client: AsyncClient,
    namespace: str,
    labels: dict[str, str],
) -> list[Pod]:
    """List the pods in a namespace that carry every given label.

    Args:
        client: Lightkube client for the target cluster.
        namespace: Namespace to list pods in.
        labels: Labels every selected pod carries.

    Returns:
        The selected pods.
    """
    selector: LabelSelector = dict(labels)

    return [pod async for pod in client.list(Pod, namespace=namespace, labels=selector)]


def extract_pod_names(pods: list[Pod]) -> list[str]:
    """Extract the names of pods.

    Args:
        pods: Pods to take names from; a pod without a name is skipped.

    Returns:
        Names of the named pods, in order.
    """
    return [pod.metadata.name for pod in pods if pod.metadata is not None and pod.metadata.name is not None]


async def read_pod_logs(
    client: AsyncClient,
    namespace: str,
    pods: list[Pod],
    since: datetime | None,
    tail: int,
) -> AsyncGenerator[LogEntry]:
    """Read the recent logs of each pod in turn.

    A pod whose logs cannot be read yields one warning entry instead.

    Args:
        client: Lightkube client for the target cluster.
        namespace: Namespace of the pods.
        pods: Pods to read.
        since: Only logs at or after this timestamp.
        tail: Maximum number of lines per pod.

    Yields:
        One entry per log line, pod by pod.
    """
    since_seconds = compute_since_seconds(since)

    for pod_name in extract_pod_names(pods):
        lines = client.log(
            pod_name,
            namespace=namespace,
            since=since_seconds,
            tail_lines=tail,
            timestamps=True,
            newlines=False,
        )

        try:
            entries = [
                entry
                async for line in lines
                if line
                for entry in [build_pod_log_entry(pod_name, line)]
                if is_entry_since(entry, since)
            ]
        except ApiError:
            yield build_pod_failure_entry(pod_name)
            continue

        for entry in entries:
            yield entry


async def follow_pod_log(
    client: AsyncClient,
    namespace: str,
    pod_name: str,
    since: datetime | None,
    tail: int,
    queue: asyncio.Queue[PodFollowItem],
) -> None:
    """Follow one pod's log over an async HTTP stream, putting each line on a queue.

    A pod whose log stream cannot be opened puts one warning entry. Any other failure, at open
    or after, is put as the exception. Unless cancelled, a ``PodFollowEnded`` is put last in every case.

    Args:
        client: Lightkube client for the target cluster.
        namespace: Namespace of the pod.
        pod_name: Pod to follow.
        since: Only logs at or after this timestamp; None for no lower bound.
        tail: Number of backlog lines sent before following.
        queue: Queue the entries, a failure and the end marker go to.

    Note:
        lightkube's follow stream raises RuntimeError instead of ApiError for an error response,
        since it reads the streamed body synchronously, so the status of a refused open is unknown
        and both are reported as a warning entry. The error response arrives before the first line,
        so only the open is guarded.
    """
    lines = aiter(
        client.log(
            pod_name,
            namespace=namespace,
            follow=True,
            since=compute_since_seconds(since),
            tail_lines=tail,
            timestamps=True,
            newlines=False,
        )
    )

    try:
        await put_pod_follow_lines(pod_name, lines, since, queue)
    except Exception as error:
        await queue.put(error)

    await queue.put(PodFollowEnded())


async def put_pod_follow_lines(
    pod_name: str,
    lines: AsyncIterator[str],
    since: datetime | None,
    queue: asyncio.Queue[PodFollowItem],
) -> None:
    """Put each line of one pod's opened follow stream at or after ``since`` on a queue.

    A refused open (ApiError or RuntimeError before the first line) puts one warning entry and returns.

    Args:
        pod_name: Pod the lines come from.
        lines: The pod's timestamped follow stream, not yet started.
        since: Only lines at or after this timestamp; None for no lower bound.
        queue: Queue the entries go to.
    """
    try:
        first_line = await anext(lines)
    except StopAsyncIteration:
        return
    except (ApiError, RuntimeError):
        await queue.put(build_pod_failure_entry(pod_name))
        return

    await put_pod_follow_line(pod_name, first_line, since, queue)

    async for line in lines:
        await put_pod_follow_line(pod_name, line, since, queue)


async def put_pod_follow_line(
    pod_name: str,
    line: str,
    since: datetime | None,
    queue: asyncio.Queue[PodFollowItem],
) -> None:
    """Put one followed line on a queue when it is non-empty and at or after ``since``.

    Args:
        pod_name: Pod the line came from.
        line: The timestamped line.
        since: Only lines at or after this timestamp; None for no lower bound.
        queue: Queue the entry goes to.
    """
    if not line:
        return

    entry = build_pod_log_entry(pod_name, line)

    if is_entry_since(entry, since):
        await queue.put(entry)


async def follow_pod_logs(
    client: AsyncClient,
    namespace: str,
    pods: list[Pod],
    since: datetime | None,
    tail: int,
) -> AsyncGenerator[LogEntry]:
    """Follow the logs of every given pod that has started, merging lines as they arrive.

    A pod still pending, whose containers have not started, is left out. A pod whose log cannot
    be opened yields one warning entry. Each pod sends its last ``tail`` lines first, then follows.
    The iterator ends when every pod's stream has ended. It awaits only async HTTP streams, so
    cancelling or closing it stops every pod stream.

    Args:
        client: Lightkube client for the target cluster.
        namespace: Namespace of the pods.
        pods: Pods to follow.
        since: Only logs at or after this timestamp.
        tail: Number of backlog lines per pod.

    Yields:
        One entry per log line, in arrival order; order holds per pod, not across pods.

    Raises:
        Exception: The first failure of a pod stream other than a refused open.
    """
    pod_names = extract_pod_names([pod for pod in pods if is_pod_log_followable(pod)])
    queue: asyncio.Queue[PodFollowItem] = asyncio.Queue(maxsize=max(1, len(pod_names)))
    tasks = [
        asyncio.create_task(follow_pod_log(client, namespace, pod_name, since, tail, queue)) for pod_name in pod_names
    ]
    remaining = len(tasks)

    try:
        while remaining:
            match await queue.get():
                case PodFollowEnded():
                    remaining -= 1
                case Exception() as error:
                    raise error
                case LogEntry() as entry:
                    yield entry
    finally:
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)


async def read_selected_pod_logs(
    client: AsyncClient,
    namespace: str,
    labels: dict[str, str],
    since: datetime | None,
    tail: int,
) -> AsyncGenerator[LogEntry]:
    """Read the recent logs of the pods a label selector picks.

    Args:
        client: Lightkube client for the target cluster.
        namespace: Namespace of the pods.
        labels: Labels every selected pod carries.
        since: Only logs at or after this timestamp.
        tail: Maximum number of lines per pod.

    Yields:
        One entry per log line, pod by pod; a pod whose logs cannot be read yields one warning entry.
    """
    pods = await list_selected_pods(client, namespace, labels)

    async for entry in read_pod_logs(client, namespace, pods, since, tail):
        yield entry


async def follow_selected_pod_logs(
    client: AsyncClient,
    namespace: str,
    labels: dict[str, str],
    since: datetime | None,
    tail: int,
) -> AsyncGenerator[LogEntry]:
    """Follow the logs of the pods a label selector picks when following starts.

    A pod started later is not picked up. Closing or cancelling the iterator stops every pod stream.

    Args:
        client: Lightkube client for the target cluster.
        namespace: Namespace of the pods.
        labels: Labels every selected pod carries.
        since: Only logs at or after this timestamp.
        tail: Number of backlog lines per pod.

    Yields:
        One entry per log line, in arrival order across pods.

    Raises:
        Exception: The first failure of a pod stream other than a refused open.
    """
    pods = await list_selected_pods(client, namespace, labels)

    async with aclosing(follow_pod_logs(client, namespace, pods, since, tail)) as entries:
        async for entry in entries:
            yield entry
