"""Lookup and deletion of Kubernetes objects by name."""

from __future__ import annotations

from typing import Any, cast

from lightkube import ApiError, AsyncClient
from lightkube.types import CascadeType

from kubernetes_provider.polling import poll_until


DELETION_POLL_INTERVAL_SECONDS = 2
DELETION_TIMEOUT_SECONDS = 300


async def fetch_object[ObjectT](
    client: AsyncClient,
    object_type: type[ObjectT],
    name: str,
    namespace: str | None = None,
) -> ObjectT | None:
    """Fetch a Kubernetes object by name.

    An object whose deletion is in progress is still returned.

    Args:
        client: Lightkube client for the target cluster.
        object_type: Lightkube resource type of the object.
        name: Object name.
        namespace: Namespace of a namespaced object; ``None`` for cluster-scoped objects.

    Returns:
        The object, or ``None`` when the cluster has no object with that name.

    Raises:
        ApiError: If the request fails for any reason other than not found.
    """
    try:
        return await client.get(cast(Any, object_type), name, namespace=namespace)
    except ApiError as error:
        if error.status.code == 404:
            return None
        raise


async def delete_object(
    client: AsyncClient,
    object_type: type,
    name: str,
    namespace: str | None = None,
) -> None:
    """Request deletion of a Kubernetes object, cascading to its dependents in the background.

    Returns once the cluster accepts the request; the object can still be terminating.
    Idempotent: an object that is already absent counts as deleted.

    Args:
        client: Lightkube client for the target cluster.
        object_type: Lightkube resource type of the object.
        name: Object name.
        namespace: Namespace of a namespaced object; ``None`` for cluster-scoped objects.

    Raises:
        ApiError: If the delete request fails for any reason other than not found.
    """
    try:
        await client.delete(cast(Any, object_type), name, namespace=namespace, cascade=CascadeType.BACKGROUND)
    except ApiError as error:
        if error.status.code != 404:
            raise


async def wait_for_absence(
    client: AsyncClient,
    object_type: type,
    name: str,
    namespace: str | None = None,
    timeout_seconds: int = DELETION_TIMEOUT_SECONDS,
) -> None:
    """Poll until a Kubernetes object is gone from the cluster.

    Args:
        client: Lightkube client for the target cluster.
        object_type: Lightkube resource type of the object.
        name: Object name.
        namespace: Namespace of a namespaced object; ``None`` for cluster-scoped objects.
        timeout_seconds: Seconds after which waiting stops.

    Raises:
        ApiError: If reading the object fails for any reason other than not found.
        TimeoutError: If the object is still present after ``timeout_seconds``; the message
            names its deletion timestamp and the finalizers holding it.
    """
    location = f"{namespace}/{name}" if namespace else name

    def describe_timeout(remaining: object) -> str:
        metadata = cast(Any, remaining).metadata
        return (
            f"{object_type.__name__} {location} still present {timeout_seconds}s after deletion "
            f"(deletionTimestamp: {metadata.deletionTimestamp}, finalizers: {metadata.finalizers})"
        )

    await poll_until(
        lambda: fetch_object(client, object_type, name, namespace),
        lambda remaining: remaining is None,
        describe_timeout,
        timeout_seconds,
        DELETION_POLL_INTERVAL_SECONDS,
    )
