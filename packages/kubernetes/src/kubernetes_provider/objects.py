"""Creation, lookup and deletion of Kubernetes objects."""

from __future__ import annotations

import logging
from typing import Any, cast

from lightkube import ApiError, AsyncClient
from lightkube.models.meta_v1 import ObjectMeta
from lightkube.types import CascadeType
from pragma_sdk import AlreadyExistsError

from kubernetes_provider.marker import (
    MARKER_MISMATCH_REASONS,
    RESOURCE_MARKER_KEY,
    MarkerMatch,
    build_resource_marker,
    match_resource_marker,
)
from kubernetes_provider.polling import poll_until


DELETION_POLL_INTERVAL_SECONDS = 2
DELETION_TIMEOUT_SECONDS = 300
FIELD_MANAGER = "pragma-kubernetes"
ALREADY_EXISTS_STATUS_CODE = 409

logger = logging.getLogger(__name__)


def build_object_metadata(
    resource_id: str,
    name: str,
    namespace: str | None = None,
    labels: dict[str, str] | None = None,
) -> ObjectMeta:
    """Build the metadata of an object a Pragmatiks resource creates and applies.

    Every object carries the resource marker, so a create that meets an existing object can
    tell this resource's earlier create (``check_create_conflict``) from any other object.

    Args:
        resource_id: Canonical ID of the Pragmatiks resource that creates the object.
        name: Object name.
        namespace: Namespace of a namespaced object; ``None`` for cluster-scoped objects.
        labels: Labels the object carries, if any.

    Returns:
        Object metadata carrying the ``pragma.io/resource`` annotation.
    """
    return ObjectMeta(
        name=name,
        namespace=namespace,
        labels=labels,
        annotations=build_resource_marker(resource_id),
    )


async def create_object[ObjectT](client: AsyncClient, kubernetes_object: ObjectT) -> ObjectT:
    """Create a Kubernetes object, refusing one that already exists.

    Args:
        client: Lightkube client for the target cluster.
        kubernetes_object: Object to create; a namespaced object names its namespace in its metadata.

    Returns:
        The object as created by the cluster.

    Raises:
        AlreadyExistsError: If the cluster already holds an object of that type and name.
        ApiError: If the create request fails for any other reason.
    """
    try:
        return await client.create(cast(Any, kubernetes_object), field_manager=FIELD_MANAGER)
    except ApiError as error:
        if error.status.code == ALREADY_EXISTS_STATUS_CODE:
            message = f"{type(kubernetes_object).__name__} already exists: {error.status.message}"
            raise AlreadyExistsError(message) from error
        raise


async def check_create_conflict(
    client: AsyncClient,
    kubernetes_object: object,
    resource_id: str,
    error: AlreadyExistsError,
) -> None:
    """Return when the object a create met is this resource's earlier create, and raise otherwise.

    Call it when ``create_object`` raised ``AlreadyExistsError``; once it returns, applying the
    desired object (as ``on_update`` does) converges it. An object that was gone when read back
    also returns, since applying creates it.

    Args:
        client: Lightkube client for the target cluster.
        kubernetes_object: Object the create sent.
        resource_id: Canonical ID of the Pragmatiks resource whose create met the object.
        error: The refusal ``create_object`` raised.

    Raises:
        AlreadyExistsError: If the existing object carries no ``pragma.io/resource`` annotation
            naming ``resource_id``; the message says which.
        RuntimeError: If the existing object is this resource's earlier create and is being deleted.
        ApiError: If reading the existing object fails for any reason other than not found.
    """
    metadata = cast(Any, kubernetes_object).metadata
    kind = type(kubernetes_object).__name__
    location = f"{metadata.namespace}/{metadata.name}" if metadata.namespace else metadata.name
    existing = await fetch_object(client, type(kubernetes_object), metadata.name, metadata.namespace)

    if existing is None:
        logger.warning(
            "kubernetes_create_conflict_absent kind=%s namespace=%s name=%s resource=%s",
            kind,
            metadata.namespace,
            metadata.name,
            resource_id,
        )
        return

    existing_metadata = cast(Any, existing).metadata
    match = match_resource_marker(existing_metadata.annotations, resource_id)

    if match is not MarkerMatch.OWN:
        logger.warning(
            "kubernetes_create_conflict_refused kind=%s namespace=%s name=%s resource=%s reason=%s owner=%s",
            kind,
            metadata.namespace,
            metadata.name,
            resource_id,
            match,
            (existing_metadata.annotations or {}).get(RESOURCE_MARKER_KEY),
        )
        message = (
            f"{kind} {location} already exists and this resource did not create it: {MARKER_MISMATCH_REASONS[match]}"
        )
        raise AlreadyExistsError(message) from error

    if existing_metadata.deletionTimestamp is not None:
        logger.warning(
            "kubernetes_create_conflict_terminating kind=%s namespace=%s name=%s resource=%s deletion_timestamp=%s",
            kind,
            metadata.namespace,
            metadata.name,
            resource_id,
            existing_metadata.deletionTimestamp,
        )
        message = (
            f"{kind} {location}, created earlier by this resource, is being deleted since "
            f"{existing_metadata.deletionTimestamp}; apply again once it is gone"
        )
        raise RuntimeError(message) from error

    logger.warning(
        "kubernetes_create_conflict_recognized kind=%s namespace=%s name=%s resource=%s",
        kind,
        metadata.namespace,
        metadata.name,
        resource_id,
    )


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
