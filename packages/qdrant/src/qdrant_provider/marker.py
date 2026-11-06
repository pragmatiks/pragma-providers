"""Resource marker: the collection metadata key that names the Pragmatiks resource whose create made a collection."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, NoReturn

from pragma_sdk import AlreadyExistsError


RESOURCE_MARKER_KEY = "pragma_resource"

logger = logging.getLogger(__name__)


class MarkerMatch(StrEnum):
    """How an existing collection's resource marker compares with the resource that met it on create.

    ``ABSENT`` is for a collection gone when read back; ``match_resource_marker`` never returns it.
    """

    OWN = "own"
    ABSENT = "absent"
    UNMARKED = "unmarked"
    OTHER_RESOURCE = "other_resource"


class MarkerRefusal(StrEnum):
    """The step that refuses a collection at a resource's name, valued as its log event."""

    CREATE_CONFLICT = "qdrant_create_conflict_refused"
    OBSERVE = "qdrant_observe_refused"


MARKER_MISMATCH_REASONS = {
    MarkerMatch.ABSENT: "it was gone when read back; apply again",
    MarkerMatch.UNMARKED: f"its metadata carries no {RESOURCE_MARKER_KEY} key, so Pragmatiks did not create it",
    MarkerMatch.OTHER_RESOURCE: f"its {RESOURCE_MARKER_KEY} metadata names another Pragmatiks resource",
}


def build_resource_marker(resource_id: str) -> dict[str, str]:
    """Build the collection metadata a create request carries so a later conflict can recognize the collection.

    Args:
        resource_id: Canonical ID of the Pragmatiks resource creating the collection.

    Returns:
        The ``pragma_resource`` metadata key set to ``resource_id``.
    """
    return {RESOURCE_MARKER_KEY: resource_id}


def match_resource_marker(metadata: Mapping[str, Any] | None, resource_id: str) -> MarkerMatch:
    """Compare an existing collection's metadata with the marker a resource's create sends.

    Args:
        metadata: Metadata read from the existing collection, or ``None`` when it has none.
        resource_id: Canonical ID of the Pragmatiks resource whose create met the collection.

    Returns:
        ``OWN`` when the ``pragma_resource`` key names ``resource_id``, ``UNMARKED`` when the
        metadata carries no such key, ``OTHER_RESOURCE`` otherwise.
    """
    value = (metadata or {}).get(RESOURCE_MARKER_KEY)

    if value is None:
        return MarkerMatch.UNMARKED

    if value == resource_id:
        return MarkerMatch.OWN

    return MarkerMatch.OTHER_RESOURCE


def build_conflict_message(subject: str, match: MarkerMatch) -> str:
    """Build the reason a collection at a resource's name is refused as not this resource's.

    Args:
        subject: The collection as the refusal names it, such as ``Collection 'documents'``.
        match: How the collection's resource marker compares with the resource; any value but ``OWN``.

    Returns:
        The message a refusal raises ``AlreadyExistsError`` with.
    """
    return f"{subject} already exists and this resource did not create it: {MARKER_MISMATCH_REASONS[match]}"


def refuse_collection(
    resource_id: str, collection_name: str, refusal: MarkerRefusal, match: MarkerMatch, owner: Any
) -> NoReturn:
    """Log the refusal of a collection at a resource's name and raise it.

    Args:
        resource_id: Canonical ID of the Pragmatiks resource whose name the collection holds.
        collection_name: Name of the collection refused.
        refusal: The step refusing it, which names the log event.
        match: How the collection's resource marker compares with the resource; any value but ``OWN``.
        owner: The collection's ``pragma_resource`` metadata value, or ``None`` when it carries none.

    Raises:
        AlreadyExistsError: Always, with ``build_conflict_message``.
    """
    logger.warning(
        "%s resource=%s collection=%s reason=%s owner=%s", refusal, resource_id, collection_name, match, owner
    )
    raise AlreadyExistsError(build_conflict_message(f"Collection {collection_name!r}", match))


def check_resource_marker(
    metadata: Mapping[str, Any] | None, resource_id: str, collection_name: str, refusal: MarkerRefusal
) -> None:
    """Return when a collection's metadata names the resource as the one that created it, and refuse it otherwise.

    The create's conflict path and ``on_observe`` both recognize the collection through it, so
    they refuse the same collections.

    Args:
        metadata: Metadata read from the collection, or ``None`` when it has none.
        resource_id: Canonical ID of the Pragmatiks resource reading the collection.
        collection_name: Name of the collection.
        refusal: The step reading it, which names the log event of a refusal.

    Raises:
        AlreadyExistsError: If the metadata carries no ``pragma_resource`` key or one naming
            another resource.
    """
    match = match_resource_marker(metadata, resource_id)

    if match is not MarkerMatch.OWN:
        refuse_collection(resource_id, collection_name, refusal, match, (metadata or {}).get(RESOURCE_MARKER_KEY))
