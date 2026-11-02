"""Resource marker: the collection metadata key that names the Pragmatiks resource whose create made a collection."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


RESOURCE_MARKER_KEY = "pragma_resource"


class MarkerMatch(StrEnum):
    """How an existing collection's resource marker compares with the resource that met it on create.

    ``ABSENT`` is for a collection gone when read back; ``match_resource_marker`` never returns it.
    """

    OWN = "own"
    ABSENT = "absent"
    UNMARKED = "unmarked"
    OTHER_RESOURCE = "other_resource"


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
