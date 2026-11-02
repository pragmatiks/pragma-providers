"""Resource marker: the annotation that names the Pragmatiks resource whose create made a Kubernetes object."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum


RESOURCE_MARKER_KEY = "pragma.io/resource"


class MarkerMatch(StrEnum):
    """How an existing object's resource marker compares with the resource that met it on create.

    ``ABSENT`` is for a caller whose object was gone when read back; ``match_resource_marker``
    never returns it.
    """

    OWN = "own"
    ABSENT = "absent"
    UNMARKED = "unmarked"
    OTHER_RESOURCE = "other_resource"


MARKER_MISMATCH_REASONS = {
    MarkerMatch.UNMARKED: f"it carries no {RESOURCE_MARKER_KEY} annotation, so Pragmatiks did not create it",
    MarkerMatch.OTHER_RESOURCE: f"its {RESOURCE_MARKER_KEY} annotation names another Pragmatiks resource",
}


def build_resource_marker(resource_id: str) -> dict[str, str]:
    """Build the annotations an object carries so a later conflict on create can recognize it.

    Args:
        resource_id: Canonical ID of the Pragmatiks resource creating the object.

    Returns:
        The ``pragma.io/resource`` annotation set to ``resource_id``.
    """
    return {RESOURCE_MARKER_KEY: resource_id}


def match_resource_marker(annotations: Mapping[str, str] | None, resource_id: str) -> MarkerMatch:
    """Compare an existing object's annotations with the marker a resource's create sends.

    Args:
        annotations: Annotations read from the existing object, or ``None`` when it has none.
        resource_id: Canonical ID of the Pragmatiks resource whose create met the object.

    Returns:
        ``OWN`` when the ``pragma.io/resource`` annotation names ``resource_id``, ``UNMARKED`` when
        the object carries no such annotation, ``OTHER_RESOURCE`` otherwise.
    """
    value = (annotations or {}).get(RESOURCE_MARKER_KEY)

    if value is None:
        return MarkerMatch.UNMARKED

    if value == resource_id:
        return MarkerMatch.OWN

    return MarkerMatch.OTHER_RESOURCE
