"""Resource marker: the label that names the Pragmatiks resource whose create made a Google Cloud object."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import StrEnum

from pragma_sdk import AlreadyExistsError


RESOURCE_MARKER_KEY = "pragma_resource"
RESOURCE_MARKER_DIGEST_SIZE_BYTES = 31


class MarkerMatch(StrEnum):
    """How an existing object's resource marker compares with the resource that met it on create."""

    OWN = "own"
    ABSENT = "absent"
    UNMARKED = "unmarked"
    OTHER_RESOURCE = "other_resource"


MARKER_MISMATCH_REASONS = {
    MarkerMatch.ABSENT: "it was gone when read back; apply again",
    MarkerMatch.UNMARKED: f"it carries no {RESOURCE_MARKER_KEY} label, so Pragmatiks did not create it",
    MarkerMatch.OTHER_RESOURCE: f"its {RESOURCE_MARKER_KEY} label names another Pragmatiks resource",
}


def compute_resource_marker_value(resource_id: str) -> str:
    """Compute the label value that names a Pragmatiks resource.

    Note:
        Google Cloud label values hold at most 63 lowercase letters, digits, dashes and
        underscores, so the canonical resource ID is carried as a 62-character hex digest.

    Args:
        resource_id: Canonical ID of the Pragmatiks resource.

    Returns:
        The BLAKE2b digest of ``resource_id`` in lowercase hex.
    """
    return hashlib.blake2b(resource_id.encode(), digest_size=RESOURCE_MARKER_DIGEST_SIZE_BYTES).hexdigest()


def build_resource_marker(resource_id: str) -> dict[str, str]:
    """Build the labels a create request carries so a later conflict can recognize the object.

    Args:
        resource_id: Canonical ID of the Pragmatiks resource creating the object.

    Returns:
        The ``pragma_resource`` label set to the resource's marker value.
    """
    return {RESOURCE_MARKER_KEY: compute_resource_marker_value(resource_id)}


def match_resource_marker(labels: Mapping[str, str] | None, resource_id: str) -> MarkerMatch:
    """Compare an existing object's labels with the marker a resource's create sends.

    Args:
        labels: Labels read from the existing object, or ``None`` when it has none.
        resource_id: Canonical ID of the Pragmatiks resource whose create met the object.

    Returns:
        ``OWN`` when the ``pragma_resource`` label names ``resource_id``, ``UNMARKED`` when the
        object carries no such label, ``OTHER_RESOURCE`` otherwise.
    """
    value = (labels or {}).get(RESOURCE_MARKER_KEY)

    if value is None:
        return MarkerMatch.UNMARKED

    if value == compute_resource_marker_value(resource_id):
        return MarkerMatch.OWN

    return MarkerMatch.OTHER_RESOURCE


def build_conflict_message(subject: str, match: MarkerMatch) -> str:
    """Build the reason an object at a resource's identity is refused as not this resource's.

    Args:
        subject: The object as the refusal names it, such as ``Secret api-key in project shop``.
        match: How the object's resource marker compares with the resource; any value but ``OWN``.

    Returns:
        The message both the create's conflict and ``on_observe`` raise ``AlreadyExistsError`` with.
    """
    return f"{subject} already exists and this resource did not create it: {MARKER_MISMATCH_REASONS[match]}"


def verify_resource_marker(labels: Mapping[str, str] | None, resource_id: str, subject: str) -> None:
    """Refuse an observed object whose labels do not name the resource as the one that created it.

    Args:
        labels: Labels read from the observed object, or ``None`` when it has none.
        resource_id: Canonical ID of the Pragmatiks resource observing the object.
        subject: The object as the refusal names it.

    Raises:
        AlreadyExistsError: If the object carries no ``pragma_resource`` label or one naming
            another resource.
    """
    match = match_resource_marker(labels, resource_id)

    if match is not MarkerMatch.OWN:
        raise AlreadyExistsError(build_conflict_message(subject, match))
