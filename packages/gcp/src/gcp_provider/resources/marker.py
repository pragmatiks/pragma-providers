"""Resource marker: the label that names the Pragmatiks resource whose create made a Google Cloud object."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

from pragma_sdk import AlreadyExistsError


RESOURCE_MARKER_KEY = "pragma_resource"
RESOURCE_MARKER_DIGEST_SIZE_BYTES = 31

logger = logging.getLogger(__name__)


class MarkerMatch(StrEnum):
    """How an existing object's resource marker compares with the resource that met it on create."""

    OWN = "own"
    ABSENT = "absent"
    UNMARKED = "unmarked"
    OTHER_RESOURCE = "other_resource"


class MarkerRefusal(StrEnum):
    """The step that refuses an object at a resource's identity, valued as its log event."""

    CREATE_CONFLICT = "gcp_create_conflict_refused"
    OBSERVE = "gcp_observe_refused"


@dataclass(frozen=True)
class MarkerTarget:
    """An object at a resource's identity, as marker checks name it in logs and refusals.

    Attributes:
        resource_id: Canonical ID of the Pragmatiks resource whose identity names the object.
        type_name: Resource type in log events, such as ``secret``.
        object_path: Google Cloud resource path of the object.
        subject: The object as the refusal message names it, such as ``Secret api-key in project shop``.
    """

    resource_id: str
    type_name: str
    object_path: str
    subject: str


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
        The message a refusal raises ``AlreadyExistsError`` with.
    """
    return f"{subject} already exists and this resource did not create it: {MARKER_MISMATCH_REASONS[match]}"


def refuse_object(target: MarkerTarget, refusal: MarkerRefusal, match: MarkerMatch, owner: str | None) -> NoReturn:
    """Log the refusal of an object at a resource's identity and raise it.

    Args:
        target: The object refused.
        refusal: The step refusing it, which names the log event.
        match: How the object's resource marker compares with the resource; any value but ``OWN``.
        owner: The object's ``pragma_resource`` label value, or ``None`` when it carries none.

    Raises:
        AlreadyExistsError: Always, with ``build_conflict_message``.
    """
    logger.warning(
        "%s type=%s resource=%s object=%s reason=%s owner=%s",
        refusal,
        target.type_name,
        target.resource_id,
        target.object_path,
        match,
        owner,
    )
    raise AlreadyExistsError(build_conflict_message(target.subject, match))


def check_resource_marker(labels: Mapping[str, str] | None, target: MarkerTarget, refusal: MarkerRefusal) -> None:
    """Return when an object's labels name the resource as the one that created it, and refuse it otherwise.

    The create's conflict path and ``on_observe`` both recognize the object through it, so they
    refuse the same objects.

    Args:
        labels: Labels read from the object, or ``None`` when it has none.
        target: The object checked.
        refusal: The step checking it, which names the log event of a refusal.

    Raises:
        AlreadyExistsError: If the object carries no ``pragma_resource`` label or one naming
            another resource.
    """
    match = match_resource_marker(labels, target.resource_id)

    if match is not MarkerMatch.OWN:
        refuse_object(target, refusal, match, (labels or {}).get(RESOURCE_MARKER_KEY))


def log_conflict_recognized(target: MarkerTarget) -> None:
    """Log that a create met this resource's own earlier create and brings it up to date.

    Args:
        target: The object the create met, whose labels name the resource.
    """
    logger.warning(
        "gcp_create_conflict_recognized type=%s resource=%s object=%s",
        target.type_name,
        target.resource_id,
        target.object_path,
    )
