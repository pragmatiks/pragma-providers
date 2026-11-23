"""Deployments a Vercel project resource triggers from its Git link.

A deployment Pragmatiks triggers carries two ``meta`` keys: ``pragma_resource``, the resource marker naming the
project resource, and ``pragma_spec``, the digest of what it was built from. Deployments without the marker, such
as the builds Vercel starts for a Git push, are never read, awaited or cancelled.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pragma_sdk import RECOVERY_GUIDANCE, Field
from pydantic import BaseModel

from vercel_provider.client import raise_for_status
from vercel_provider.resources.budget import compute_request_timeout, sleep_until_next_poll


logger = logging.getLogger(__name__)

RESOURCE_MARKER_KEY = "pragma_resource"
SPEC_META_KEY = "pragma_spec"
POLL_INTERVAL_SECONDS = 10.0
LISTING_POLL_INTERVAL_SECONDS = 2.0
LISTING_GRACE_SECONDS = 60.0
"""How long a retried create waits for a trigger sent before the host stopped to appear in the listing."""
LOOKUP_PAGE_SIZE = 20
LOOKUP_MAX_PAGES = 5
LOOKUP_WINDOW_DAYS = 90
IN_FLIGHT_STATES = frozenset({"QUEUED", "INITIALIZING", "BUILDING"})
CURRENT_STATES = frozenset({"READY", *IN_FLIGHT_STATES})
FAILED_STATES = frozenset({"ERROR", "CANCELED", "BLOCKED"})
"""Ended states of a build that did not deploy; ``BLOCKED`` is a build a deployment policy or check refused."""
UNSERVED_STATES = frozenset({"DELETED", *FAILED_STATES})
PERMANENT_READ_FAILURES = frozenset({401, 403, 404})


class DeploymentConfig(BaseModel):
    """The Git ref a Vercel project deploys, and the environment it deploys to.

    Attributes:
        ref: Branch, tag or commit SHA to build. A branch builds its head when the deployment's inputs change;
            a tag or SHA pins a release. When None, the production branch of the project's Git link.
        target: ``production`` or ``preview``. A project's first deployment lands as production whatever this
            says, and ``deployment_target`` reports where it landed.
    """

    model_config = {"extra": "forbid"}

    ref: Field[str] | None = None
    target: Field[Literal["production", "preview"]] = "production"


@dataclass(frozen=True)
class StampedDeployments:
    """The two deployments a lookup of a project resource's stamped deployments answers.

    Attributes:
        newest: The newest deployment stamped with the resource marker, in any state.
        newest_served: The newest stamped deployment that did not fail and was not deleted.
    """

    newest: dict[str, Any] | None
    newest_served: dict[str, Any] | None


@dataclass(frozen=True)
class DeploymentRequest:
    """The inputs that keep a project's stamped deployment current.

    Attributes:
        project_id: Vercel project ID.
        resource_id: Canonical ID of the project resource, the value of the resource marker.
        body: ``POST /v13/deployments`` body, from ``build_deployment_body``.
        digest: Digest of the deployment's inputs, from ``build_deployment_digest``.
        is_prefix_retry: True when the handler may be finishing a create a stopped host left part done.
        params: Query parameters, such as the team ID.
        deadline: Handler deadline on the ``time.monotonic`` clock.
    """

    project_id: str
    resource_id: str
    body: dict[str, Any]
    digest: str
    is_prefix_retry: bool
    params: dict[str, str]
    deadline: float


def build_git_source(link: dict[str, Any] | None, repository: str, ref: str) -> dict[str, Any]:
    """Build the ``gitSource`` of a deployment of ``ref`` from the project's own Git link.

    Note:
        Vercel resolves ``ref`` whether it names a branch, a tag or a commit SHA, full or short, and records the
        commit it built as ``gitSource.sha``; ``sha`` is needed only to pin a branch to a commit (probed
        2026-11-23).

    Args:
        link: The project's ``link`` from the Vercel API, or None when it has none.
        repository: The configured repository, ``owner/repo``.
        ref: Branch, tag or commit SHA to build.

    Returns:
        The GitHub ``gitSource`` naming the linked repository by its ID.

    Raises:
        ValueError: If the project has no Git link, its link is not GitHub, or it links another repository.
    """
    if not link:
        message = (
            f"The Vercel project has no Git link to deploy {repository} from. Vercel links a repository only when the "
            "token's account has a GitHub Login Connection and the Vercel GitHub app can reach the repository; "
            "add them, then recreate the project."
        )
        raise ValueError(message)

    if link.get("type") != "github":
        message = f"The Vercel project links a {link.get('type')} repository; Pragmatiks deploys only GitHub links."
        raise ValueError(message)

    linked = f"{link.get('org')}/{link.get('repo')}"

    if linked.lower() != repository.lower():
        message = (
            f"The Vercel project links {linked}, but the configuration names {repository}. Reconnect the repository on "
            "Vercel to refresh the link, or recreate the project."
        )
        raise ValueError(message)

    return {"type": "github", "repoId": link["repoId"], "ref": ref}


def resolve_ref(ref: str | None, link: dict[str, Any] | None) -> str:
    """Resolve the Git ref a deployment builds.

    Args:
        ref: The configured ``deployment.ref``, or None.
        link: The project's Git ``link``, or None.

    Returns:
        The configured ref, else the production branch of the link.

    Raises:
        ValueError: If no ref is configured and the link names no production branch.
    """
    if ref:
        return ref

    branch = (link or {}).get("productionBranch")

    if branch:
        return branch

    message = "The Vercel project's Git link names no production branch; set deployment.ref to the branch, tag or SHA."
    raise ValueError(message)


def build_deployment_digest(
    resource_id: str,
    ref: str,
    target: str,
    build_settings: dict[str, str | None],
    variables: list[tuple[str, str, str]],
) -> str:
    """Build the digest of everything a deployment is built from.

    The digest is an HMAC-SHA256 keyed by the resource ID, so one precomputed table cannot match variable
    values across resources. The key is not secret: a low-entropy value is guessable by a reader of the
    deployment ``meta``.

    Args:
        resource_id: Canonical ID of the project resource, the HMAC key.
        ref: The resolved Git ref.
        target: The configured deployment target.
        build_settings: Framework, build, output and install commands, and root directory.
        variables: ``(key, variable type, value)`` of each declared variable the target sees, from
            ``build_variable_versions``.

    Returns:
        The hex HMAC-SHA256 of the canonical JSON of those inputs.
    """
    payload = {
        "deployment": {"ref": ref, "target": target},
        "settings": build_settings,
        "variables": sorted(list(variable) for variable in variables),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    return hmac.new(resource_id.encode(), canonical.encode(), hashlib.sha256).hexdigest()


def build_deployment_body(
    project_id: str,
    project_name: str,
    git_source: dict[str, Any],
    target: str,
    resource_id: str,
    digest: str,
) -> dict[str, Any]:
    """Build the ``POST /v13/deployments`` body of a stamped deployment.

    Args:
        project_id: Vercel project ID.
        project_name: Vercel project name.
        git_source: The ``gitSource``, from ``build_git_source``.
        target: ``production`` or ``preview``; Vercel refuses ``preview``, so only ``production`` is sent.
        resource_id: Canonical ID of the project resource.
        digest: Digest of the deployment's inputs.

    Returns:
        The request body, its ``meta`` holding exactly the resource marker and the digest.
    """
    body: dict[str, Any] = {
        "name": project_name,
        "project": project_id,
        "gitSource": git_source,
        "meta": {RESOURCE_MARKER_KEY: resource_id, SPEC_META_KEY: digest},
    }

    if target == "production":
        body["target"] = "production"

    return body


def extract_deployment_id(deployment: dict[str, Any]) -> str:
    """Extract a deployment's ID from a list entry (``uid``) or a deployment read (``id``).

    Args:
        deployment: Deployment data from the Vercel API.

    Returns:
        The deployment ID.
    """
    return deployment.get("uid") or deployment["id"]


def is_current_deployment(deployment: dict[str, Any], digest: str) -> bool:
    """Tell whether a deployment was built from ``digest`` and is READY or still building.

    Args:
        deployment: Deployment data from the Vercel API.
        digest: Digest of the deployment's current inputs.

    Returns:
        True when its ``pragma_spec`` equals ``digest`` and it did not fail; a failed one is retried.
    """
    meta = deployment.get("meta") or {}

    return meta.get(SPEC_META_KEY) == digest and deployment.get("readyState") in CURRENT_STATES


def compute_lookup_since() -> int:
    """Compute the start of the lookup window, ``LOOKUP_WINDOW_DAYS`` ago.

    Returns:
        A JavaScript timestamp in milliseconds.
    """
    return int((time.time() - LOOKUP_WINDOW_DAYS * 86400) * 1000)


async def fetch_stamped_deployments(
    client: httpx.AsyncClient,
    project_id: str,
    resource_id: str,
    params: dict[str, str],
    deadline: float,
) -> StampedDeployments:
    """Fetch the newest deployments of the project stamped with the resource marker.

    Reads newest first, at most ``LOOKUP_MAX_PAGES`` pages of the last ``LOOKUP_WINDOW_DAYS``, and stops at the
    first page holding both answers. A stamped deployment outside that window counts as not found.

    Args:
        client: Authenticated Vercel API client.
        project_id: Vercel project ID.
        resource_id: Canonical ID of the project resource.
        params: Query parameters, such as the team ID.
        deadline: Handler deadline on the ``time.monotonic`` clock.

    Returns:
        The newest stamped deployment, and the newest stamped one that did not fail and was not deleted.
    """
    query = {
        **params,
        "projectId": project_id,
        f"meta-{RESOURCE_MARKER_KEY}": resource_id,
        "since": str(compute_lookup_since()),
        "limit": str(LOOKUP_PAGE_SIZE),
    }
    newest: dict[str, Any] | None = None
    newest_served: dict[str, Any] | None = None

    for _ in range(LOOKUP_MAX_PAGES):
        timeout = compute_request_timeout(deadline, f"the deployments of Vercel project {project_id}")
        response = await client.get("/v7/deployments", params=query, timeout=timeout)
        await raise_for_status(response)
        page = response.json()

        for deployment in page.get("deployments", []):
            if (deployment.get("meta") or {}).get(RESOURCE_MARKER_KEY) != resource_id:
                continue

            newest = newest or deployment

            if newest_served is None and deployment.get("readyState") not in UNSERVED_STATES:
                newest_served = deployment

        next_page = (page.get("pagination") or {}).get("next")

        if newest_served is not None or next_page is None:
            break

        query["until"] = str(next_page)

    return StampedDeployments(newest=newest, newest_served=newest_served)


async def trigger_deployment(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    params: dict[str, str],
    deadline: float,
) -> str:
    """Trigger a deployment.

    Vercel refuses the first deployment of a project whose framework was never set unless it is asked to
    auto-detect, so the trigger always asks with ``skipAutoDetectionConfirmation``; a set framework wins.

    Args:
        client: Authenticated Vercel API client.
        body: Request body, from ``build_deployment_body``.
        params: Query parameters, such as the team ID.
        deadline: Handler deadline on the ``time.monotonic`` clock.

    Returns:
        The new deployment's ID.
    """
    timeout = compute_request_timeout(deadline, f"a deployment of Vercel project {body['project']}")
    response = await client.post(
        "/v13/deployments", params={**params, "skipAutoDetectionConfirmation": "1"}, json=body, timeout=timeout
    )
    await raise_for_status(response)

    return response.json()["id"]


async def fetch_request_deployments(client: httpx.AsyncClient, request: DeploymentRequest) -> StampedDeployments:
    """Fetch the stamped deployments of the project a deployment request names.

    Args:
        client: Authenticated Vercel API client.
        request: The deployment request.

    Returns:
        The lookup ``fetch_stamped_deployments`` answers.
    """
    return await fetch_stamped_deployments(
        client, request.project_id, request.resource_id, request.params, request.deadline
    )


@dataclass(frozen=True)
class DeploymentPoll:
    """One read of a deployment while waiting for it.

    Attributes:
        deployment: Deployment data from the Vercel API, empty when the read failed.
        failure: Why the read failed, such as ``HTTP 502`` or ``ConnectTimeout``, or None when it succeeded.
    """

    deployment: dict[str, Any]
    failure: str | None


async def fetch_deployment_poll(
    client: httpx.AsyncClient,
    deployment_id: str,
    params: dict[str, str],
    deadline: float,
) -> DeploymentPoll:
    """Read a deployment once, answering a transient failure instead of raising it.

    Args:
        client: Authenticated Vercel API client.
        deployment_id: Deployment to read.
        params: Query parameters, such as the team ID.
        deadline: Handler deadline on the ``time.monotonic`` clock.

    Returns:
        The deployment, or the transient failure: a transport error or a status other than 401, 403 and 404.

    Raises:
        httpx.HTTPStatusError: If Vercel answers 401, 403 or 404.
    """
    timeout = compute_request_timeout(deadline, f"Vercel deployment {deployment_id}")

    try:
        response = await client.get(f"/v13/deployments/{deployment_id}", params=params, timeout=timeout)
    except httpx.TransportError as error:
        return DeploymentPoll(deployment={}, failure=type(error).__name__)

    if response.status_code in PERMANENT_READ_FAILURES:
        await raise_for_status(response)

    if not response.is_success:
        return DeploymentPoll(deployment={}, failure=f"HTTP {response.status_code}")

    return DeploymentPoll(deployment=response.json(), failure=None)


async def wait_for_ready(
    client: httpx.AsyncClient,
    deployment_id: str,
    params: dict[str, str],
    deadline: float,
) -> dict[str, Any]:
    """Poll a deployment until it is READY, polling on through a transient read failure.

    Args:
        client: Authenticated Vercel API client.
        deployment_id: Deployment to poll.
        params: Query parameters, such as the team ID.
        deadline: Handler deadline on the ``time.monotonic`` clock.

    Returns:
        The READY deployment's data.

    Raises:
        RuntimeError: If the build failed, was cancelled or was blocked; the message carries Vercel's error code
            and message.
        TimeoutError: If it is not READY by the deadline; the message names the last read failure, if any.
        httpx.HTTPStatusError: If Vercel answers 401, 403 or 404.
    """
    subject = f"Vercel deployment {deployment_id}"
    ready_state: str | None = None
    last_failure: str | None = None

    while True:
        poll = await fetch_deployment_poll(client, deployment_id, params, deadline)

        if poll.failure is not None:
            last_failure = poll.failure
            logger.warning("vercel_deployment_poll_failed deployment_id=%s failure=%s", deployment_id, poll.failure)

        ready_state = poll.deployment.get("readyState", ready_state)

        if ready_state == "READY":
            return poll.deployment

        if ready_state in FAILED_STATES:
            message = (
                f"{subject} ended {ready_state} ({poll.deployment.get('errorCode')}: "
                f"{poll.deployment.get('errorMessage')}). Fix the build, then apply again to retry it."
            )
            raise RuntimeError(message)

        if not await sleep_until_next_poll(POLL_INTERVAL_SECONDS, deadline):
            break

    failure = f" (last read failed: {last_failure})" if last_failure else ""
    message = f"{subject} was still {ready_state} when the operation ran out of time{failure}. {RECOVERY_GUIDANCE}"
    raise TimeoutError(message)


async def wait_for_stamped_listing(
    client: httpx.AsyncClient,
    request: DeploymentRequest,
    deployment_id: str,
) -> None:
    """Poll the stamped deployments until the newest is ``deployment_id``, so a later read finds it.

    Args:
        client: Authenticated Vercel API client.
        request: The deployment request.
        deployment_id: The deployment the listing must show as newest.

    Raises:
        TimeoutError: If the listing does not show it by the deadline.
    """
    while True:
        lookup = await fetch_request_deployments(client, request)

        if lookup.newest is not None and extract_deployment_id(lookup.newest) == deployment_id:
            return

        if not await sleep_until_next_poll(LISTING_POLL_INTERVAL_SECONDS, request.deadline):
            break

    message = (
        f"Vercel did not list deployment {deployment_id} before the operation ran out of time. {RECOVERY_GUIDANCE}"
    )
    raise TimeoutError(message)


async def wait_for_first_stamped(client: httpx.AsyncClient, request: DeploymentRequest) -> StampedDeployments:
    """Poll the stamped deployments for up to ``LISTING_GRACE_SECONDS`` until one is listed.

    Args:
        client: Authenticated Vercel API client.
        request: The deployment request.

    Returns:
        The last lookup: the first that lists a stamped deployment, or the one at the end of the grace period.
    """
    grace_deadline = min(request.deadline, time.monotonic() + LISTING_GRACE_SECONDS)

    while True:
        lookup = await fetch_request_deployments(client, request)

        if lookup.newest is not None:
            return lookup

        if not await sleep_until_next_poll(LISTING_POLL_INTERVAL_SECONDS, grace_deadline):
            return lookup


def classify_trigger_reason(newest: dict[str, Any] | None, digest: str, waited: bool) -> str:
    """Classify why a stamped deployment that is not current needs a new trigger.

    Args:
        newest: The newest stamped deployment, or None when none is listed.
        digest: Digest of the deployment's current inputs.
        waited: Whether the lookup waited out ``LISTING_GRACE_SECONDS`` for a first listing.

    Returns:
        ``grace_expired``, ``none_listed``, ``failed_previous`` or ``digest_changed``.
    """
    if newest is None:
        return "grace_expired" if waited else "none_listed"

    if (newest.get("meta") or {}).get(SPEC_META_KEY) == digest:
        return "failed_previous"

    return "digest_changed"


async def converge_deployment(client: httpx.AsyncClient, request: DeploymentRequest) -> None:
    """Make the project's newest stamped deployment one built from ``request.digest`` and READY.

    A current stamped deployment is adopted, awaited while it builds, and never duplicated; otherwise a new one
    is triggered, awaited until READY and until listed. On a possible create retry with nothing listed, it first
    waits ``LISTING_GRACE_SECONDS`` for a trigger the stopped host sent. Logs ``vercel_deployment_adopted``,
    ``vercel_deployment_grace_expired`` and ``vercel_deployment_triggered`` with its reason.

    Args:
        client: Authenticated Vercel API client.
        request: The deployment request.

    Raises:
        RuntimeError: If the build failed, was cancelled or was blocked.
        TimeoutError: If the deployment is not READY, or not listed, by the deadline.
        httpx.HTTPStatusError: If Vercel refuses a lookup or the trigger, or answers 401, 403 or 404 to a poll.
    """
    lookup = await fetch_request_deployments(client, request)
    waited = lookup.newest is None and request.is_prefix_retry

    if waited:
        lookup = await wait_for_first_stamped(client, request)

    newest = lookup.newest

    if newest is not None and is_current_deployment(newest, request.digest):
        deployment_id = extract_deployment_id(newest)
        logger.info(
            "vercel_deployment_adopted project_id=%s resource_id=%s deployment_id=%s digest=%s",
            request.project_id,
            request.resource_id,
            deployment_id,
            request.digest,
        )

        if newest.get("readyState") in IN_FLIGHT_STATES:
            await wait_for_ready(client, deployment_id, request.params, request.deadline)

        return

    reason = classify_trigger_reason(newest, request.digest, waited)

    if reason == "grace_expired":
        logger.warning(
            "vercel_deployment_grace_expired project_id=%s resource_id=%s digest=%s",
            request.project_id,
            request.resource_id,
            request.digest,
        )

    deployment_id = await trigger_deployment(client, request.body, request.params, request.deadline)
    logger.info(
        "vercel_deployment_triggered project_id=%s resource_id=%s deployment_id=%s digest=%s reason=%s",
        request.project_id,
        request.resource_id,
        deployment_id,
        request.digest,
        reason,
    )
    await wait_for_ready(client, deployment_id, request.params, request.deadline)
    await wait_for_stamped_listing(client, request, deployment_id)
