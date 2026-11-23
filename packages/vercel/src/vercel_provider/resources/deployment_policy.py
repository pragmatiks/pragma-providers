"""The Vercel project ``deploymentPolicy`` that the ``git_push_deploys`` setting maps to."""

from __future__ import annotations

from typing import Any


DEPLOYMENT_SOURCES_BUT_GIT = ("cli", "rest-api", "deploy-hook", "integration", "v0")
"""Every deployment source but ``git``, as https://vercel.com/openapi.json enumerates them. Vercel classes a
``POST /v13/deployments`` carrying ``gitSource`` as ``rest-api``, so the rule leaves Pragmatiks' own deployments
allowed (probed 2026-11-23)."""


def build_push_blocking_policy() -> dict[str, Any]:
    """Build the project ``deploymentPolicy`` that blocks deployments from Git pushes.

    Returns:
        One project rule, for production and preview, allowing every deployment source but ``git``. It overrides
        the team's deployment-source rules for the project.
    """
    return {
        "deploymentSources": [
            {
                "enabled": True,
                "environments": [
                    {"type": "system", "target": "production"},
                    {"type": "system", "target": "preview"},
                ],
                "sources": list(DEPLOYMENT_SOURCES_BUT_GIT),
            }
        ]
    }


def build_team_policy() -> dict[str, Any]:
    """Build the project ``deploymentPolicy`` that drops the project's own deployment-sources rules.

    A null ``deploymentSources`` removes the project's deployment-sources rules and leaves its other rule types,
    such as ``gitSources``, alone; once none is left the project reads ``deploymentPolicy: null``, as a new
    project does. Vercel refuses a string ``deploymentPolicy``, ignores ``{}``, and keeps an empty list as a rule
    set of its own.

    Returns:
        A policy whose ``deploymentSources`` is null.
    """
    return {"deploymentSources": None}


def build_deployment_policy(git_push_deploys: bool) -> dict[str, Any]:
    """Build the project ``deploymentPolicy`` that ``git_push_deploys`` maps to.

    This is the one place the setting meets Vercel's policy shape.

    Args:
        git_push_deploys: Whether Vercel builds the pushes to the linked repository.

    Returns:
        ``build_team_policy`` when pushes deploy, so the project follows its team's policy; otherwise
        ``build_push_blocking_policy``.
    """
    if git_push_deploys:
        return build_team_policy()

    return build_push_blocking_policy()


def build_rule_signature(rule: dict[str, Any]) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Build an order-free signature of one deployment-sources rule.

    A rule read from Vercel without ``environments`` or ``sources`` signs as empty, so it never matches the
    rule ``build_push_blocking_policy`` writes and is overwritten.

    Args:
        rule: A ``deploymentSources`` rule, as sent to or read from Vercel.

    Returns:
        Whether the rule is enabled, its sorted environments and its sorted sources.
    """
    environments = tuple(
        sorted(
            f"{entry.get('type')}:{entry.get('target') or entry.get('environmentId')}"
            for entry in rule.get("environments", [])
        )
    )

    return bool(rule.get("enabled")), environments, tuple(sorted(rule.get("sources", [])))


def is_deployment_policy_current(project: dict[str, Any], git_push_deploys: bool) -> bool:
    """Tell whether a project's deployment policy already is the one ``git_push_deploys`` asks for.

    Args:
        project: Project data from the Vercel API.
        git_push_deploys: Whether Vercel builds the pushes to the linked repository.

    Returns:
        For ``True``, whether the project has no deployment-sources rules of its own; for ``False``, whether its
        rules are exactly the one ``build_push_blocking_policy`` writes.
    """
    sources = (project.get("deploymentPolicy") or {}).get("deploymentSources")
    rules = sources if isinstance(sources, list) else []

    if git_push_deploys:
        return not rules

    wanted = build_push_blocking_policy()["deploymentSources"]

    return sorted(map(build_rule_signature, rules)) == sorted(map(build_rule_signature, wanted))
