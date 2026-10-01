"""Settings the Pragmatiks runtime passes to the pragma provider through the environment."""

from __future__ import annotations

import os


def require_environment_variable(name: str, purpose: str) -> str:
    """Read an environment variable the Pragmatiks runtime sets.

    Args:
        name: Environment variable name.
        purpose: What the variable is for, quoted in the error.

    Returns:
        The variable's value.

    Raises:
        RuntimeError: If the variable is not set.
    """
    try:
        return os.environ[name]
    except KeyError as exc:
        msg = f"{name} is not set ({purpose}). The pragma provider must be executed by the Pragmatiks runtime."
        raise RuntimeError(msg) from exc


def file_bucket() -> str:
    """Return the object-store bucket holding uploaded file content.

    Returns:
        The value of ``PRAGMA_FILE_GCS_BUCKET``.
    """
    return require_environment_variable("PRAGMA_FILE_GCS_BUCKET", "object-store bucket for uploaded file content")


def organization_id() -> str:
    """Return the id of the organization the runtime serves.

    Returns:
        The value of ``PRAGMA_RUNTIME_ORGANIZATION_ID``.
    """
    return require_environment_variable("PRAGMA_RUNTIME_ORGANIZATION_ID", "organization file prefix in the bucket")


def file_public_base_url() -> str:
    """Return the base URL of public file download links.

    Returns:
        The value of ``PRAGMA_FILE_PUBLIC_URL``.
    """
    return require_environment_variable("PRAGMA_FILE_PUBLIC_URL", "base URL for public file download links")
