"""Base utilities for Cloud SQL resources."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from google.oauth2 import service_account
from googleapiclient import discovery
from googleapiclient.errors import HttpError

from gcp_provider.resources.polling import poll_until


if TYPE_CHECKING:
    from gcp_provider.resources.cloudsql.database_instance import DatabaseInstanceConfig


logger = logging.getLogger(__name__)

DB_CONNECTION_INFO = {
    "POSTGRES": ("postgresql", "5432"),
    "MYSQL": ("mysql", "3306"),
    "SQLSERVER": ("sqlserver", "1433"),
}


def get_credentials(credentials_data: dict[str, Any] | str) -> service_account.Credentials:
    """Create GCP credentials from config data.

    Returns:
        GCP service account credentials.
    """
    if isinstance(credentials_data, str):
        credentials_data = json.loads(credentials_data)

    return service_account.Credentials.from_service_account_info(credentials_data)


async def get_sqladmin_service(credentials: service_account.Credentials) -> Any:
    """Build the Cloud SQL Admin API service off the event loop.

    Returns:
        Cloud SQL Admin API service resource.
    """
    return await run_in_executor(partial(discovery.build, "sqladmin", "v1", credentials=credentials))


async def run_in_executor(func: Any) -> Any:
    """Run a blocking function in the default executor.

    Returns:
        Result of the function execution.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func)


def extract_ips(instance: dict) -> tuple[str | None, str | None]:
    """Extract public and private IPs from instance dict.

    Returns:
        Tuple of (public_ip, private_ip), either may be None.
    """
    public_ip = None
    private_ip = None

    for ip_addr in instance.get("ipAddresses", []):
        ip_type = ip_addr.get("type")

        if ip_type == "PRIMARY":
            public_ip = ip_addr.get("ipAddress")
        elif ip_type == "PRIVATE":
            private_ip = ip_addr.get("ipAddress")

    return public_ip, private_ip


def connection_info(database_version: str) -> tuple[str, str]:
    """Get connection type and port from database version string.

    Returns:
        Tuple of (db_type, port) for the database family.
    """
    db_family = database_version.split("_")[0]
    return DB_CONNECTION_INFO.get(db_family, ("postgresql", "5432"))


async def execute(request: Any, ignore_404: bool = False, ignore_exists: bool = False) -> Any:
    """Execute a GCP API request, optionally ignoring 404 or 409 (conflict/exists) errors.

    Returns:
        API response, or None if error was ignored.

    Raises:
        HttpError: If request fails and error is not ignored.
    """
    try:
        return await run_in_executor(request.execute)
    except HttpError as e:
        if ignore_404 and e.resp.status == 404:
            return None
        if ignore_404 and e.resp.status == 400 and "does not exist" in str(e):
            return None
        if ignore_exists and e.resp.status in (409, 400) and "already exists" in str(e):
            return None
        raise


def format_operation(operation: dict) -> str:
    """Format a Cloud SQL Admin operation for messages.

    Args:
        operation: Operation as the Cloud SQL Admin API returns it.

    Returns:
        The operation type, name and target, such as ``CREATE_DATABASE operation abc on my-project/my-instance``.
    """
    operation_type = operation.get("operationType", "UNKNOWN")
    target = f"{operation.get('targetProject')}/{operation.get('targetId')}"
    return f"{operation_type} operation {operation['name']} on {target}"


def format_operation_fields(operation: dict) -> str:
    """Format a Cloud SQL Admin operation as ``key=value`` log fields.

    Args:
        operation: Operation as the Cloud SQL Admin API returns it.

    Returns:
        The operation type, name and target, such as
        ``operation_type=CREATE_DATABASE operation=abc target=my-project/my-instance``.
    """
    operation_type = operation.get("operationType", "UNKNOWN")
    target = f"{operation.get('targetProject')}/{operation.get('targetId')}"
    return f"operation_type={operation_type} operation={operation['name']} target={target}"


def is_operation_in_progress(error: HttpError) -> bool:
    """Tell whether Cloud SQL rejected a request because another operation runs on the instance.

    Args:
        error: Error the Cloud SQL Admin API raised.

    Returns:
        True for HTTP 409 whose ``error.errors`` list carries the reason ``operationInProgress``.
    """
    if error.resp.status != 409:
        return False

    try:
        body = json.loads(error.content)
    except ValueError:
        return False

    error_body = body.get("error") if isinstance(body, dict) else None
    entries = error_body.get("errors") if isinstance(error_body, dict) else None

    if not isinstance(entries, list):
        return False

    return any(isinstance(entry, dict) and entry.get("reason") == "operationInProgress" for entry in entries)


async def wait_for_operation_done(service: Any, operation: dict, deadline: float) -> dict:
    """Wait until a Cloud SQL Admin operation is done, whatever its outcome.

    Args:
        service: Cloud SQL Admin API service.
        operation: Operation as the Cloud SQL Admin API returns it.
        deadline: Deadline on the ``time.monotonic`` clock, shared by the handler's waits.

    Returns:
        The finished operation, including any errors it ended with.

    Raises:
        TimeoutError: If the operation is not done by ``deadline``.
    """
    current = operation

    async for _ in poll_until(deadline):
        current = await execute(
            service.operations().get(project=operation["targetProject"], operation=operation["name"])
        )

        if current["status"] == "DONE":
            return current

    msg = f"Cloud SQL {format_operation(current)} was still {current.get('status')} at the operation deadline"
    raise TimeoutError(msg)


async def wait_for_operation_success(service: Any, operation: dict | None, deadline: float) -> None:
    """Wait until a Cloud SQL Admin operation is done and require that it succeeded.

    Args:
        service: Cloud SQL Admin API service.
        operation: Operation the write returned, or ``None`` when no write was queued.
        deadline: Deadline on the ``time.monotonic`` clock, shared by the handler's waits.

    Raises:
        RuntimeError: If the operation finished with errors.
        TimeoutError: If the operation is not done by ``deadline``.
    """
    if operation is None:
        return

    current = await wait_for_operation_done(service, operation, deadline)
    errors = current.get("error", {}).get("errors", [])

    if errors:
        messages = "; ".join(error.get("message", error.get("code", "")) for error in errors)
        msg = f"Cloud SQL {format_operation(current)} failed: {messages}"
        raise RuntimeError(msg)


async def wait_for_pending_operations(
    service: Any,
    instance_config: DatabaseInstanceConfig,
    resource_name: str,
    deadline: float,
) -> None:
    """Wait until every operation on a Cloud SQL instance is done, whatever its outcome.

    Operations are listed newest first; listing stops at the first page without a pending operation.
    Logs ``cloudsql_pending_operation_wait`` at WARNING before each wait, then ``cloudsql_pending_operation_done``,
    or ``cloudsql_pending_operation_failed`` with the errors the operation ended with. Those errors are not raised.

    Args:
        service: Cloud SQL Admin API service.
        instance_config: Configuration that locates the instance.
        resource_name: Name of the Pragmatiks resource waiting, logged with each event.
        deadline: Deadline on the ``time.monotonic`` clock, shared by the handler's waits.

    Raises:
        TimeoutError: If a pending operation is not done by ``deadline``.
    """
    pending = []
    request = service.operations().list(project=instance_config.project_id, instance=instance_config.instance_name)

    while request is not None:
        response = await execute(request, ignore_404=True) or {}
        page_pending = [operation for operation in response.get("items", []) if operation["status"] != "DONE"]

        if not page_pending:
            break

        pending += page_pending
        request = service.operations().list_next(request, response)

    for operation in pending:
        logger.warning(
            "cloudsql_pending_operation_wait resource=%s %s", resource_name, format_operation_fields(operation)
        )
        finished = await wait_for_operation_done(service, operation, deadline)
        errors = finished.get("error", {}).get("errors", [])

        if errors:
            logger.warning(
                "cloudsql_pending_operation_failed resource=%s %s errors=%s",
                resource_name,
                format_operation_fields(finished),
                errors,
            )
        else:
            logger.warning(
                "cloudsql_pending_operation_done resource=%s %s", resource_name, format_operation_fields(finished)
            )


async def run_instance_operation(
    service: Any,
    instance_config: DatabaseInstanceConfig,
    resource_name: str,
    write: Callable[[], Awaitable[dict | None]],
    deadline: float,
) -> None:
    """Send a write to a Cloud SQL instance once its pending operations are done, and wait until it succeeds.

    Logs ``cloudsql_write_refused_busy`` at WARNING when Cloud SQL refuses the write because another operation
    runs on the instance.

    Note:
        Cloud SQL runs one operation per instance at a time.

    Args:
        service: Cloud SQL Admin API service.
        instance_config: Configuration of the instance the write targets.
        resource_name: Name of the Pragmatiks resource writing, logged with each event.
        write: Sends the write and returns its operation, or None when no write was queued.
        deadline: Deadline on the ``time.monotonic`` clock, shared by the handler's waits.

    Raises:
        HttpError: If Cloud SQL refuses the write for a reason other than another running operation.
        RuntimeError: If Cloud SQL refuses the write because another operation runs on the instance,
            or the write's operation finished with errors.
        TimeoutError: If a pending operation, or the write's operation, is not done by ``deadline``.
    """
    await wait_for_pending_operations(service, instance_config, resource_name, deadline)

    try:
        operation = await write()
    except HttpError as error:
        if is_operation_in_progress(error):
            logger.warning(
                "cloudsql_write_refused_busy resource=%s instance=%s/%s",
                resource_name,
                instance_config.project_id,
                instance_config.instance_name,
            )
            message = f"Instance {instance_config.instance_name} is busy with another operation: {error.reason}"
            raise RuntimeError(message) from error
        raise

    await wait_for_operation_success(service, operation, deadline)
