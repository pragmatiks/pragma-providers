"""Base utilities for Cloud SQL resources."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from google.oauth2 import service_account
from googleapiclient import discovery
from googleapiclient.errors import HttpError

DB_CONNECTION_INFO = {
    "POSTGRES": ("postgresql", "5432"),
    "MYSQL": ("mysql", "3306"),
    "SQLSERVER": ("sqlserver", "1433"),
}


def get_credentials(credentials_data: dict[str, Any] | str) -> service_account.Credentials:
    """Create GCP credentials from config data."""
    if isinstance(credentials_data, str):
        credentials_data = json.loads(credentials_data)

    return service_account.Credentials.from_service_account_info(credentials_data)


def get_sqladmin_service(credentials: service_account.Credentials) -> Any:
    """Get Cloud SQL Admin API service."""
    return discovery.build("sqladmin", "v1", credentials=credentials)


async def run_in_executor(func: Any) -> Any:
    """Run a blocking function in the default executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func)


def extract_ips(instance: dict) -> tuple[str | None, str | None]:
    """Extract public and private IPs from instance dict."""
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
    """Get connection type and port from database version string."""
    db_family = database_version.split("_")[0]
    return DB_CONNECTION_INFO.get(db_family, ("postgresql", "5432"))


async def execute(request: Any, ignore_404: bool = False, ignore_exists: bool = False) -> Any:
    """Execute a GCP API request, optionally ignoring 404 or 'already exists' errors."""
    try:
        return await run_in_executor(request.execute)
    except HttpError as e:
        if ignore_404 and e.resp.status == 404:
            return None
        if ignore_exists and "already exists" in str(e).lower():
            return None
        raise
