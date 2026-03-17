#!/usr/bin/env python3
"""Seed official providers into the Pragmatiks store.

Publishes all 4 official providers (gcp, kubernetes, agno, qdrant) to the
store and sets their trust tier to OFFICIAL via direct SurrealDB access.

Usage:
    # Publish and set trust tier (requires API + SurrealDB access)
    python scripts/seed_store.py --api-url http://localhost:8000 --surreal-url http://localhost:9000

    # Publish only (uses API)
    python scripts/seed_store.py --api-url http://localhost:8000 --publish-only

    # Set trust tier only (uses SurrealDB directly)
    python scripts/seed_store.py --surreal-url http://localhost:9000 --trust-tier-only
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import httpx
from pragma_sdk.client import PragmaClient
from pragma_sdk.models import VersionStatus


REPO_ROOT = Path(__file__).resolve().parent.parent

TARBALL_EXCLUDES = {
    ".git",
    "__pycache__",
    ".venv",
    ".env",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    "dist",
    "build",
    ".tox",
    ".nox",
    "uv.lock",
}

PROVIDERS = ["gcp", "kubernetes", "agno", "qdrant"]

BUILD_POLL_INTERVAL = 5.0
BUILD_TIMEOUT = 600


@dataclass
class ProviderInfo:
    """Parsed provider metadata from pyproject.toml."""

    name: str
    version: str
    package_dir: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Seed official providers into the Pragmatiks store.",
    )
    parser.add_argument(
        "--api-url",
        default="http://dev-pragmatiks-api.pragmatiks.svc.cluster.local:8000",
        help="Pragma API base URL (default: local dev)",
    )
    parser.add_argument(
        "--surreal-url",
        default="http://pragmatiks-surrealdb.pragmatiks.svc.cluster.local:9000",
        help="SurrealDB base URL (default: local dev)",
    )
    parser.add_argument(
        "--surreal-user",
        default="root",
        help="SurrealDB username (default: root)",
    )
    parser.add_argument(
        "--surreal-pass",
        default=None,
        help="SurrealDB password (reads from SURREAL_PASS env if not set)",
    )
    parser.add_argument(
        "--surreal-ns",
        default=None,
        help="SurrealDB namespace (organization_id). Required for --trust-tier-only or full run.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Override version for all providers (default: read from each pyproject.toml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-publish even if version/hash already exists",
    )
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="Only publish to API, skip trust tier update",
    )
    parser.add_argument(
        "--trust-tier-only",
        action="store_true",
        help="Only set trust tier via SurrealDB, skip publish",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=PROVIDERS,
        help="Specific providers to seed (default: all)",
    )

    return parser.parse_args()


def load_provider_info(name: str, version_override: str | None = None) -> ProviderInfo:
    """Load provider metadata from its pyproject.toml.

    Args:
        name: Provider directory name under packages/.
        version_override: If set, use this version instead of pyproject.toml.

    Returns:
        ProviderInfo with name, version, and package directory path.
    """
    package_dir = REPO_ROOT / "packages" / name
    pyproject_path = package_dir / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    provider_name = data["tool"]["pragma"]["provider"]
    version = version_override or data["project"]["version"]

    return ProviderInfo(name=provider_name, version=version, package_dir=package_dir)


def create_tarball(source_dir: Path) -> bytes:
    """Create a gzipped tarball of a provider source directory.

    Excludes common development artifacts like .git, __pycache__, .venv, etc.

    Args:
        source_dir: Path to the provider package directory.

    Returns:
        Gzipped tarball bytes.
    """
    buffer = io.BytesIO()

    def exclude_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(tarinfo.name).parts

        for part in parts:
            if part in TARBALL_EXCLUDES:
                return None
            for pattern in TARBALL_EXCLUDES:
                if pattern.startswith("*") and part.endswith(pattern[1:]):
                    return None

        return tarinfo

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(source_dir, arcname=".", filter=exclude_filter)

    buffer.seek(0)
    return buffer.read()


def publish_provider(client: PragmaClient, info: ProviderInfo, *, force: bool = False) -> None:
    """Publish a provider to the store and wait for build completion.

    Args:
        client: Authenticated PragmaClient.
        info: Provider metadata.
        force: If True, bypass content hash deduplication.

    Raises:
        RuntimeError: If build fails or times out.
        httpx.HTTPStatusError: If the publish API request fails.
    """
    print(f"  Creating tarball from {info.package_dir}...")
    tarball = create_tarball(info.package_dir)
    print(f"  Tarball size: {len(tarball) / 1024:.1f} KB")

    print(f"  Publishing {info.name} v{info.version}...")

    try:
        result = client.publish_store_provider(
            info.name,
            tarball,
            info.version,
            force=force,
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 409 and not force:
            print("  Version already exists (use --force to overwrite)")
            return
        raise

    print(f"  Build started (status: {result.status})")

    if result.status == VersionStatus.PUBLISHED:
        print("  Already published!")
        return

    start = time.monotonic()

    while time.monotonic() - start < BUILD_TIMEOUT:
        time.sleep(BUILD_POLL_INTERVAL)

        status = client.get_store_build_status(info.name, info.version)

        if status.status == VersionStatus.PUBLISHED:
            print("  Build complete!")
            return

        if status.status == VersionStatus.FAILED:
            raise RuntimeError(f"Build failed for {info.name} v{info.version}: {status.error_message}")

        elapsed = int(time.monotonic() - start)
        print(f"  Waiting for build... ({elapsed}s, status: {status.status})")

    raise RuntimeError(f"Build timed out for {info.name} v{info.version} after {BUILD_TIMEOUT}s")


def set_trust_tier(
    surreal_url: str,
    surreal_user: str,
    surreal_pass: str,
    namespace: str,
    provider_name: str,
) -> None:
    """Set a provider's trust_tier to OFFICIAL via direct SurrealDB HTTP API.

    Args:
        surreal_url: SurrealDB base URL.
        surreal_user: SurrealDB username.
        surreal_pass: SurrealDB password.
        namespace: SurrealDB namespace (organization_id).
        provider_name: Store provider name to update.
    """
    query = "UPDATE store_providers SET trust_tier = 'official' WHERE name = $name"

    response = httpx.post(
        f"{surreal_url}/sql",
        json={"query": query, "vars": {"name": provider_name}},
        headers={
            "Accept": "application/json",
            "Surreal-NS": namespace,
            "Surreal-DB": "app",
        },
        auth=(surreal_user, surreal_pass),
    )
    response.raise_for_status()

    results = response.json()
    print(f"  SurrealDB response: {results}")


def main() -> None:
    """Run the seed script."""
    args = parse_args()

    if args.publish_only and args.trust_tier_only:
        print("Error: --publish-only and --trust-tier-only are mutually exclusive")
        sys.exit(1)

    surreal_pass = args.surreal_pass
    need_surreal = not args.publish_only

    if need_surreal and not surreal_pass:
        surreal_pass = os.environ.get("SURREAL_PASS")

        if not surreal_pass:
            print("Error: SurrealDB password required. Use --surreal-pass or set SURREAL_PASS env var.")
            sys.exit(1)

    if need_surreal and not args.surreal_ns:
        print("Error: --surreal-ns (organization namespace) is required for trust tier updates.")
        sys.exit(1)

    providers = [load_provider_info(name, args.version) for name in args.providers]

    if not args.trust_tier_only:
        client = PragmaClient(base_url=args.api_url, require_auth=True)

        for info in providers:
            print(f"\n{'=' * 60}")
            print(f"Publishing: {info.name} v{info.version}")
            print(f"{'=' * 60}")

            try:
                publish_provider(client, info, force=args.force)
            except Exception as e:
                print(f"  FAILED: {e}")
                sys.exit(1)

    if not args.publish_only:
        print(f"\n{'=' * 60}")
        print("Setting trust tier to OFFICIAL")
        print(f"{'=' * 60}")

        for info in providers:
            print(f"\n  Updating {info.name}...")

            try:
                set_trust_tier(
                    args.surreal_url,
                    args.surreal_user,
                    surreal_pass,
                    args.surreal_ns,
                    info.name,
                )
            except Exception as e:
                print(f"  FAILED: {e}")
                sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Seed complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
