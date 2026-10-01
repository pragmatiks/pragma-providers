"""Pragma platform file resource.

Reads storage metadata for uploaded files and exposes URLs and file metadata
as outputs. File content is uploaded separately through the Pragmatiks API.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import obstore as obs
from obstore.store import GCSStore
from pragma_sdk import Config, Outputs, Resource

from pragma_provider import settings


class FileConfig(Config):
    """Configuration for a platform-managed file.

    File content is uploaded separately through the Pragmatiks API. This
    resource tracks the uploaded file and exposes its metadata as outputs,
    so no user-configurable fields are required today.
    """


class FileOutputs(Outputs):
    """Outputs from platform file storage.

    Attributes:
        url: Internal pragma:// URL for resource references.
        public_url: Public HTTP URL for external/user access.
        size: File size in bytes.
        content_type: MIME type of the file.
        checksum: SHA256 hash of the file content.
        uploaded_at: Timestamp when file was uploaded.
    """

    url: str
    public_url: str
    size: int
    content_type: str
    checksum: str
    uploaded_at: datetime


class File(Resource[FileConfig, FileOutputs]):
    """Platform-managed file storage.

    Reads metadata from object storage for files uploaded via the Pragmatiks
    API. Files are stored under ``files/{organization_id}/{name}`` and their
    metadata alongside them as ``files/{organization_id}/{name}.meta``.
    """

    def load_store(self) -> GCSStore:
        """Load the object store holding uploaded file content.

        Returns:
            Store for the bucket ``PRAGMA_FILE_GCS_BUCKET`` names.
        """
        return GCSStore(settings.file_bucket())

    def file_path(self) -> str:
        """Return the storage path of the file content.

        Returns:
            ``files/{organization_id}/{name}``.
        """
        return f"files/{settings.organization_id()}/{self.name}"

    def metadata_path(self) -> str:
        """Return the storage path of the file's metadata sidecar.

        Returns:
            The file content path with a ``.meta`` suffix.
        """
        return f"{self.file_path()}.meta"

    def internal_url(self) -> str:
        """Return the pragma:// URL other resources reference the file by.

        Returns:
            ``pragma://files/{name}``.
        """
        return f"pragma://files/{self.name}"

    def public_url(self) -> str:
        """Return the file's public download URL.

        Returns:
            ``{PRAGMA_FILE_PUBLIC_URL}/files/{name}/download``.
        """
        return f"{settings.file_public_base_url()}/files/{self.name}/download"

    async def fetch_content_metadata(self, store: GCSStore) -> dict[str, Any]:
        """Derive the file's metadata from its stored content.

        Args:
            store: Store holding the file content.

        Returns:
            Size, content type, SHA256 checksum, and upload timestamp.
        """
        response = await obs.get_async(store, self.file_path())
        checksum = hashlib.sha256()

        async for chunk in response:
            checksum.update(chunk)

        return {
            "size": response.meta["size"],
            "content_type": response.attributes.get("Content-Type", "application/octet-stream"),
            "checksum": checksum.hexdigest(),
            "uploaded_at": response.meta["last_modified"],
        }

    async def on_observe(self) -> FileOutputs | None:
        """Read the uploaded file's metadata from storage.

        Returns:
            Outputs with file URL, size, checksum, and upload timestamp, or
            None when no content is uploaded.
        """
        store = self.load_store()

        try:
            await obs.head_async(store, self.file_path())
        except FileNotFoundError:
            return None

        try:
            response = await obs.get_async(store, self.metadata_path())
            content = await response.bytes_async()
            data = json.loads(bytes(content))
        except FileNotFoundError:
            data = await self.fetch_content_metadata(store)

        return FileOutputs(
            url=self.internal_url(),
            public_url=self.public_url(),
            size=data["size"],
            content_type=data["content_type"],
            checksum=data["checksum"],
            uploaded_at=data["uploaded_at"],
        )

    async def on_create(self) -> FileOutputs:
        """Read file metadata from storage.

        Returns:
            Outputs with file URL, size, checksum, and upload timestamp.

        Raises:
            FileNotFoundError: If the file content has not been uploaded.
        """
        outputs = await self.on_observe()

        if outputs is None:
            raise FileNotFoundError(f"File content not uploaded. Use POST /files/{self.name}/upload first.")

        return outputs

    async def on_update(self, previous_config: FileConfig | None) -> FileOutputs:
        """Re-read file metadata from storage.

        Args:
            previous_config: Previous file configuration, if any.

        Returns:
            Outputs with updated file metadata.
        """
        return await self.on_create()

    async def on_delete(self) -> None:
        """Delete file and metadata from storage."""
        store = self.load_store()

        for path in [self.metadata_path(), self.file_path()]:
            try:
                await obs.delete_async(store, path)
            except FileNotFoundError:
                pass
