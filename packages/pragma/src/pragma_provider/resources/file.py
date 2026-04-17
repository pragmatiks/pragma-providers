"""Pragma platform file resource.

Reads GCS metadata for uploaded files and exposes URLs and metadata as outputs.
File content is uploaded separately via POST /files/{name}/upload on the API.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import obstore as obs
from obstore.store import GCSStore
from pragma_sdk import Config, Field, Outputs, Resource
from pydantic import Field as PydanticField


class FileConfig(Config):
    """Configuration for a platform-managed file.

    File content is uploaded separately via the /files endpoint.
    This config only contains metadata about the file.

    Attributes:
        content_type: MIME type of the file.
        description: Optional human-readable description.
    """

    content_type: Field[str] = PydanticField(default="application/octet-stream")
    description: Field[str] | None = None


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

    Reads metadata from GCS for files uploaded via the API's /files endpoint.
    Files are stored at: gs://{bucket}/files/{organization_id}/{name}
    Metadata at: gs://{bucket}/files/{organization_id}/{name}.meta
    """

    def _get_store(self) -> GCSStore:
        bucket = os.environ["PRAGMA_FILE_GCS_BUCKET"]
        return GCSStore(bucket)

    def _organization_id(self) -> str:
        return os.environ["PRAGMA_RUNTIME_ORGANIZATION_ID"]

    def _meta_path(self) -> str:
        return f"files/{self._organization_id()}/{self.name}.meta"

    def _internal_url(self) -> str:
        return f"pragma://files/{self.name}"

    def _public_url(self) -> str:
        base_url = os.environ["PRAGMA_FILE_PUBLIC_URL"]
        return f"{base_url}/files/{self.name}/download"

    async def _read_metadata(self) -> FileOutputs:
        store = self._get_store()
        meta_path = self._meta_path()

        try:
            response = await obs.get_async(store, meta_path)
            content = await response.bytes_async()
            data = json.loads(bytes(content))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File content not uploaded. Use POST /files/{self.name}/upload first.") from exc

        return FileOutputs(
            url=self._internal_url(),
            public_url=self._public_url(),
            size=data["size"],
            content_type=data["content_type"],
            checksum=data["checksum"],
            uploaded_at=data["uploaded_at"],
        )

    async def on_create(self) -> FileOutputs:
        """Read file metadata from storage.

        Returns:
            Outputs with file URL, size, checksum, and upload timestamp.
        """
        return await self._read_metadata()

    async def on_update(self, previous_config: FileConfig) -> FileOutputs:
        """Re-read file metadata from storage.

        Args:
            previous_config: Previous file configuration.

        Returns:
            Outputs with updated file metadata.
        """
        return await self._read_metadata()

    async def on_delete(self) -> None:
        """Delete file and metadata from storage."""
        store = self._get_store()
        file_path = f"files/{self._organization_id()}/{self.name}"
        meta_path = self._meta_path()

        for path in [file_path, meta_path]:
            try:
                await obs.delete_async(store, path)
            except FileNotFoundError:
                pass
