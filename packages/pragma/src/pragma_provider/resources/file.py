"""Pragma platform file resource.

Reads what storage records for an uploaded file and exposes its URLs, size,
content type, checksum and upload time as outputs. File content is uploaded
separately, by ``pragma resources apply`` or the SDK's ``upload_file``.
"""

from __future__ import annotations

import contextlib
from datetime import datetime

import obstore as obs
from obstore.store import GCSStore
from pragma_sdk import AlreadyExistsError, Config, Outputs, Resource

from pragma_provider import settings


CHECKSUM_ATTRIBUTE = "checksum"
"""User metadata attribute holding the hex MD5 storage verified when the file was stored."""

DEFAULT_CONTENT_TYPE = "application/octet-stream"
"""Media type of an object stored without one."""


class FileConfig(Config):
    """Configuration for a platform-managed file.

    File content is uploaded separately, by ``pragma resources apply`` or the
    SDK's ``upload_file``. This resource tracks the stored file and exposes its
    metadata as outputs, so it has no user-configurable fields.
    """


class FileOutputs(Outputs):
    """Outputs from platform file storage.

    Attributes:
        url: Internal pragma:// URL for resource references.
        public_url: Download address under the project's file routes; it needs
            Pragmatiks credentials and redirects to the file's content.
        size: File size in bytes.
        content_type: Media type the file is served with.
        checksum: Hex MD5 of the file content, which storage verified when the
            file was stored.
        uploaded_at: When the file's current content was stored.
    """

    url: str
    public_url: str
    size: int
    content_type: str
    checksum: str
    uploaded_at: datetime


class File(Resource[FileConfig, FileOutputs]):
    """Platform-managed file storage.

    Reads the object Pragmatiks stores for an uploaded file at
    ``files/{organization_id}/{project_id}/{name}``. The object is the record
    of what is stored: its size, its ``Content-Type`` and its ``checksum``
    attribute, all fixed by the upload.

    Note:
        Requires the Pragmatiks upload flow that records the ``checksum``
        attribute on every stored object, so this provider releases after
        that flow is deployed. An object at the key without the attribute
        was not stored by that flow and is refused as not this resource's.
        The outputs change only when ``on_observe`` next runs: uploading new
        content under the name leaves the configuration unchanged.
    """

    def build_store(self) -> GCSStore:
        """Build the object store holding uploaded file content.

        Returns:
            Store for the bucket ``PRAGMA_FILE_GCS_BUCKET`` names.
        """
        return GCSStore(settings.file_bucket())

    def file_path(self) -> str:
        """Return the storage path of the file content.

        Returns:
            ``files/{organization_id}/{project_id}/{name}``.
        """
        return f"files/{settings.organization_id()}/{self.project_id}/{self.name}"

    def internal_url(self) -> str:
        """Return the pragma:// URL other resources reference the file by.

        Returns:
            ``pragma://files/{name}``.
        """
        return f"pragma://files/{self.name}"

    def public_url(self) -> str:
        """Return the file's download address, which needs Pragmatiks credentials and redirects to the content.

        Returns:
            ``{PRAGMA_FILE_PUBLIC_URL}/projects/{project_id}/files/{name}/download``.
        """
        return f"{settings.file_public_base_url()}/projects/{self.project_id}/files/{self.name}/download"

    async def on_observe(self) -> FileOutputs | None:
        """Read what storage records for the file, with one head-only read.

        Returns:
            Outputs with the file's URLs, size, content type, checksum and
            upload time, or None when no file is stored under the name.

        Raises:
            AlreadyExistsError: If the stored object carries no ``checksum``
                attribute, so a Pragmatiks upload did not store it; the
                platform then neither writes nor deletes it.
        """
        try:
            result = await obs.get_async(self.build_store(), self.file_path(), options={"head": True})
        except FileNotFoundError:
            return None

        checksum = result.attributes.get(CHECKSUM_ATTRIBUTE)

        if checksum is None:
            message = (
                f"File {self.name} in project {self.project_id} holds content a Pragmatiks upload did not store. "
                "Upload the file again with pragma resources apply, which replaces it, then apply again."
            )
            raise AlreadyExistsError(message)

        return FileOutputs(
            url=self.internal_url(),
            public_url=self.public_url(),
            size=result.meta["size"],
            content_type=result.attributes.get("Content-Type") or DEFAULT_CONTENT_TYPE,
            checksum=checksum,
            uploaded_at=result.meta["last_modified"],
        )

    async def on_create(self) -> FileOutputs:
        """Read the stored file's metadata.

        Returns:
            Outputs with the file's URLs, size, content type, checksum and
            upload time.

        Raises:
            AlreadyExistsError: As :meth:`on_observe` describes.
            FileNotFoundError: If no file is stored under the name.
        """
        outputs = await self.on_observe()

        if outputs is None:
            raise FileNotFoundError(
                f"File {self.name} in project {self.project_id} is not uploaded. "
                "Upload it with pragma resources apply, or the SDK's upload_file, then apply again."
            )

        return outputs

    async def on_update(self, previous_config: FileConfig | None) -> FileOutputs:
        """Re-read the stored file's metadata.

        Args:
            previous_config: Previous file configuration, if any.

        Returns:
            Outputs with the file's current metadata.
        """
        return await self.on_create()

    async def on_delete(self) -> None:
        """Delete the file's object from storage.

        Note:
            Idempotent: a missing object is treated as already deleted.
        """
        with contextlib.suppress(FileNotFoundError):
            await obs.delete_async(self.build_store(), self.file_path())
