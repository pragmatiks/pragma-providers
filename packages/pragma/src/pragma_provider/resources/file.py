"""Pragma platform file resource.

Reads what storage records for the file uploaded under the resource's name,
through the short-lived download address Pragmatiks signs for the resource's
``content`` reference, and exposes its reference, size, content type,
checksum and upload time as outputs. File content is uploaded separately, by
``pragma resources apply`` or the SDK's ``upload_file``.
"""

from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime

import httpx
from pragma_sdk import RECOVERY_GUIDANCE, Config, FileField, FileReference, Outputs, Resource


CHECKSUM_HEADER = "x-goog-meta-checksum"
"""Response header holding the hex MD5 storage verified when the file was stored."""

STORED_SIZE_HEADER = "x-goog-stored-content-length"
"""Response header holding the stored file's size in bytes."""

UPLOADED_AT_HEADER = "Last-Modified"
"""Response header holding when the file's current content was stored."""

DEFAULT_CONTENT_TYPE = "application/octet-stream"
"""Media type of a file stored without one."""

FILE_REFERENCE_PREFIX = "pragma://files/"
"""Prefix of a Pragmatiks file reference, followed by the file's name."""

STORAGE_READ_TIMEOUT = httpx.Timeout(30.0)
"""Time allowed to connect to storage and receive the response headers."""

UNNAMED_FILE_MESSAGE = (
    "The file this resource reads arrived without its pragma://files/<name> reference. "
    "The pragma provider needs pragmatiks-sdk 32.1.0 or later, whose worker records the reference "
    "each download address is signed for; install a pragma provider version that requires it, then apply again."
)
"""Failure reason for a ``content`` field that carries no reference, under a pragmatiks-sdk older than 32.1.0."""


class FileConfig(Config):
    """Configuration for a platform-managed file.

    Attributes:
        content: The resource's own file, as the reference
            ``pragma://files/<name>`` with the resource's name.
            ``pragma resources apply`` uploads a ``content: "@./path"`` file
            under the resource's name and writes this reference in its place.
    """

    content: FileField


class FileOutputs(Outputs):
    """Outputs from platform file storage.

    Attributes:
        url: The file's ``pragma://files/<name>`` reference, which other
            resources' configs take.
        size: File size in bytes.
        content_type: Media type the file is served with.
        checksum: Hex MD5 of the file content, which storage verified when the
            file was stored.
        uploaded_at: When the file's current content was stored.
    """

    url: str
    size: int
    content_type: str
    checksum: str
    uploaded_at: datetime


def require_own_file_reference(resource_name: str, content: FileField) -> FileReference:
    """Return the file a ``pragma/file`` resource reads, after checking it is its own.

    Pragmatiks deletes the file named after a ``pragma/file`` resource once
    the resource is deleted, so a resource reads only that file and never
    views another resource's content.

    Args:
        resource_name: Name of the resource, which is its file's name.
        content: The resource's ``content`` field, as the worker built it.

    Returns:
        The file reference, holding the signed download address.

    Raises:
        RuntimeError: If ``content`` carries no reference, as
            ``UNNAMED_FILE_MESSAGE`` states, or names another file.
    """
    if not isinstance(content, FileReference) or content.name is None:
        raise RuntimeError(UNNAMED_FILE_MESSAGE)

    own_reference = f"{FILE_REFERENCE_PREFIX}{resource_name}"

    if content.name != own_reference:
        message = f"pragma/file {resource_name} reads only its own file; set content to {own_reference}."
        raise RuntimeError(message)

    return content


def check_storage_status(file_name: str, status_code: int) -> None:
    """Check storage's answer to the read of a file's download address.

    Args:
        file_name: Name of the file, quoted in the error.
        status_code: HTTP status storage answered with.

    Raises:
        FileNotFoundError: If no file is stored under the name.
        RuntimeError: If storage refused the address, which expired before
            the operation read it, or answered with any other failure.
    """
    if status_code == httpx.codes.NOT_FOUND:
        message = (
            f"File {file_name} is not uploaded. "
            "Upload it with pragma resources apply, or the SDK's upload_file, then apply again."
        )
        raise FileNotFoundError(message)

    if status_code == httpx.codes.FORBIDDEN:
        message = (
            f"The download address for file {file_name} expired or was refused before the operation read it. "
            f"{RECOVERY_GUIDANCE}"
        )
        raise RuntimeError(message)

    if not httpx.codes.is_success(status_code):
        message = f"Reading file {file_name} from storage failed (status {status_code}). {RECOVERY_GUIDANCE}"
        raise RuntimeError(message)


def build_file_outputs(file_name: str, headers: httpx.Headers) -> FileOutputs:
    """Build the file's outputs from the headers storage answered its read with.

    Args:
        file_name: Name of the file.
        headers: Response headers of the read of the file's download address.

    Returns:
        Outputs with the file's reference, size, content type, checksum and
        upload time.

    Raises:
        RuntimeError: If the headers carry no checksum, so a Pragmatiks
            upload did not store the file, or no readable size or upload time.
    """
    checksum = headers.get(CHECKSUM_HEADER)

    if checksum is None:
        message = (
            f"File {file_name} holds content a Pragmatiks upload did not store. "
            "Upload the file again with pragma resources apply, which replaces it, then apply again."
        )
        raise RuntimeError(message)

    try:
        size = int(headers[STORED_SIZE_HEADER])
        uploaded_at = parsedate_to_datetime(headers[UPLOADED_AT_HEADER])
    except (KeyError, ValueError, TypeError):
        message = f"Storage answered without the size or upload time of file {file_name}. {RECOVERY_GUIDANCE}"
        raise RuntimeError(message) from None

    return FileOutputs(
        url=f"{FILE_REFERENCE_PREFIX}{file_name}",
        size=size,
        content_type=headers.get("Content-Type") or DEFAULT_CONTENT_TYPE,
        checksum=checksum,
        uploaded_at=uploaded_at,
    )


async def read_stored_headers(client: httpx.AsyncClient, file_name: str, address: str) -> httpx.Headers:
    """Read the headers storage answers a file's download address with, without downloading the body.

    Args:
        client: HTTP client the read is sent with.
        file_name: Name of the file, quoted in the error.
        address: The signed download address of the file.

    Returns:
        The response headers.

    Raises:
        FileNotFoundError: As :func:`check_storage_status` describes.
        RuntimeError: As :func:`check_storage_status` describes, or if
            storage cannot be reached or does not answer in time.
    """  # noqa: DOC502
    try:
        response = await client.send(client.build_request("GET", address), stream=True)
        await response.aclose()
    except httpx.TransportError as error:
        message = f"Reading file {file_name} from storage failed ({type(error).__name__}). {RECOVERY_GUIDANCE}"
        raise RuntimeError(message) from None

    check_storage_status(file_name, response.status_code)
    return response.headers


class File(Resource[FileConfig, FileOutputs]):
    """Platform-managed file: a read-only view of the file uploaded under the resource's name.

    Reads the file through the short-lived download address Pragmatiks signs
    for ``config.content`` on every operation, and never reads the bucket,
    writes or deletes, or holds a storage identity. ``config.content`` must
    name the resource's own file, which Pragmatiks deletes once the resource
    is deleted.

    Pragmatiks guarantees the address answers a GET with ``Content-Type``,
    ``x-goog-stored-content-length``, ``x-goog-meta-checksum`` and
    ``Last-Modified``; the outputs come from those headers alone.

    Note:
        The type is computed, so it has no drift detection: the outputs change
        only when the resource is applied again, and uploading new content
        under the name leaves the configuration unchanged. A file
        deleted through the files API leaves the resource ``READY`` until its
        next operation.
    """

    computed = True

    async def read_stored_file(self, client: httpx.AsyncClient) -> FileOutputs:
        """Read what storage records for the file ``config.content`` references.

        Args:
            client: HTTP client the read is sent with.

        Returns:
            Outputs with the file's reference, size, content type, checksum
            and upload time.

        Raises:
            FileNotFoundError: If no file is stored under the name.
            RuntimeError: As :func:`require_own_file_reference`,
                :func:`read_stored_headers` and :func:`build_file_outputs`
                describe.
        """  # noqa: DOC502
        own_file = require_own_file_reference(self.name, self.config.content)
        headers = await read_stored_headers(client, self.name, own_file.url)
        return build_file_outputs(self.name, headers)

    async def on_create(self) -> FileOutputs:
        """Read the stored file's metadata.

        Returns:
            Outputs as :meth:`read_stored_file` describes.
        """
        async with httpx.AsyncClient(timeout=STORAGE_READ_TIMEOUT) as client:
            return await self.read_stored_file(client)

    async def on_update(self, previous_config: FileConfig | None) -> FileOutputs:
        """Re-read the stored file's metadata.

        Args:
            previous_config: Previous file configuration, if any.

        Returns:
            Outputs as :meth:`read_stored_file` describes.
        """
        return await self.on_create()

    async def on_delete(self) -> None:
        """Tear the resource down without touching the stored content.

        Note:
            The provider never deletes: deactivating the resource keeps the
            content, and Pragmatiks deletes it once the resource is deleted.
        """
