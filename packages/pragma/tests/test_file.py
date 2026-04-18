"""Tests for pragma/file resource."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pragma_sdk.provider import ProviderHarness

from pragma_provider import File, FileConfig


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


SAMPLE_METADATA = {
    "size": 1024,
    "content_type": "application/pdf",
    "checksum": "abc123def456",
    "uploaded_at": "2025-01-15T12:00:00+00:00",
}


def _mock_obstore(mocker: MockerFixture, metadata: dict | None = None) -> None:
    """Mock obstore get_async and delete_async for file tests."""
    if metadata is None:
        metadata = SAMPLE_METADATA

    meta_bytes = json.dumps(metadata).encode()

    mock_response = mocker.MagicMock()
    mock_response.bytes_async = mocker.AsyncMock(return_value=meta_bytes)

    mocker.patch(
        "pragma_provider.resources.file.obs.get_async",
        new_callable=mocker.AsyncMock,
        return_value=mock_response,
    )
    mocker.patch("pragma_provider.resources.file.obs.delete_async", new_callable=mocker.AsyncMock)


async def test_file_create_reads_metadata(
    harness: ProviderHarness,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRAGMA_FILE_GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("PRAGMA_FILE_PUBLIC_URL", "https://api.example.com")
    monkeypatch.setenv("PRAGMA_RUNTIME_ORGANIZATION_ID", "org-123")

    _mock_obstore(mocker)

    config = FileConfig()

    result = await harness.invoke_create(File, name="report.pdf", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.url == "pragma://files/report.pdf"
    assert result.outputs.public_url == "https://api.example.com/files/report.pdf/download"
    assert result.outputs.size == 1024
    assert result.outputs.content_type == "application/pdf"
    assert result.outputs.checksum == "abc123def456"


async def test_file_update_re_reads_metadata(
    harness: ProviderHarness,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRAGMA_FILE_GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("PRAGMA_FILE_PUBLIC_URL", "https://api.example.com")
    monkeypatch.setenv("PRAGMA_RUNTIME_ORGANIZATION_ID", "org-123")

    _mock_obstore(mocker)

    previous = FileConfig()
    current = FileConfig()

    result = await harness.invoke_update(
        File,
        name="report.pdf",
        config=current,
        previous_config=previous,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.size == 1024


async def test_file_delete_removes_file_and_metadata(
    harness: ProviderHarness,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRAGMA_FILE_GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("PRAGMA_FILE_PUBLIC_URL", "https://api.example.com")
    monkeypatch.setenv("PRAGMA_RUNTIME_ORGANIZATION_ID", "org-123")

    mock_delete = mocker.patch(
        "pragma_provider.resources.file.obs.delete_async",
        new_callable=mocker.AsyncMock,
    )

    config = FileConfig()

    result = await harness.invoke_delete(File, name="report.pdf", config=config)

    assert result.success
    assert mock_delete.call_count == 2


async def test_file_delete_idempotent_when_not_found(
    harness: ProviderHarness,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRAGMA_FILE_GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("PRAGMA_FILE_PUBLIC_URL", "https://api.example.com")
    monkeypatch.setenv("PRAGMA_RUNTIME_ORGANIZATION_ID", "org-123")

    mocker.patch(
        "pragma_provider.resources.file.obs.delete_async",
        new_callable=mocker.AsyncMock,
        side_effect=FileNotFoundError("not found"),
    )

    config = FileConfig()

    result = await harness.invoke_delete(File, name="missing.pdf", config=config)

    assert result.success


async def test_file_create_fails_when_not_uploaded(
    harness: ProviderHarness,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRAGMA_FILE_GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("PRAGMA_FILE_PUBLIC_URL", "https://api.example.com")
    monkeypatch.setenv("PRAGMA_RUNTIME_ORGANIZATION_ID", "org-123")

    mocker.patch(
        "pragma_provider.resources.file.obs.get_async",
        new_callable=mocker.AsyncMock,
        side_effect=FileNotFoundError("not found"),
    )

    config = FileConfig()

    result = await harness.invoke_create(File, name="missing.pdf", config=config)

    assert result.failed
    assert result.error is not None
    assert "not uploaded" in str(result.error).lower()


async def test_file_create_fails_with_runtime_error_when_env_missing(
    harness: ProviderHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRAGMA_FILE_GCS_BUCKET", raising=False)
    monkeypatch.delenv("PRAGMA_FILE_PUBLIC_URL", raising=False)
    monkeypatch.delenv("PRAGMA_RUNTIME_ORGANIZATION_ID", raising=False)

    config = FileConfig()

    result = await harness.invoke_create(File, name="report.pdf", config=config)

    assert result.failed
    assert result.error is not None
    error_message = str(result.error)
    assert "PRAGMA_FILE_GCS_BUCKET" in error_message
    assert "Pragmatiks runtime" in error_message
