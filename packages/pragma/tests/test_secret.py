"""Tests for pragma/secret resource."""

from __future__ import annotations

from pragma_sdk.provider import ProviderHarness

from pragma_provider import Secret, SecretConfig


async def test_secret_create_exposes_data_as_outputs(harness: ProviderHarness) -> None:
    config = SecretConfig(data={"api_key": "sk-123", "db_password": "hunter2"})

    result = await harness.invoke_create(Secret, name="my-secret", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.api_key == "sk-123"
    assert result.outputs.db_password == "hunter2"


async def test_secret_update_reflects_changed_data(harness: ProviderHarness) -> None:
    previous = SecretConfig(data={"api_key": "sk-old"})
    current = SecretConfig(data={"api_key": "sk-new", "extra": "value"})

    result = await harness.invoke_update(
        Secret,
        name="my-secret",
        config=current,
        previous_config=previous,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.api_key == "sk-new"
    assert result.outputs.extra == "value"


async def test_secret_delete_succeeds(harness: ProviderHarness) -> None:
    config = SecretConfig(data={"key": "value"})

    result = await harness.invoke_delete(Secret, name="my-secret", config=config)

    assert result.success


async def test_secret_create_with_empty_data(harness: ProviderHarness) -> None:
    config = SecretConfig(data={})

    result = await harness.invoke_create(Secret, name="empty-secret", config=config)

    assert result.success
    assert result.outputs is not None
