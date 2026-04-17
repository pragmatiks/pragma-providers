"""Tests for pragma/config resource."""

from __future__ import annotations

from pragma_sdk.provider import ProviderHarness

from pragma_provider import ConfigResource, ConfigResourceConfig


async def test_config_create_exposes_string_data(harness: ProviderHarness) -> None:
    config = ConfigResourceConfig(data={"project_id": "my-project", "region": "us-central1"})

    result = await harness.invoke_create(ConfigResource, name="gcp", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.project_id == "my-project"
    assert result.outputs.region == "us-central1"


async def test_config_create_supports_mixed_json_types(harness: ProviderHarness) -> None:
    config = ConfigResourceConfig(
        data={
            "name": "test",
            "count": 42,
            "enabled": True,
            "tags": ["a", "b"],
            "nested": {"key": "value"},
        }
    )

    result = await harness.invoke_create(ConfigResource, name="mixed", config=config)

    assert result.success
    assert result.outputs is not None
    assert result.outputs.name == "test"
    assert result.outputs.count == 42
    assert result.outputs.enabled is True
    assert result.outputs.tags == ["a", "b"]
    assert result.outputs.nested == {"key": "value"}


async def test_config_update_reflects_changed_data(harness: ProviderHarness) -> None:
    previous = ConfigResourceConfig(data={"region": "us-central1"})
    current = ConfigResourceConfig(data={"region": "us-east1", "zone": "a"})

    result = await harness.invoke_update(
        ConfigResource,
        name="gcp",
        config=current,
        previous_config=previous,
    )

    assert result.success
    assert result.outputs is not None
    assert result.outputs.region == "us-east1"
    assert result.outputs.zone == "a"


async def test_config_delete_succeeds(harness: ProviderHarness) -> None:
    config = ConfigResourceConfig(data={"key": "value"})

    result = await harness.invoke_delete(ConfigResource, name="gcp", config=config)

    assert result.success


async def test_config_create_with_empty_data(harness: ProviderHarness) -> None:
    config = ConfigResourceConfig(data={})

    result = await harness.invoke_create(ConfigResource, name="empty", config=config)

    assert result.success
    assert result.outputs is not None
