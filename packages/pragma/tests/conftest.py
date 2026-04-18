"""Pytest configuration for pragma provider tests."""

from __future__ import annotations

import pytest
from pragma_sdk.provider import ProviderHarness


@pytest.fixture
def harness() -> ProviderHarness:
    """Test harness for invoking lifecycle methods."""
    return ProviderHarness()
