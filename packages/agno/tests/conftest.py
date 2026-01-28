"""Shared fixtures for agno_provider tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_gke_outputs(mocker: "MockerFixture") -> "MagicMock":
    """Mock GKE outputs with cluster credentials."""
    outputs = mocker.MagicMock()
    outputs.endpoint = "10.0.0.1"
    outputs.cluster_ca_certificate = "Y2VydGlmaWNhdGU="  # base64 of "certificate"
    outputs.name = "test-cluster"
    outputs.location = "europe-west4"
    outputs.status = "RUNNING"
    return outputs


@pytest.fixture
def mock_gke_resource(mocker: "MockerFixture", mock_gke_outputs: "MagicMock") -> "MagicMock":
    """Mock GKE resource with outputs."""
    resource = mocker.MagicMock()
    resource.outputs = mock_gke_outputs
    return resource


@pytest.fixture
def mock_model_outputs(mocker: "MockerFixture") -> "MagicMock":
    """Mock model outputs with URL."""
    outputs = mocker.MagicMock()
    outputs.url = "http://anthropic-messages.default.svc.cluster.local"
    return outputs


@pytest.fixture
def mock_model_resource(mocker: "MockerFixture", mock_model_outputs: "MagicMock") -> "MagicMock":
    """Mock model resource with outputs."""
    resource = mocker.MagicMock()
    resource.outputs = mock_model_outputs
    return resource


@pytest.fixture
def mock_embeddings_outputs(mocker: "MockerFixture") -> "MagicMock":
    """Mock embeddings outputs with URL."""
    outputs = mocker.MagicMock()
    outputs.url = "http://openai-embeddings.default.svc.cluster.local"
    return outputs


@pytest.fixture
def mock_embeddings_resource(mocker: "MockerFixture", mock_embeddings_outputs: "MagicMock") -> "MagicMock":
    """Mock embeddings resource with outputs."""
    resource = mocker.MagicMock()
    resource.outputs = mock_embeddings_outputs
    return resource


@pytest.fixture
def mock_vector_store_outputs(mocker: "MockerFixture") -> "MagicMock":
    """Mock vector store outputs with URL."""
    outputs = mocker.MagicMock()
    outputs.url = "http://qdrant-collection.default.svc.cluster.local:6333"
    return outputs


@pytest.fixture
def mock_vector_store_resource(mocker: "MockerFixture", mock_vector_store_outputs: "MagicMock") -> "MagicMock":
    """Mock vector store resource with outputs."""
    resource = mocker.MagicMock()
    resource.outputs = mock_vector_store_outputs
    return resource


@pytest.fixture
def mock_subprocess(mocker: "MockerFixture") -> "MagicMock":
    """Mock subprocess.run for kubectl commands."""
    mock_run = mocker.patch("subprocess.run")

    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        # Handle kubectl get deployment (readiness check)
        if cmd[0] == "kubectl" and "deployment" in cmd:
            result.stdout = "1"  # 1 ready replica

        return result

    mock_run.side_effect = run_side_effect
    return mock_run


@pytest.fixture
def mock_asyncio_sleep(mocker: "MockerFixture") -> "MagicMock":
    """Mock asyncio.sleep to avoid actual waits."""
    return mocker.patch("agno_provider.resources.agent.asyncio.sleep", return_value=None)
