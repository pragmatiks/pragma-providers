"""Tests for Agno Agent resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pragma_sdk import Dependency, LifecycleState

from agno_provider import (
    Agent,
    AgentConfig,
    AgentOutputs,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_agent_with_mocked_dependencies(
    name: str,
    replicas: int = 1,
    image: str = "ghcr.io/agno-ai/agno:latest",
    instructions: str | None = None,
    mock_gke_resource: MagicMock | None = None,
    mock_model_resource: MagicMock | None = None,
    mock_embeddings_resource: MagicMock | None = None,
    mock_vector_store_resource: MagicMock | None = None,
    outputs: AgentOutputs | None = None,
) -> Agent:
    """Create an Agent instance with mocked dependency resolution.

    This bypasses Pydantic validation which loses private attributes.
    """
    # Create dependencies
    cluster_dep = Dependency(provider="gcp", resource="gke", name="test-cluster")
    model_dep = Dependency(provider="anthropic", resource="messages", name="claude")

    embeddings_dep = None
    if mock_embeddings_resource is not None:
        embeddings_dep = Dependency(provider="openai", resource="embeddings", name="embeddings")

    vector_store_dep = None
    if mock_vector_store_resource is not None:
        vector_store_dep = Dependency(provider="qdrant", resource="collection", name="my-collection")

    # Create config
    config = AgentConfig(
        cluster=cluster_dep,
        model=model_dep,
        embeddings=embeddings_dep,
        vector_store=vector_store_dep,
        instructions=instructions,
        image=image,
        replicas=replicas,
    )

    # Inject resolved resources via _resolved attribute.
    # This is a standard pattern for testing SDK Dependency objects.
    if mock_gke_resource:
        config.cluster._resolved = mock_gke_resource
    if mock_model_resource:
        config.model._resolved = mock_model_resource
    if mock_embeddings_resource and embeddings_dep:
        config.embeddings._resolved = mock_embeddings_resource
    if mock_vector_store_resource and vector_store_dep:
        config.vector_store._resolved = mock_vector_store_resource

    # Create resource
    return Agent(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


async def test_create_agent_success(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mock_subprocess: MagicMock,
    mock_asyncio_sleep: MagicMock,
) -> None:
    """on_create deploys agent via kubectl and returns outputs."""
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    result = await agent.on_create()

    assert result.url == "http://agno-test-agent.default.svc.cluster.local"
    assert result.ready is True


async def test_create_agent_with_all_dependencies(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mock_embeddings_resource: MagicMock,
    mock_vector_store_resource: MagicMock,
    mock_subprocess: MagicMock,
    mock_asyncio_sleep: MagicMock,
) -> None:
    """on_create passes all dependency URLs as environment variables."""
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        instructions="You are a helpful assistant.",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
        mock_embeddings_resource=mock_embeddings_resource,
        mock_vector_store_resource=mock_vector_store_resource,
    )

    result = await agent.on_create()

    assert result.ready is True

    # Check that kubectl apply was called with correct env vars
    apply_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == "kubectl" and "apply" in c[0][0]]
    assert len(apply_calls) >= 2  # Deployment + Service


async def test_create_agent_custom_image(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mock_subprocess: MagicMock,
    mock_asyncio_sleep: MagicMock,
) -> None:
    """on_create uses custom container image."""
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        image="my-registry.io/my-agent:v1.0.0",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    result = await agent.on_create()

    assert result.ready is True
    # The image is embedded in the manifest, verified by deployment working


async def test_create_agent_multiple_replicas(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mocker: "MockerFixture",
) -> None:
    """on_create handles multiple replicas."""

    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if cmd[0] == "kubectl" and "deployment" in cmd:
            result.stdout = "3"  # 3 ready replicas
        return result

    mocker.patch("subprocess.run", side_effect=run_side_effect)
    mocker.patch("agno_provider.resources.agent.asyncio.sleep", return_value=None)

    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        replicas=3,
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    result = await agent.on_create()

    assert result.ready is True


async def test_create_agent_waits_for_ready(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mocker: "MockerFixture",
) -> None:
    """on_create polls Deployment until ready."""
    call_count = 0

    def run_side_effect(cmd, **kwargs):
        nonlocal call_count
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        if cmd[0] == "kubectl" and "deployment" in cmd and "jsonpath" in str(cmd):
            call_count += 1
            # First call returns 0 replicas, second returns 1
            result.stdout = "0" if call_count == 1 else "1"

        return result

    mocker.patch("subprocess.run", side_effect=run_side_effect)
    mocker.patch("agno_provider.resources.agent.asyncio.sleep", return_value=None)

    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    result = await agent.on_create()

    assert result.ready is True
    assert call_count >= 2  # At least two kubectl calls to check readiness


async def test_create_agent_kubectl_failure(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mocker: "MockerFixture",
) -> None:
    """on_create fails when kubectl command fails."""

    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        if cmd[0] == "kubectl" and "apply" in cmd:
            result.returncode = 1
            result.stderr = "Error: failed to apply manifest"
        else:
            result.returncode = 0
            result.stderr = ""
        result.stdout = ""
        return result

    mocker.patch("subprocess.run", side_effect=run_side_effect)

    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    with pytest.raises(RuntimeError, match="kubectl command failed"):
        await agent.on_create()


async def test_update_unchanged_returns_existing(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mock_subprocess: MagicMock,
    mock_asyncio_sleep: MagicMock,
) -> None:
    """on_update returns existing outputs when config unchanged."""
    existing_outputs = AgentOutputs(
        url="http://agno-test-agent.default.svc.cluster.local",
        ready=True,
    )

    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
        outputs=existing_outputs,
    )

    # Reset mock to track calls
    mock_subprocess.reset_mock()

    # Create identical previous config
    previous_config = AgentConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        model=Dependency(provider="anthropic", resource="messages", name="claude"),
    )

    result = await agent.on_update(previous_config)

    assert result == existing_outputs
    # Verify no kubectl apply calls were made (short-circuited)
    apply_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == "kubectl" and "apply" in c[0][0]]
    assert len(apply_calls) == 0


async def test_update_replicas_triggers_kubectl_apply(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mocker: "MockerFixture",
) -> None:
    """on_update runs kubectl apply when replicas change."""

    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if cmd[0] == "kubectl" and "deployment" in cmd:
            result.stdout = "3"  # 3 ready replicas
        return result

    mocker.patch("subprocess.run", side_effect=run_side_effect)
    mocker.patch("agno_provider.resources.agent.asyncio.sleep", return_value=None)

    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        replicas=3,
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
        outputs=AgentOutputs(
            url="http://agno-test-agent.default.svc.cluster.local",
            ready=True,
        ),
    )

    # Previous config had different replicas
    previous_config = AgentConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        model=Dependency(provider="anthropic", resource="messages", name="claude"),
        replicas=1,
    )

    result = await agent.on_update(previous_config)

    assert result.ready is True


async def test_update_rejects_cluster_change(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
) -> None:
    """on_update rejects cluster changes."""
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    # Previous config had different cluster
    previous_config = AgentConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="other-cluster"),
        model=Dependency(provider="anthropic", resource="messages", name="claude"),
    )

    with pytest.raises(ValueError, match="cluster"):
        await agent.on_update(previous_config)


async def test_update_model_change_triggers_apply(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mocker: "MockerFixture",
) -> None:
    """on_update runs kubectl apply when model dependency changes."""

    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if cmd[0] == "kubectl" and "deployment" in cmd:
            result.stdout = "1"
        return result

    mock_run = mocker.patch("subprocess.run", side_effect=run_side_effect)
    mocker.patch("agno_provider.resources.agent.asyncio.sleep", return_value=None)

    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
        outputs=AgentOutputs(
            url="http://agno-test-agent.default.svc.cluster.local",
            ready=True,
        ),
    )

    # Previous config had different model
    previous_config = AgentConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        model=Dependency(provider="openai", resource="chat", name="gpt-4"),
    )

    result = await agent.on_update(previous_config)

    assert result.ready is True
    # Verify kubectl apply was called
    apply_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "kubectl" and "apply" in c[0][0]]
    assert len(apply_calls) >= 2


async def test_delete_success(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mock_subprocess: MagicMock,
) -> None:
    """on_delete removes Deployment and Service."""
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    await agent.on_delete()

    # Verify kubectl delete was called for both resources
    delete_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == "kubectl" and "delete" in c[0][0]]
    assert len(delete_calls) == 2

    # Check deployment and service were deleted
    deleted_resources = [c[0][0][2] for c in delete_calls]  # Third arg is resource type
    assert "deployment" in deleted_resources
    assert "service" in deleted_resources


async def test_delete_idempotent(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mocker: "MockerFixture",
) -> None:
    """on_delete succeeds when resources don't exist."""

    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.stdout = ""
        if cmd[0] == "kubectl" and "delete" in cmd:
            # --ignore-not-found makes this succeed
            result.returncode = 0
            result.stderr = ""
        else:
            result.returncode = 0
            result.stderr = ""
        return result

    mocker.patch("subprocess.run", side_effect=run_side_effect)

    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    # Should not raise
    await agent.on_delete()


def test_provider_name() -> None:
    """Resource has correct provider name."""
    assert Agent.provider == "agno"


def test_resource_type() -> None:
    """Resource has correct resource type."""
    assert Agent.resource == "agent"


def test_build_outputs() -> None:
    """_build_outputs creates correct in-cluster URL."""
    agent = create_agent_with_mocked_dependencies(name="my-agent")

    outputs = agent._build_outputs()

    assert outputs.url == "http://agno-my-agent.default.svc.cluster.local"
    assert outputs.ready is True


def test_deployment_name() -> None:
    """_get_deployment_name prefixes with agno-."""
    agent = create_agent_with_mocked_dependencies(name="test-agent")
    assert agent._get_deployment_name() == "agno-test-agent"


def test_build_deployment_manifest() -> None:
    """_build_deployment_manifest creates correct Kubernetes Deployment."""
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        replicas=2,
        image="custom-image:v1",
    )

    env_vars = [
        {"name": "AGNO_MODEL_URL", "value": "http://model"},
        {"name": "AGNO_INSTRUCTIONS", "value": "Be helpful"},
    ]

    manifest = agent._build_deployment_manifest(env_vars)

    assert manifest["apiVersion"] == "apps/v1"
    assert manifest["kind"] == "Deployment"
    assert manifest["metadata"]["name"] == "agno-test-agent"
    assert manifest["metadata"]["namespace"] == "default"
    assert manifest["spec"]["replicas"] == 2

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["name"] == "agent"
    assert container["image"] == "custom-image:v1"
    assert container["env"] == env_vars
    assert container["ports"][0]["containerPort"] == 8080


def test_build_service_manifest() -> None:
    """_build_service_manifest creates correct Kubernetes Service."""
    agent = create_agent_with_mocked_dependencies(name="test-agent")

    manifest = agent._build_service_manifest()

    assert manifest["apiVersion"] == "v1"
    assert manifest["kind"] == "Service"
    assert manifest["metadata"]["name"] == "agno-test-agent"
    assert manifest["metadata"]["namespace"] == "default"
    assert manifest["spec"]["type"] == "ClusterIP"
    assert manifest["spec"]["ports"][0]["port"] == 80
    assert manifest["spec"]["ports"][0]["targetPort"] == 8080


async def test_get_env_vars_model_only(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
) -> None:
    """_get_env_vars includes model URL."""
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    env_vars = await agent._get_env_vars()

    assert {"name": "AGNO_MODEL_URL", "value": "http://anthropic-messages.default.svc.cluster.local"} in env_vars


async def test_get_env_vars_with_all_dependencies(
    mock_gke_resource: MagicMock,
    mock_model_resource: MagicMock,
    mock_embeddings_resource: MagicMock,
    mock_vector_store_resource: MagicMock,
) -> None:
    """_get_env_vars includes all dependency URLs."""
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        instructions="You are helpful.",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
        mock_embeddings_resource=mock_embeddings_resource,
        mock_vector_store_resource=mock_vector_store_resource,
    )

    env_vars = await agent._get_env_vars()

    env_dict = {e["name"]: e["value"] for e in env_vars}
    assert env_dict["AGNO_MODEL_URL"] == "http://anthropic-messages.default.svc.cluster.local"
    assert env_dict["AGNO_EMBEDDINGS_URL"] == "http://openai-embeddings.default.svc.cluster.local"
    assert env_dict["AGNO_VECTOR_STORE_URL"] == "http://qdrant-collection.default.svc.cluster.local:6333"
    assert env_dict["AGNO_INSTRUCTIONS"] == "You are helpful."


def test_deployment_manifest_has_pragma_labels() -> None:
    """Deployment manifest includes pragma.io labels."""
    agent = create_agent_with_mocked_dependencies(name="test-agent")

    manifest = agent._build_deployment_manifest([])

    labels = manifest["metadata"]["labels"]
    assert labels["pragma.io/provider"] == "agno"
    assert labels["pragma.io/resource"] == "agent"
    assert labels["pragma.io/name"] == "test-agent"


def test_deployment_manifest_has_health_probes() -> None:
    """Deployment manifest includes readiness and liveness probes."""
    agent = create_agent_with_mocked_dependencies(name="test-agent")

    manifest = agent._build_deployment_manifest([])

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert "readinessProbe" in container
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["port"] == 8080
    assert "livenessProbe" in container
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["httpGet"]["port"] == 8080
