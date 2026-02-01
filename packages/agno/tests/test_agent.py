"""Tests for Agno Agent resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pragma_sdk import Dependency, LifecycleState

from agno_provider import (
    Agent,
    AgentConfig,
    AgentOutputs,
)


pytestmark = pytest.mark.skip(reason="AgentConfig forward references not fully defined (PRA-172)")


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_agent_with_mocked_dependencies(
    name: str,
    replicas: int = 1,
    image: str = "ghcr.io/agno-ai/agno:latest",
    instructions: str | None = None,
    mock_gke_resource: Any = None,
    mock_model_resource: Any = None,
    mock_embeddings_resource: Any = None,
    mock_vector_store_resource: Any = None,
    outputs: AgentOutputs | None = None,
) -> Agent:
    cluster_dep = Dependency(provider="gcp", resource="gke", name="test-cluster")
    model_dep = Dependency(provider="anthropic", resource="messages", name="claude")

    embeddings_dep = None
    if mock_embeddings_resource is not None:
        embeddings_dep = Dependency(provider="openai", resource="embeddings", name="embeddings")

    vector_store_dep = None
    if mock_vector_store_resource is not None:
        vector_store_dep = Dependency(provider="qdrant", resource="collection", name="my-collection")

    config = AgentConfig(
        cluster=cluster_dep,
        model=model_dep,
        embeddings=embeddings_dep,
        vector_store=vector_store_dep,
        instructions=instructions,
        image=image,
        replicas=replicas,
    )

    if mock_gke_resource:
        config.cluster._resolved = mock_gke_resource
    if mock_model_resource:
        config.model._resolved = mock_model_resource
    if mock_embeddings_resource and embeddings_dep:
        config.embeddings._resolved = mock_embeddings_resource
    if mock_vector_store_resource and vector_store_dep:
        config.vector_store._resolved = mock_vector_store_resource

    return Agent(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


async def test_create_agent_success(
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mock_subprocess: Any,
    mock_asyncio_sleep: Any,
) -> None:
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    result = await agent.on_create()

    assert result.url == "http://agno-test-agent.default.svc.cluster.local"
    assert result.ready is True


async def test_create_agent_with_all_dependencies(
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mock_embeddings_resource: Any,
    mock_vector_store_resource: Any,
    mock_subprocess: Any,
    mock_asyncio_sleep: Any,
) -> None:
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

    apply_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == "kubectl" and "apply" in c[0][0]]
    assert len(apply_calls) >= 2


async def test_create_agent_custom_image(
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mock_subprocess: Any,
    mock_asyncio_sleep: Any,
) -> None:
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        image="my-registry.io/my-agent:v1.0.0",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    result = await agent.on_create()

    assert result.ready is True


async def test_create_agent_multiple_replicas(
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mocker: MockerFixture,
) -> None:
    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if cmd[0] == "kubectl" and "deployment" in cmd:
            result.stdout = "3"
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
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mocker: MockerFixture,
) -> None:
    call_count = 0

    def run_side_effect(cmd, **kwargs):
        nonlocal call_count
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        if cmd[0] == "kubectl" and "deployment" in cmd and "jsonpath" in str(cmd):
            call_count += 1
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
    assert call_count >= 2


async def test_create_agent_kubectl_failure(
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mocker: MockerFixture,
) -> None:
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
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mock_subprocess: Any,
    mock_asyncio_sleep: Any,
) -> None:
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

    mock_subprocess.reset_mock()

    previous_config = AgentConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        model=Dependency(provider="anthropic", resource="messages", name="claude"),
    )

    result = await agent.on_update(previous_config)

    assert result == existing_outputs
    apply_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == "kubectl" and "apply" in c[0][0]]
    assert len(apply_calls) == 0


async def test_update_replicas_triggers_kubectl_apply(
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mocker: MockerFixture,
) -> None:
    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if cmd[0] == "kubectl" and "deployment" in cmd:
            result.stdout = "3"
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

    previous_config = AgentConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        model=Dependency(provider="anthropic", resource="messages", name="claude"),
        replicas=1,
    )

    result = await agent.on_update(previous_config)

    assert result.ready is True


async def test_update_rejects_cluster_change(
    mock_gke_resource: Any,
    mock_model_resource: Any,
) -> None:
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    previous_config = AgentConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="other-cluster"),
        model=Dependency(provider="anthropic", resource="messages", name="claude"),
    )

    with pytest.raises(ValueError, match="cluster"):
        await agent.on_update(previous_config)


async def test_update_model_change_triggers_apply(
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mocker: MockerFixture,
) -> None:
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

    previous_config = AgentConfig(
        cluster=Dependency(provider="gcp", resource="gke", name="test-cluster"),
        model=Dependency(provider="openai", resource="chat", name="gpt-4"),
    )

    result = await agent.on_update(previous_config)

    assert result.ready is True
    apply_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "kubectl" and "apply" in c[0][0]]
    assert len(apply_calls) >= 2


async def test_delete_success(
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mock_subprocess: Any,
) -> None:
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    await agent.on_delete()

    delete_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == "kubectl" and "delete" in c[0][0]]
    assert len(delete_calls) == 2

    deleted_resources = [c[0][0][2] for c in delete_calls]
    assert "deployment" in deleted_resources
    assert "service" in deleted_resources


async def test_delete_idempotent(
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mocker: MockerFixture,
) -> None:
    def run_side_effect(cmd, **kwargs):
        result = mocker.MagicMock()
        result.stdout = ""
        if cmd[0] == "kubectl" and "delete" in cmd:
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

    await agent.on_delete()


def test_provider_name() -> None:
    assert Agent.provider == "agno"


def test_resource_type() -> None:
    assert Agent.resource == "agent"


def test_build_outputs() -> None:
    agent = create_agent_with_mocked_dependencies(name="my-agent")

    outputs = agent._build_outputs()

    assert outputs.url == "http://agno-my-agent.default.svc.cluster.local"
    assert outputs.ready is True


def test_deployment_name() -> None:
    agent = create_agent_with_mocked_dependencies(name="test-agent")
    assert agent._get_deployment_name() == "agno-test-agent"


def test_build_deployment_manifest() -> None:
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
    mock_gke_resource: Any,
    mock_model_resource: Any,
) -> None:
    agent = create_agent_with_mocked_dependencies(
        name="test-agent",
        mock_gke_resource=mock_gke_resource,
        mock_model_resource=mock_model_resource,
    )

    env_vars = await agent._get_env_vars()

    assert {"name": "AGNO_MODEL_URL", "value": "http://anthropic-messages.default.svc.cluster.local"} in env_vars


async def test_get_env_vars_with_all_dependencies(
    mock_gke_resource: Any,
    mock_model_resource: Any,
    mock_embeddings_resource: Any,
    mock_vector_store_resource: Any,
) -> None:
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
    agent = create_agent_with_mocked_dependencies(name="test-agent")

    manifest = agent._build_deployment_manifest([])

    labels = manifest["metadata"]["labels"]
    assert labels["pragma.io/provider"] == "agno"
    assert labels["pragma.io/resource"] == "agent"
    assert labels["pragma.io/name"] == "test-agent"


def test_deployment_manifest_has_health_probes() -> None:
    agent = create_agent_with_mocked_dependencies(name="test-agent")

    manifest = agent._build_deployment_manifest([])

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert "readinessProbe" in container
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["port"] == 8080
    assert "livenessProbe" in container
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["httpGet"]["port"] == 8080
