"""Tests for Agno knowledge resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pragma_sdk import Dependency, LifecycleState
from pydantic import ValidationError

from agno_provider.resources.knowledge.embedder.openai import EmbedderOpenAI
from agno_provider.resources.knowledge.knowledge import (
    Knowledge,
    KnowledgeConfig,
    KnowledgeOutputs,
    KnowledgeSpec,
)
from agno_provider.resources.vectordb.qdrant import (
    VectordbQdrant,
    VectordbQdrantConfig,
    VectordbQdrantOutputs,
    VectordbQdrantSpec,
)


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_vectordb_resource(name: str = "test-vectordb") -> VectordbQdrant:
    """Create a VectordbQdrant resource for testing."""
    config = VectordbQdrantConfig(
        url="http://localhost:6333",
        collection="test-collection",
        search_type="vector",
    )

    return VectordbQdrant(
        name=name,
        config=config,
        lifecycle_state=LifecycleState.READY,
    )


def create_knowledge_config(
    vectordb: VectordbQdrant | None = None,
    max_results: int = 10,
) -> KnowledgeConfig:
    """Create KnowledgeConfig with resolved dependencies."""
    if vectordb is None:
        vectordb = create_vectordb_resource()

    vectordb_dep = Dependency[VectordbQdrant](
        provider="agno",
        resource="vectordb/qdrant",
        name=vectordb.name,
    )
    vectordb_dep._resolved = vectordb

    return KnowledgeConfig(
        vector_db=vectordb_dep,
        max_results=max_results,
    )


def create_knowledge(
    name: str = "test-knowledge",
    vectordb: VectordbQdrant | None = None,
    max_results: int = 10,
    outputs: KnowledgeOutputs | None = None,
) -> Knowledge:
    """Create a Knowledge resource for testing."""
    config = create_knowledge_config(
        vectordb=vectordb,
        max_results=max_results,
    )

    return Knowledge(
        name=name,
        config=config,
        outputs=outputs,
        lifecycle_state=LifecycleState.PROCESSING,
    )


def test_resource_metadata_resource_type() -> None:
    """Resource has correct resource type."""
    assert Knowledge.resource == "knowledge"


def test_outputs_contain_expected_fields() -> None:
    """Outputs contain pip_dependencies and spec."""
    outputs = KnowledgeOutputs(
        pip_dependencies=[],
        spec=KnowledgeSpec(
            name="test-knowledge",
            vector_db_spec=VectordbQdrantSpec(
                url="http://localhost:6333",
                collection="test-collection",
            ),
        ),
    )

    assert outputs.pip_dependencies == []
    assert outputs.spec.name == "test-knowledge"


def test_outputs_are_serializable() -> None:
    """Outputs can be serialized to JSON."""
    outputs = KnowledgeOutputs(
        pip_dependencies=["qdrant-client"],
        spec=KnowledgeSpec(
            name="test-knowledge",
            vector_db_spec=VectordbQdrantSpec(
                url="http://localhost:6333",
                collection="test-collection",
            ),
        ),
    )

    serialized = outputs.model_dump_json()

    assert "pip_dependencies" in serialized
    assert "spec" in serialized


def test_get_pip_dependencies_from_vectordb() -> None:
    """_get_pip_dependencies includes vectordb dependencies."""
    vectordb = create_vectordb_resource()
    vectordb.outputs = VectordbQdrantOutputs(
        spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
        ),
        pip_dependencies=["qdrant-client"],
    )
    resource = create_knowledge(vectordb=vectordb)

    deps = resource._get_pip_dependencies()

    assert "qdrant-client" in deps


def test_from_spec_returns_agno_knowledge(mocker: MockerFixture) -> None:
    """from_spec() returns configured AgnoKnowledge instance."""
    mocker.patch("agno.vectordb.qdrant.Qdrant.__init__", return_value=None)
    mock_knowledge_init = mocker.patch(
        "agno.knowledge.knowledge.Knowledge.__init__",
        return_value=None,
    )

    spec = KnowledgeSpec(
        name="my-knowledge",
        max_results=5,
        vector_db_spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
        ),
    )

    Knowledge.from_spec(spec)

    mock_knowledge_init.assert_called_once()
    call_kwargs = mock_knowledge_init.call_args.kwargs
    assert call_kwargs["name"] == "my-knowledge"
    assert call_kwargs["max_results"] == 5


def test_build_spec_contains_knowledge_config() -> None:
    """_build_spec() returns spec with knowledge configuration."""
    vectordb = create_vectordb_resource(name="vectordb")
    vectordb.outputs = VectordbQdrantOutputs(
        spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
            search_type="vector",
        ),
        pip_dependencies=[],
    )
    resource = create_knowledge(name="kb", vectordb=vectordb, max_results=5)

    spec = resource._build_spec()

    assert spec.name == "kb"
    assert spec.max_results == 5
    assert spec.vector_db_spec is not None
    assert spec.vector_db_spec.collection == "test-collection"


def test_build_spec_includes_vectordb_spec() -> None:
    """_build_spec() includes nested vectordb spec."""
    vectordb = create_vectordb_resource(name="vectordb")
    vectordb.outputs = VectordbQdrantOutputs(
        spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
            search_type="vector",
        ),
        pip_dependencies=[],
    )
    resource = create_knowledge(name="kb", vectordb=vectordb)

    spec = resource._build_spec()

    assert spec.vector_db_spec is not None
    assert spec.vector_db_spec.url == "http://localhost:6333"
    assert spec.vector_db_spec.collection == "test-collection"


async def test_lifecycle_on_create_returns_outputs(mocker: MockerFixture) -> None:
    """on_create returns outputs."""
    vectordb = create_vectordb_resource(name="vectordb")
    vectordb.outputs = VectordbQdrantOutputs(
        spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
            search_type="vector",
        ),
        pip_dependencies=[],
    )
    resource = create_knowledge(vectordb=vectordb)

    mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    result = await resource.on_create()

    assert isinstance(result, KnowledgeOutputs)


async def test_lifecycle_on_create_resolves_dependencies(mocker: MockerFixture) -> None:
    """on_create calls resolve() on dependencies."""
    vectordb = create_vectordb_resource(name="vectordb")
    vectordb.outputs = VectordbQdrantOutputs(
        spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
            search_type="vector",
        ),
        pip_dependencies=[],
    )
    resource = create_knowledge(vectordb=vectordb)

    resolve_mock = mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    await resource.on_create()

    assert resolve_mock.call_count == 1


async def test_lifecycle_on_update_returns_outputs(mocker: MockerFixture) -> None:
    """on_update returns updated outputs."""
    vectordb = create_vectordb_resource(name="vectordb")
    vectordb.outputs = VectordbQdrantOutputs(
        spec=VectordbQdrantSpec(
            url="http://localhost:6333",
            collection="test-collection",
            search_type="vector",
        ),
        pip_dependencies=[],
    )
    resource = create_knowledge(vectordb=vectordb)

    mocker.patch("pragma_sdk.Dependency.resolve", return_value=None)

    vectordb_dep = Dependency[VectordbQdrant](
        provider="agno",
        resource="vectordb/qdrant",
        name="test-vectordb",
    )
    previous_config = KnowledgeConfig(
        vector_db=vectordb_dep,
        max_results=5,
    )

    result = await resource.on_update(previous_config)

    assert isinstance(result, KnowledgeOutputs)


async def test_lifecycle_on_delete_is_noop() -> None:
    """on_delete completes without error (stateless resource)."""
    resource = create_knowledge()
    await resource.on_delete()


def test_config_requires_vector_db() -> None:
    """Config requires vector_db dependency."""
    with pytest.raises(ValidationError, match="vector_db"):
        KnowledgeConfig()


def test_config_accepts_embedder_dependency() -> None:
    """Config accepts optional embedder dependency."""
    vectordb_dep = Dependency[VectordbQdrant](
        provider="agno",
        resource="vectordb/qdrant",
        name="test-vectordb",
    )
    embedder_dep = Dependency[EmbedderOpenAI](
        provider="agno",
        resource="knowledge/embedder/openai",
        name="test-embedder",
    )

    config = KnowledgeConfig(
        vector_db=vectordb_dep,
        embedder=embedder_dep,
        max_results=20,
    )

    assert config.embedder is not None
    assert config.embedder.name == "test-embedder"
    assert config.max_results == 20


def test_config_default_max_results() -> None:
    """Config defaults max_results to 10."""
    vectordb_dep = Dependency[VectordbQdrant](
        provider="agno",
        resource="vectordb/qdrant",
        name="test-vectordb",
    )

    config = KnowledgeConfig(
        vector_db=vectordb_dep,
    )

    assert config.max_results == 10
