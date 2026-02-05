"""Agno MemoryManager resource for agent memory management."""

from __future__ import annotations

from typing import ClassVar

from agno.memory.manager import MemoryManager as AgnoMemoryManager
from pragma_sdk import Config, Dependency, Outputs

from agno_provider.resources.base import AgnoResource, AgnoSpec
from agno_provider.resources.db.postgres import DbPostgres, DbPostgresSpec
from agno_provider.resources.models.anthropic import (
    AnthropicModelOutputs,
    AnthropicModelSpec,
)
from agno_provider.resources.models.base import Model, model_from_spec
from agno_provider.resources.models.openai import (
    OpenAIModelOutputs,
    OpenAIModelSpec,
)


class MemoryManagerSpec(AgnoSpec):
    """Specification for reconstructing MemoryManager at runtime.

    Contains all configuration needed to reconstruct an AgnoMemoryManager
    at runtime, with nested specs for dependencies.

    Attributes:
        db_spec: Nested database specification.
        model_spec: Nested model specification (OpenAI or Anthropic).
        system_message: System message for memory classification.
        memory_capture_instructions: Instructions for memory capture.
        additional_instructions: Additional instructions.
        add_memories: Whether to add new memories.
        update_memories: Whether to update existing memories.
        delete_memories: Whether to delete memories.
        clear_memories: Whether to clear all memories.
        debug_mode: Enable debug mode.
    """

    db_spec: DbPostgresSpec
    model_spec: OpenAIModelSpec | AnthropicModelSpec | None = None
    system_message: str | None = None
    memory_capture_instructions: str | None = None
    additional_instructions: str | None = None
    add_memories: bool = True
    update_memories: bool = True
    delete_memories: bool = False
    clear_memories: bool = False
    debug_mode: bool = False


class MemoryManagerConfig(Config):
    """Configuration for Agno MemoryManager.

    Attributes:
        db: Database dependency for memory storage.
        model: Optional model dependency for memory classification.
        system_message: Optional system message for memory classification.
        memory_capture_instructions: Optional instructions for memory capture.
        additional_instructions: Optional additional instructions.
        add_memories: Whether to add new memories. Defaults to True.
        update_memories: Whether to update existing memories. Defaults to True.
        delete_memories: Whether to delete memories. Defaults to False.
        clear_memories: Whether to clear all memories. Defaults to False.
        debug_mode: Enable debug mode. Defaults to False.
    """

    db: Dependency[DbPostgres]
    model: Dependency[Model] | None = None
    system_message: str | None = None
    memory_capture_instructions: str | None = None
    additional_instructions: str | None = None
    add_memories: bool = True
    update_memories: bool = True
    delete_memories: bool = False
    clear_memories: bool = False
    debug_mode: bool = False


class MemoryManagerOutputs(Outputs):
    """Outputs from Agno MemoryManager resource.

    Attributes:
        spec: The memory manager specification for runtime reconstruction.
    """

    spec: MemoryManagerSpec


class MemoryManager(AgnoResource[MemoryManagerConfig, MemoryManagerOutputs, MemoryManagerSpec]):
    """Agno MemoryManager resource for agent memory management.

    Wraps Agno's MemoryManager to provide memory storage and retrieval
    for agents. The MemoryManager instance is created via from_spec()
    at runtime.

    Runtime reconstruction from spec:
        manager = MemoryManager.from_spec(spec)
        agent = Agent(memory=manager, ...)

    Lifecycle:
        - on_create: Return serializable metadata (no actual setup)
        - on_update: Return serializable metadata
        - on_delete: No-op (stateless wrapper)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "memory/manager"

    @staticmethod
    def from_spec(spec: MemoryManagerSpec) -> AgnoMemoryManager:
        """Factory: construct Agno MemoryManager from spec.

        Constructs nested dependencies (db, model) from their specs
        and returns a configured MemoryManager instance.

        Args:
            spec: The memory manager specification.

        Returns:
            Configured AgnoMemoryManager instance.
        """
        db = DbPostgres.from_spec(spec.db_spec)
        model = model_from_spec(spec.model_spec) if spec.model_spec else None

        return AgnoMemoryManager(
            db=db,
            model=model,
            system_message=spec.system_message,
            memory_capture_instructions=spec.memory_capture_instructions,
            additional_instructions=spec.additional_instructions,
            add_memories=spec.add_memories,
            update_memories=spec.update_memories,
            delete_memories=spec.delete_memories,
            clear_memories=spec.clear_memories,
            debug_mode=spec.debug_mode,
        )

    async def _build_spec(self) -> MemoryManagerSpec:
        """Build spec from current config and resolved dependencies.

        Creates a specification that can be serialized and used to
        reconstruct the memory manager at runtime.

        Returns:
            MemoryManagerSpec with nested specs for dependencies.
        """
        db_resource = await self.config.db.resolve()
        db_outputs = db_resource.outputs
        assert db_outputs is not None
        db_spec = db_outputs.spec

        model_spec: OpenAIModelSpec | AnthropicModelSpec | None = None
        if self.config.model is not None:
            model_resource = await self.config.model.resolve()
            model_outputs = model_resource.outputs

            if isinstance(model_outputs, OpenAIModelOutputs):
                model_spec = model_outputs.spec
            elif isinstance(model_outputs, AnthropicModelOutputs):
                model_spec = model_outputs.spec

        return MemoryManagerSpec(
            db_spec=db_spec,
            model_spec=model_spec,
            system_message=self.config.system_message,
            memory_capture_instructions=self.config.memory_capture_instructions,
            additional_instructions=self.config.additional_instructions,
            add_memories=self.config.add_memories,
            update_memories=self.config.update_memories,
            delete_memories=self.config.delete_memories,
            clear_memories=self.config.clear_memories,
            debug_mode=self.config.debug_mode,
        )

    async def _build_outputs(self) -> MemoryManagerOutputs:
        """Build outputs from current config.

        Returns:
            MemoryManagerOutputs with spec.
        """
        return MemoryManagerOutputs(spec=await self._build_spec())

    async def on_create(self) -> MemoryManagerOutputs:
        """Create resource and return serializable outputs.

        Returns:
            MemoryManagerOutputs with configuration metadata.
        """
        return await self._build_outputs()

    async def on_update(self, previous_config: MemoryManagerConfig) -> MemoryManagerOutputs:  # noqa: ARG002
        """Update resource and return serializable outputs.

        Returns:
            MemoryManagerOutputs with updated configuration metadata.
        """
        return await self._build_outputs()

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
