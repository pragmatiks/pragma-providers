"""Agno MemoryManager resource for agent memory management."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from agno.memory.manager import MemoryManager as AgnoMemoryManager
from pragma_sdk import Config, Dependency, Outputs, Resource


if TYPE_CHECKING:
    from agno_provider.resources.db.postgres import DbPostgres
    from agno_provider.resources.models.base import Model


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
        ready: Whether the memory manager is configured.
        db_schema: The configured database schema.
        has_model: Whether a model is configured for memory classification.
    """

    ready: bool
    db_schema: str
    has_model: bool


class MemoryManager(Resource[MemoryManagerConfig, MemoryManagerOutputs]):
    """Agno MemoryManager resource for agent memory management.

    Wraps Agno's MemoryManager to provide memory storage and retrieval
    for agents. Dependent resources receive this resource and call
    memory_manager() to get the configured MemoryManager instance.

    Usage by dependent resources:
        memory: Dependency[MemoryManager]

        async def on_create(self):
            manager = await self.memory.resolve()
            mm = await manager.memory_manager()
            agent = Agent(memory=mm, ...)

    Lifecycle:
        - on_create: Return serializable metadata (no actual setup)
        - on_update: Return serializable metadata
        - on_delete: No-op (stateless wrapper)
    """

    provider: ClassVar[str] = "agno"
    resource: ClassVar[str] = "memory/manager"

    async def memory_manager(self) -> AgnoMemoryManager:
        """Return configured AgnoMemoryManager instance.

        Called by dependent resources (e.g., agno/agent) that need
        the memory manager instance.

        Returns:
            Configured AgnoMemoryManager ready for use.
        """
        db_resource = await self.config.db.resolve()
        db = db_resource.db()

        model = None
        if self.config.model is not None:
            model_resource = await self.config.model.resolve()
            model = model_resource.model()

        return AgnoMemoryManager(
            db=db,
            model=model,
            system_message=self.config.system_message,
            memory_capture_instructions=self.config.memory_capture_instructions,
            additional_instructions=self.config.additional_instructions,
            add_memories=self.config.add_memories,
            update_memories=self.config.update_memories,
            delete_memories=self.config.delete_memories,
            clear_memories=self.config.clear_memories,
            debug_mode=self.config.debug_mode,
        )

    async def on_create(self) -> MemoryManagerOutputs:
        """Create resource and return serializable outputs.

        Returns:
            MemoryManagerOutputs with configuration metadata.
        """
        db_resource = await self.config.db.resolve()

        return MemoryManagerOutputs(
            ready=True,
            db_schema=db_resource.config.db_schema,
            has_model=self.config.model is not None,
        )

    async def on_update(self, previous_config: MemoryManagerConfig) -> MemoryManagerOutputs:  # noqa: ARG002
        """Update resource and return serializable outputs.

        Returns:
            MemoryManagerOutputs with updated configuration metadata.
        """
        db_resource = await self.config.db.resolve()

        return MemoryManagerOutputs(
            ready=True,
            db_schema=db_resource.config.db_schema,
            has_model=self.config.model is not None,
        )

    async def on_delete(self) -> None:
        """Delete is a no-op since this resource is stateless."""
