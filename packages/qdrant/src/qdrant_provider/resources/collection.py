"""Qdrant Collection resource."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel
from pragma_sdk import Config, Field, Outputs, Resource
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models


class VectorConfig(BaseModel):
    """Vector configuration for a Qdrant collection.

    Attributes:
        size: Vector dimension (must match your embedding model's output).
        distance: Distance metric for similarity search.
    """

    size: int
    distance: Literal["Cosine", "Euclid", "Dot"] = "Cosine"


class CollectionConfig(Config):
    """Configuration for a Qdrant collection.

    Attributes:
        api_key: Qdrant Cloud API key. Use a FieldReference to inject from pragma/secret.
            Optional for local Qdrant instances.
        url: Qdrant server URL.
        name: Collection name within Qdrant.
        vectors: Vector configuration including dimension and distance metric.
        on_disk: Store vectors on disk instead of memory for larger datasets.
    """

    api_key: Field[str] | None = None
    url: str = "http://localhost:6333"
    name: str
    vectors: VectorConfig
    on_disk: bool = False


class CollectionOutputs(Outputs):
    """Outputs from Qdrant collection operations.

    Attributes:
        name: Collection name.
        vectors_count: Number of indexed vectors in the collection.
        points_count: Total number of points in the collection.
        status: Collection status (green, yellow, red).
    """

    name: str
    vectors_count: int
    points_count: int
    status: str


class Collection(Resource[CollectionConfig, CollectionOutputs]):
    """Qdrant collection resource.

    Manages Qdrant vector collections for similarity search. Supports both
    Qdrant Cloud (with API key) and local instances.

    Lifecycle:
        - on_create: Create collection if not exists
        - on_update: Recreate if vector config changes (destructive)
        - on_delete: Delete collection
    """

    provider: ClassVar[str] = "qdrant"
    resource: ClassVar[str] = "collection"

    def _get_client(self) -> AsyncQdrantClient:
        """Get Qdrant async client with configured credentials."""
        return AsyncQdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
        )

    def _get_distance(self) -> models.Distance:
        """Map distance string to Qdrant Distance enum."""
        distance_map = {
            "Cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }
        return distance_map[self.config.vectors.distance]

    async def _get_collection_info(self, client: AsyncQdrantClient) -> CollectionOutputs:
        """Fetch collection info and build outputs."""
        info = await client.get_collection(self.config.name)
        return CollectionOutputs(
            name=self.config.name,
            vectors_count=info.vectors_count or 0,
            points_count=info.points_count or 0,
            status=info.status.value if info.status else "unknown",
        )

    async def _create_collection(self, client: AsyncQdrantClient) -> None:
        """Create the collection with configured parameters."""
        await client.create_collection(
            collection_name=self.config.name,
            vectors_config=models.VectorParams(
                size=self.config.vectors.size,
                distance=self._get_distance(),
                on_disk=self.config.on_disk,
            ),
        )

    def _vector_config_changed(self, previous_config: CollectionConfig) -> bool:
        """Check if vector configuration changed (requires recreation)."""
        return (
            previous_config.vectors.size != self.config.vectors.size
            or previous_config.vectors.distance != self.config.vectors.distance
            or previous_config.on_disk != self.config.on_disk
        )

    async def on_create(self) -> CollectionOutputs:
        """Create Qdrant collection if it doesn't exist.

        Idempotent: If collection already exists, returns its current info.

        Returns:
            CollectionOutputs with collection metadata.
        """
        async with self._get_client() as client:
            exists = await client.collection_exists(self.config.name)

            if not exists:
                await self._create_collection(client)

            return await self._get_collection_info(client)

    async def on_update(self, previous_config: CollectionConfig) -> CollectionOutputs:
        """Update collection by recreating if vector config changed.

        Vector configuration changes (size, distance, on_disk) require
        deleting and recreating the collection. This is destructive and
        will lose all existing vectors.

        Args:
            previous_config: The previous configuration before update.

        Returns:
            CollectionOutputs with updated collection metadata.

        Raises:
            ValueError: If collection name changed (requires delete + create).
        """
        if previous_config.name != self.config.name:
            msg = "Cannot change collection name; delete and recreate resource"
            raise ValueError(msg)

        async with self._get_client() as client:
            if self._vector_config_changed(previous_config):
                # Destructive recreation - vector config changed
                exists = await client.collection_exists(self.config.name)
                if exists:
                    await client.delete_collection(self.config.name)
                await self._create_collection(client)

            return await self._get_collection_info(client)

    async def on_delete(self) -> None:
        """Delete collection and all its vectors.

        Idempotent: Succeeds if collection doesn't exist.
        """
        async with self._get_client() as client:
            exists = await client.collection_exists(self.config.name)
            if exists:
                await client.delete_collection(self.config.name)
