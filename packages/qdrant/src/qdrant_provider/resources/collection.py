"""Qdrant Collection resource."""

from __future__ import annotations

from http import HTTPStatus
from typing import Literal, cast

from pragma_sdk import Config, Field, HealthStatus, ImmutableField, Outputs, Resource
from pydantic import BaseModel
from pydantic import Field as PydanticField
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse


COLLECTION_STATUS_HEALTH: dict[models.CollectionStatus, Literal["healthy", "degraded", "unhealthy"]] = {
    models.CollectionStatus.GREEN: "healthy",
    models.CollectionStatus.YELLOW: "degraded",
    models.CollectionStatus.GREY: "degraded",
    models.CollectionStatus.RED: "unhealthy",
}


class VectorConfig(BaseModel):
    """Vector configuration for a Qdrant collection.

    Attributes:
        size: Vector dimension (must match your embedding model's output).
        distance: Distance metric for similarity search.
    """

    size: int = PydanticField(gt=0)
    distance: Literal["Cosine", "Euclid", "Dot"] = "Cosine"


class CollectionConfig(Config):
    """Configuration for a Qdrant collection.

    Attributes:
        api_key: Qdrant Cloud API key. Use a FieldReference to inject from pragma/secret.
            Optional for local Qdrant instances.
        url: Qdrant server URL. Immutable.
        name: Collection name within Qdrant.
        vectors: Vector configuration including dimension and distance metric. Immutable.
        on_disk: Store vectors on disk instead of memory for larger datasets.
    """

    api_key: Field[str] | None = None
    url: ImmutableField[str] = "http://localhost:6333"
    name: ImmutableField[str]
    vectors: ImmutableField[VectorConfig]
    on_disk: Field[bool] = False


class CollectionOutputs(Outputs):
    """Outputs from Qdrant collection operations.

    Attributes:
        name: Collection name.
    """

    name: str


class Collection(Resource[CollectionConfig, CollectionOutputs]):
    """Qdrant collection resource.

    Manages Qdrant vector collections for similarity search. Supports both
    Qdrant Cloud (with API key) and local instances.

    Lifecycle:
        - on_create: Create collection if not exists
        - on_observe: Look up the collection by name
        - on_update: Apply on_disk in place
        - on_delete: Delete collection
    """

    def _get_client(self) -> AsyncQdrantClient:
        """Get Qdrant async client with configured credentials.

        Returns:
            Configured AsyncQdrantClient instance.
        """
        api_key = cast(str, self.config.api_key) if self.config.api_key else None

        return AsyncQdrantClient(
            url=self.config.url,
            api_key=api_key,
        )

    def _get_distance(self) -> models.Distance:
        """Map distance string to Qdrant Distance enum.

        Returns:
            Qdrant Distance enum value.
        """
        distance_map = {
            "Cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }
        return distance_map[self.config.vectors.distance]

    def build_outputs(self) -> CollectionOutputs:
        """Build outputs for the configured collection.

        Returns:
            CollectionOutputs naming the collection.
        """
        return CollectionOutputs(name=self.config.name)

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

    def vector_config_changed(self, live_vectors: models.VectorsConfig | None) -> bool:
        """Check whether the live vector size or distance differs from config.

        Args:
            live_vectors: Vector configuration read from the existing collection.

        Returns:
            True if the live collection does not hold one unnamed vector of the
            configured size and distance.
        """
        if not isinstance(live_vectors, models.VectorParams):
            return True

        return live_vectors.size != self.config.vectors.size or live_vectors.distance != self._get_distance()

    async def on_create(self) -> CollectionOutputs:
        """Create Qdrant collection if it doesn't exist.

        Idempotent: If collection already exists, leaves it as is.

        Returns:
            CollectionOutputs naming the collection.
        """
        client = self._get_client()

        try:
            exists = await client.collection_exists(self.config.name)

            if not exists:
                await self._create_collection(client)

            return self.build_outputs()
        finally:
            await client.close()

    async def on_observe(self) -> CollectionOutputs | None:
        """Look up the configured collection.

        Returns:
            CollectionOutputs naming the collection, or ``None`` if it does not exist.
        """
        client = self._get_client()

        try:
            if not await client.collection_exists(self.config.name):
                return None

            return self.build_outputs()
        finally:
            await client.close()

    async def on_update(self, previous_config: CollectionConfig | None) -> CollectionOutputs:
        """Apply the configured on_disk setting to the live collection in place.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            CollectionOutputs naming the collection.

        Raises:
            RuntimeError: If the live vector size or distance differs from config.
        """
        name = cast(str, self.config.name)
        vectors = cast(VectorConfig, self.config.vectors)
        on_disk = cast(bool, self.config.on_disk)
        client = self._get_client()

        try:
            live_vectors = (await client.get_collection(name)).config.params.vectors

            if self.vector_config_changed(live_vectors):
                msg = (
                    f"Collection {name!r} holds vectors other than the configured size {vectors.size} "
                    f"and distance {vectors.distance}; vectors are immutable, so delete and recreate the resource"
                )
                raise RuntimeError(msg)

            if bool(cast(models.VectorParams, live_vectors).on_disk) != on_disk:
                await client.update_collection(
                    collection_name=name,
                    vectors_config={"": models.VectorParamsDiff(on_disk=on_disk)},
                )

            return self.build_outputs()
        finally:
            await client.close()

    async def on_delete(self) -> None:
        """Delete collection and all its vectors.

        Idempotent: Succeeds if collection doesn't exist.
        """
        client = self._get_client()

        try:
            exists = await client.collection_exists(self.config.name)
            if exists:
                await client.delete_collection(self.config.name)
        finally:
            await client.close()

    async def health(self) -> HealthStatus:
        """Report the live collection's status from Qdrant.

        Green is healthy, yellow and grey are degraded, red is unhealthy. The
        point, indexed vector and segment counts go in the details.

        Returns:
            HealthStatus of the collection, unhealthy if it does not exist.

        Raises:
            UnexpectedResponse: If Qdrant fails the lookup for a reason other than a missing collection.
        """
        name = cast(str, self.config.name)
        client = self._get_client()

        try:
            info = await client.get_collection(name)
        except UnexpectedResponse as error:
            if error.status_code != HTTPStatus.NOT_FOUND:
                raise

            return HealthStatus(status="unhealthy", message=f"Collection {name!r} not found")
        finally:
            await client.close()

        return HealthStatus(
            status=COLLECTION_STATUS_HEALTH[info.status],
            message=f"Collection {name!r} is {info.status.value}",
            details={
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "segments_count": info.segments_count,
            },
        )
