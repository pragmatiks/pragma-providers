"""Qdrant Collection resource."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Literal, cast

from pragma_sdk import AlreadyExistsError, Config, Field, HealthStatus, ImmutableField, Outputs, Resource
from pydantic import BaseModel
from pydantic import Field as PydanticField
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from qdrant_provider.marker import MARKER_MISMATCH_REASONS, MarkerMatch, build_resource_marker, match_resource_marker


logger = logging.getLogger(__name__)

MINIMUM_SERVER_VERSION = (1, 16)

COLLECTION_STATUS_HEALTH: dict[models.CollectionStatus, Literal["healthy", "degraded", "unhealthy"]] = {
    models.CollectionStatus.GREEN: "healthy",
    models.CollectionStatus.YELLOW: "degraded",
    models.CollectionStatus.GREY: "degraded",
    models.CollectionStatus.RED: "unhealthy",
}


def reports_existing_collection(error: UnexpectedResponse) -> bool:
    """Tell whether Qdrant refused a create because the collection already exists.

    Qdrant answers 409 Conflict for an existing collection; older servers
    answer 400 Bad Request with a body saying it already exists.

    Args:
        error: The refusal Qdrant returned for a create request.

    Returns:
        True if the refusal reports an existing collection.
    """
    if error.status_code == HTTPStatus.CONFLICT:
        return True

    return error.status_code == HTTPStatus.BAD_REQUEST and b"already exists" in error.content


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
        - on_create: Create the marked collection; update its own earlier create, refuse any other
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
            url=cast(str, self.config.url),
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
        return distance_map[cast(VectorConfig, self.config.vectors).distance]

    def build_outputs(self) -> CollectionOutputs:
        """Build outputs for the configured collection.

        Returns:
            CollectionOutputs naming the collection.
        """
        return CollectionOutputs(name=cast(str, self.config.name))

    async def _create_collection(self, client: AsyncQdrantClient) -> None:
        """Create the collection with configured parameters."""
        await client.create_collection(
            collection_name=cast(str, self.config.name),
            vectors_config=models.VectorParams(
                size=cast(VectorConfig, self.config.vectors).size,
                distance=self._get_distance(),
                on_disk=cast(bool, self.config.on_disk),
            ),
            metadata=build_resource_marker(self.id),
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

        return (
            live_vectors.size != cast(VectorConfig, self.config.vectors).size
            or live_vectors.distance != self._get_distance()
        )

    async def check_server_version(self, client: AsyncQdrantClient) -> None:
        """Refuse a Qdrant server too old to store the collection metadata the create marks it with.

        Args:
            client: Client of the Qdrant server receiving the create.

        Raises:
            RuntimeError: If the server runs a Qdrant version before 1.16.
            UnexpectedResponse: If Qdrant fails the version read.
        """
        version = (await client.info()).version
        major, minor = (int(part) for part in version.split(".")[:2])

        if (major, minor) < MINIMUM_SERVER_VERSION:
            message = (
                f"The Qdrant server at {self.config.url} runs version {version}; collections need Qdrant 1.16 "
                "or later, which stores the collection metadata Pragmatiks recognizes its own collections by"
            )
            raise RuntimeError(message)

    async def read_marker_match(self, client: AsyncQdrantClient) -> MarkerMatch:
        """Read the existing collection and compare its resource marker with this resource.

        Args:
            client: Client of the Qdrant server holding the collection.

        Returns:
            How the collection's ``pragma_resource`` metadata compares with this resource;
            ``ABSENT`` when the collection is gone.

        Raises:
            UnexpectedResponse: If the read fails for any reason other than the collection being absent.
        """
        try:
            collection = await client.get_collection(cast(str, self.config.name))
        except UnexpectedResponse as error:
            if error.status_code != HTTPStatus.NOT_FOUND:
                raise

            return MarkerMatch.ABSENT

        return match_resource_marker(collection.config.metadata, self.id)

    async def on_create(self) -> CollectionOutputs:
        """Create the Qdrant collection, marked with this resource's ID in its ``pragma_resource`` metadata.

        A collection that already exists is refused, never adopted, unless its metadata
        names this resource: then it is this resource's earlier create and is updated in place.

        Returns:
            CollectionOutputs naming the collection.

        Raises:
            AlreadyExistsError: If a collection with the configured name that this resource
                did not create already exists.
            RuntimeError: If the server runs a Qdrant version before 1.16.
            UnexpectedResponse: If Qdrant refuses the create for any other reason.

        Note:
            Collection metadata needs Qdrant server 1.16 or later; an older server is refused
            before the create, since it would not keep the marker.
        """
        client = self._get_client()

        try:
            await self.check_server_version(client)
            await self._create_collection(client)
        except UnexpectedResponse as error:
            if not reports_existing_collection(error):
                raise

            match = await self.read_marker_match(client)

            if match is MarkerMatch.OWN:
                logger.warning("qdrant_create_conflict_recognized resource=%s collection=%s", self.id, self.config.name)
                return await self.on_update(None)

            logger.warning(
                "qdrant_create_conflict_refused resource=%s collection=%s reason=%s", self.id, self.config.name, match
            )
            message = (
                f"Collection {self.config.name!r} already exists and this resource did not create it: "
                f"{MARKER_MISMATCH_REASONS[match]}"
            )
            raise AlreadyExistsError(message) from error
        finally:
            await client.close()

        return self.build_outputs()

    async def on_observe(self) -> CollectionOutputs | None:
        """Look up the configured collection.

        Returns:
            CollectionOutputs naming the collection, or ``None`` if it does not exist.
        """
        client = self._get_client()

        try:
            if not await client.collection_exists(cast(str, self.config.name)):
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
            name = cast(str, self.config.name)
            if await client.collection_exists(name):
                await client.delete_collection(name)
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
