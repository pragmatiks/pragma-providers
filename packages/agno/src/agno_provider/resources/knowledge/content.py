"""Content resource for Agno knowledge sources.

Represents a single content source (URL or text) written into the Qdrant collection
of a Knowledge base.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID, uuid5

import httpx
from agno.knowledge.document import Document
from agno.knowledge.reader.base import Reader
from agno.knowledge.reader.reader_factory import ReaderFactory
from agno.utils import log as agno_log
from pragma_sdk import Config, Field, FieldReference, ImmutableDependency, Outputs
from pydantic import model_validator
from qdrant_client import AsyncQdrantClient, models

from agno_provider.resources.base import AgnoResource, AgnoSpec
from agno_provider.resources.knowledge.knowledge import Knowledge, KnowledgeSpec
from agno_provider.resources.vectordb.qdrant import VectordbQdrant, VectordbQdrantSpec


SUPPORTED_URL_EXTENSIONS = frozenset({".json", ".markdown", ".md", ".text", ".txt"})
DOCUMENT_ID_NAMESPACE = UUID("2d91e743-3e56-4e5a-93a3-a903601a02d0")
CHUNK_INDEX_KEY = "chunk_index"
DOWNLOAD_BYTE_LIMIT = 100 * 1024 * 1024
UPSERT_BATCH_SIZE = 64
HTML_MEDIA_TYPES = frozenset({"application/xhtml+xml", "text/html"})
LOGGED_MESSAGE_LIMIT = 500

logger = logging.getLogger(__name__)


class ContentSpec(AgnoSpec):
    """Specification of a content source written into a Knowledge base.

    Attributes:
        name: Content name stored on every point.
        knowledge_spec: Nested spec for the knowledge base configuration.
        url: URL of a plain text, Markdown or JSON file.
        text_content: Raw inline text content.
        description: Content description.
        metadata: Custom metadata key-value pairs stored on every point.
        topics: Topic tags for categorization.
    """

    name: str
    knowledge_spec: KnowledgeSpec
    url: str | None = None
    text_content: str | None = None
    description: str | None = None
    metadata: dict[str, str] | None = None
    topics: list[str] | None = None


class ContentConfig(Config):
    """Configuration for a knowledge content source.

    Supports two mutually exclusive content types:
    1. URL: a plain text, Markdown or JSON file, recognized by a path ending in
       .txt, .text, .md, .markdown or .json. Websites, PDF, Office, CSV and
       YouTube URLs are refused.
    2. Text: Raw inline text content

    Attributes:
        knowledge: Knowledge resource whose Qdrant collection receives the content. Immutable.
        url: URL of a plain text, Markdown or JSON file.
        text_content: Raw inline text content.
        name: Content name for identification. Defaults to the resource name.
        description: Content description.
        metadata: Custom metadata key-value pairs.
        topics: Topic tags for categorization.
    """

    knowledge: ImmutableDependency[Knowledge]
    url: Field[str] | None = None
    text_content: Field[str] | None = None
    name: Field[str] | None = None
    description: Field[str] | None = None
    metadata: Field[dict[str, str]] | None = None
    topics: Field[list[str]] | None = None

    @model_validator(mode="after")
    def validate_content_source(self) -> ContentConfig:
        """Validate that exactly one of url or text_content is provided.

        An unresolved reference counts as a provided url.

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If neither or both url and text_content are provided.
        """
        has_url = self.url is not None and (isinstance(self.url, FieldReference) or len(self.url) > 0)
        has_text = self.text_content is not None

        if has_url and has_text:
            message = "Exactly one of url or text_content must be provided, not both"
            raise ValueError(message)

        if not has_url and not has_text:
            message = "Either url or text_content must be provided"
            raise ValueError(message)

        return self

    @model_validator(mode="after")
    def validate_url_extension(self) -> ContentConfig:
        """Validate that a literal url names a plain text, Markdown or JSON file.

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If the url's path does not end in a supported extension.
        """
        if isinstance(self.url, str) and self.url and parse_url_extension(self.url) not in SUPPORTED_URL_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_URL_EXTENSIONS))
            message = (
                f"url {build_display_url(self.url)!r} is not a supported content source; content reads "
                "inline text_content or a plain text, Markdown or JSON file whose url path ends in one of: "
                f"{supported}"
            )
            raise ValueError(message)

        return self


class ContentOutputs(Outputs):
    """Outputs from Content resource.

    Attributes:
        spec: The content specification.
        content_hash: Content hash of the spec the stored points were written from,
            or None if the points disagree.
    """

    spec: ContentSpec
    content_hash: str | None


active_warning_collector: ContextVar[LogMessageCollector | None] = ContextVar("active_warning_collector", default=None)
"""Collector of the :func:`collect_agno_warnings` block the current context runs in, or ``None``."""


class LogMessageCollector(logging.Handler):
    """Logging handler that keeps the warnings and errors logged from the context of its own block.

    Attributes:
        messages: The collected messages, in emission order.
    """

    def __init__(self) -> None:
        """Create a collector for records at warning level and above."""
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record's message when it was logged from this collector's block.

        Args:
            record: The log record Agno emitted.
        """
        if active_warning_collector.get() is self:
            self.messages.append(record.getMessage())


@contextmanager
def collect_agno_warnings() -> Iterator[list[str]]:
    """Collect the warnings and errors Agno logs inside the block.

    Agno readers log a read error and return no documents, and Agno embedders log an
    embedding error and return an empty embedding, instead of raising.
    Only records logged from the block's own context are collected, so concurrent
    work of other resources in the process is left out.

    Note:
        Agno embedders and the YouTube reader log from ``loop.run_in_executor(None, ...)``
        threads; their records are attributed to the block because the worker's default
        executor (``pragma_sdk.worker.executor.RequestScopedExecutor``) runs each piece of
        work in a copy of the submitting context. Under an executor that does not copy the
        context, those records are not collected.

    Yields:
        The list the messages are appended to while the block runs.
    """
    agno_logger = agno_log.logger
    collector = LogMessageCollector()
    token = active_warning_collector.set(collector)
    agno_logger.addHandler(collector)

    try:
        yield collector.messages
    finally:
        agno_logger.removeHandler(collector)
        active_warning_collector.reset(token)


def format_agno_warnings(messages: list[str]) -> str:
    """Format a bounded clause counting the warnings and errors Agno logged, quoting none of them.

    Args:
        messages: The collected messages.

    Returns:
        A short clause naming how many messages Agno logged.
    """
    count = len(messages)
    plural = "" if count == 1 else "s"
    return f"Agno logged {count} warning{plural} or error{plural}"


def format_first_message(messages: list[str]) -> str:
    """Format the first collected message cut to ``LOGGED_MESSAGE_LIMIT`` characters.

    Args:
        messages: The collected messages.

    Returns:
        The first message, truncated with an ellipsis when longer than the limit, or an empty string.
    """
    first = messages[0] if messages else ""
    return first if len(first) <= LOGGED_MESSAGE_LIMIT else f"{first[:LOGGED_MESSAGE_LIMIT]}..."


def parse_url_extension(url: str) -> str:
    """Parse the lowercase file extension of a URL's path.

    Args:
        url: Remote file URL.

    Returns:
        The extension with its leading dot, or an empty string when the path has none.
    """
    return Path(urlparse(url).path).suffix.lower()


def build_display_url(url: str) -> str:
    """Build the form of a URL safe to show in errors.

    Args:
        url: Remote file URL.

    Returns:
        The URL without credentials, query string or fragment.
    """
    parsed = urlparse(url)
    return parsed._replace(netloc=parsed.netloc.rpartition("@")[2], query="", fragment="").geturl()


def compute_content_hash(spec: ContentSpec) -> str:
    """Compute a stable hash of the content a spec describes.

    Args:
        spec: The content specification.

    Returns:
        Hex SHA-256 of the canonical JSON of the spec's content fields.
    """
    content_fields = spec.model_dump(include={"name", "text_content", "url", "description", "metadata", "topics"})
    return hashlib.sha256(json.dumps(content_fields, sort_keys=True).encode()).hexdigest()


def build_document_id(resource_id: str, chunk_index: int) -> str:
    """Build the deterministic Agno document id of one content chunk.

    The id depends only on the resource id and the chunk position, so rewriting
    unchanged content yields the same ids.

    Args:
        resource_id: Canonical id of the content resource.
        chunk_index: Position of the chunk in the read documents.

    Returns:
        UUID string, stable per chunk position.
    """
    return str(uuid5(DOCUMENT_ID_NAMESPACE, f"{resource_id}:{chunk_index}"))


def build_qdrant_client(vector_database_spec: VectordbQdrantSpec) -> AsyncQdrantClient:
    """Build a client of the Qdrant server a vector database spec names.

    Args:
        vector_database_spec: Spec of the knowledge's vector database.

    Returns:
        Client the caller must close.
    """
    return AsyncQdrantClient(url=vector_database_spec.url, api_key=vector_database_spec.api_key)


def build_resource_condition(resource_id: str) -> models.FieldCondition:
    """Build the Qdrant condition matching every point a content resource owns.

    Args:
        resource_id: Canonical id of the content resource.

    Returns:
        Condition on Agno's ``content_id`` payload key.
    """
    return models.FieldCondition(key="content_id", match=models.MatchValue(value=resource_id))


def build_stale_points_selector(resource_id: str, content_hash: str, chunk_count: int) -> models.FilterSelector:
    """Build the selector of a resource's points that a write did not overwrite.

    Args:
        resource_id: Canonical id of the content resource.
        content_hash: Content hash the write stored on its points.
        chunk_count: Number of chunks the write stored.

    Returns:
        Selector of the resource's points with another content hash or a chunk
        index at or beyond ``chunk_count``.
    """
    return models.FilterSelector(
        filter=models.Filter(
            must=[build_resource_condition(resource_id)],
            must_not=[
                models.Filter(
                    must=[
                        models.FieldCondition(key="content_hash", match=models.MatchValue(value=content_hash)),
                        models.FieldCondition(key=f"meta_data.{CHUNK_INDEX_KEY}", range=models.Range(lt=chunk_count)),
                    ]
                )
            ],
        )
    )


def assign_chunk_payload(documents: list[Document], spec: ContentSpec, resource_id: str) -> None:
    """Assign each chunk its deterministic id, owning resource and metadata.

    Args:
        documents: The chunks, in read order; updated in place.
        spec: The content specification whose metadata every chunk carries.
        resource_id: Canonical id of the content resource.
    """
    for chunk_index, document in enumerate(documents):
        document.id = build_document_id(resource_id, chunk_index)
        document.content_id = resource_id
        document.meta_data.update({
            **(spec.metadata or {}),
            "linked_to": spec.knowledge_spec.name,
            CHUNK_INDEX_KEY: chunk_index,
        })


def build_points(documents: list[Document], content_hash: str) -> list[models.PointStruct]:
    """Build the Qdrant points of embedded chunks with Agno's point ids and payload keys.

    Args:
        documents: The chunks, each carrying its document id and embedding.
        content_hash: Content hash stored on every point.

    Returns:
        One point per chunk, readable by Agno's Qdrant knowledge.
    """
    points = []

    for document in documents:
        content = document.content.replace("\x00", "\ufffd")
        point_id = hashlib.md5(f"{document.id}_{content_hash}".encode(), usedforsecurity=False).hexdigest()
        payload = {
            "name": document.name,
            "meta_data": document.meta_data,
            "content": content,
            "usage": document.usage,
            "content_id": document.content_id,
            "content_hash": content_hash,
        }
        points.append(models.PointStruct(id=point_id, vector=document.embedding or [], payload=payload))

    return points


def require_documents(
    documents: list[Document], source: str, reader: Reader, read_errors: list[str], resource_id: str
) -> None:
    """Fail when an Agno reader yielded no documents.

    Args:
        documents: The documents the reader yielded.
        source: URL or description of the content read.
        reader: The Agno reader that read the source.
        read_errors: Warnings and errors Agno logged during the read.
        resource_id: ID of the content resource, named in the log event.

    Raises:
        RuntimeError: If the reader yielded no documents.
    """
    if not documents:
        logger.warning(
            "agno_content_read_error resource=%s source=%s count=%d first_message=%s",
            resource_id,
            source,
            len(read_errors),
            format_first_message(read_errors),
        )

        message = (
            f"{type(reader).__name__} yielded no documents for {source}; "
            f"{format_agno_warnings(read_errors)} during the read"
        )
        raise RuntimeError(message)


def require_embeddings(
    documents: list[Document], vector_size: int, collection: str, embed_errors: list[str], resource_id: str
) -> None:
    """Fail when a chunk carries no embedding of the collection's vector size.

    Args:
        documents: The chunks after Agno embedded them.
        vector_size: Vector size the collection stores.
        collection: Name of the collection receiving the chunks.
        embed_errors: Warnings and errors Agno logged while embedding.
        resource_id: ID of the content resource, named in the log event.

    Raises:
        RuntimeError: If a chunk's embedding is missing or has another size.
    """
    for document in documents:
        embedding_size = len(document.embedding or [])

        if embedding_size != vector_size:
            logger.warning(
                "agno_content_embed_error resource=%s collection=%s count=%d first_message=%s",
                resource_id,
                collection,
                len(embed_errors),
                format_first_message(embed_errors),
            )

            message = (
                f"Chunk {document.meta_data[CHUNK_INDEX_KEY]} of {document.name!r} has an embedding of size "
                f"{embedding_size}, but collection {collection!r} stores vectors of size {vector_size}; "
                f"{format_agno_warnings(embed_errors)} while embedding"
            )
            raise RuntimeError(message)


def require_json_container(content: bytes, source: str) -> None:
    """Fail when a JSON file does not hold a JSON object or array at its top level.

    Args:
        content: The downloaded file body.
        source: URL of the file, shown in errors.

    Raises:
        RuntimeError: If the body is not valid JSON or its top level is neither an object nor an array.
    """
    try:
        parsed = json.loads(content)
    except ValueError as error:
        message = f"{source} is not valid JSON: {error}"
        raise RuntimeError(message) from error

    if not isinstance(parsed, dict | list):
        message = f"{source} holds a JSON scalar; content reads a JSON file whose top level is an object or array"
        raise RuntimeError(message)


async def fetch_vector_size(client: AsyncQdrantClient, collection: str) -> int:
    """Read the size of the one unnamed dense vector a collection stores.

    Args:
        client: Client of the Qdrant server holding the collection.
        collection: Name of the collection.

    Returns:
        The collection's vector size.

    Raises:
        RuntimeError: If the collection does not exist or stores named vectors.
    """
    if not await client.collection_exists(collection):
        message = (
            f"Qdrant collection {collection!r} does not exist; "
            "apply a qdrant/collection resource that creates it before this content"
        )
        raise RuntimeError(message)

    vectors = (await client.get_collection(collection)).config.params.vectors

    if not isinstance(vectors, models.VectorParams):
        message = f"Qdrant collection {collection!r} stores named vectors; content needs one unnamed dense vector"
        raise RuntimeError(message)

    return vectors.size


async def fetch_url_content(url: str) -> bytes:
    """Download a URL's body, following redirects.

    Args:
        url: Remote file URL.

    Returns:
        The response body.

    Raises:
        RuntimeError: If the response has a non-success status, is an HTML page
            or the body exceeds ``DOWNLOAD_BYTE_LIMIT``.
    """
    body = bytearray()

    async with httpx.AsyncClient(follow_redirects=True) as client, client.stream("GET", url) as response:
        if not response.is_success:
            message = f"Downloading {build_display_url(url)} returned HTTP status {response.status_code}"
            raise RuntimeError(message)

        media_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()

        if media_type in HTML_MEDIA_TYPES:
            message = f"{build_display_url(url)} serves an HTML page; content reads a plain text, Markdown or JSON file"
            raise RuntimeError(message)

        async for chunk in response.aiter_bytes():
            body.extend(chunk)

            if len(body) > DOWNLOAD_BYTE_LIMIT:
                message = f"{build_display_url(url)} is larger than the {DOWNLOAD_BYTE_LIMIT}-byte download limit"
                raise RuntimeError(message)

    return bytes(body)


async def fetch_url_documents(url: str, name: str, resource_id: str) -> list[Document]:
    """Download a URL and read it into chunked documents with the Agno reader of its extension.

    Args:
        url: URL of a plain text, Markdown or JSON file.
        name: Document name.
        resource_id: ID of the content resource, named in log events.

    Returns:
        The chunked documents, never empty.

    Raises:
        RuntimeError: If the download returns a non-success status, the response is an HTML page,
            the download exceeds ``DOWNLOAD_BYTE_LIMIT``,
            a JSON file holds no JSON object or array, or the reader yields no documents.
    """
    content = await fetch_url_content(url)
    extension = parse_url_extension(url)
    source = build_display_url(url)

    if extension == ".json":
        require_json_container(content, source)

    reader = ReaderFactory.get_reader_for_extension(extension)

    with collect_agno_warnings() as read_errors:
        documents = await reader.async_read(BytesIO(content), name=name)

    require_documents(documents, source, reader, read_errors, resource_id)
    return documents


async def fetch_documents(spec: ContentSpec, resource_id: str) -> list[Document]:
    """Read the spec's content into chunked documents.

    Args:
        spec: The content specification.
        resource_id: ID of the content resource, named in log events.

    Returns:
        The chunked documents, never empty.

    Raises:
        RuntimeError: If downloading the URL returns a non-success status, the URL serves an HTML page,
            a download exceeds ``DOWNLOAD_BYTE_LIMIT``,
            a JSON file holds no JSON object or array, or the reader yields no documents.
    """
    if spec.url:
        return await fetch_url_documents(spec.url, spec.name, resource_id)

    reader = ReaderFactory.create_reader("text")

    with collect_agno_warnings() as read_errors:
        documents = await reader.async_read(BytesIO((spec.text_content or "").encode()), name=spec.name)

    require_documents(documents, "inline text", reader, read_errors, resource_id)
    return documents


async def write_points(client: AsyncQdrantClient, collection: str, points: list[models.PointStruct]) -> None:
    """Upsert points in batches of ``UPSERT_BATCH_SIZE``, each confirmed applied before the next.

    Args:
        client: Client of the Qdrant server holding the collection.
        collection: Collection receiving the points.
        points: The points to upsert.
    """
    for batch in itertools.batched(points, UPSERT_BATCH_SIZE):
        await client.upsert(collection_name=collection, points=list(batch), wait=True)


async def fetch_content_hashes(client: AsyncQdrantClient, collection: str, resource_id: str) -> set[str]:
    """Read the content hashes stored on every point a content resource owns.

    Args:
        client: Client of the Qdrant server holding the collection.
        collection: Collection holding the points.
        resource_id: Canonical id of the content resource.

    Returns:
        The distinct content hashes, empty when the resource owns no points.
    """
    content_hashes: set[str] = set()
    offset = None

    while True:
        records, offset = await client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=[build_resource_condition(resource_id)]),
            with_payload=["content_hash"],
            with_vectors=False,
            offset=offset,
        )
        content_hashes.update(record.payload["content_hash"] for record in records if record.payload)

        if offset is None:
            return content_hashes


class Content(AgnoResource[ContentConfig, ContentOutputs, ContentSpec]):
    """Content resource for Agno knowledge sources.

    Reads inline text, or a plain text, Markdown or JSON file URL, with Agno's
    readers and writes the chunks into the Knowledge base's Qdrant collection,
    which must already exist. The content is located by the points carrying its
    resource id.

    Lifecycle:
        - on_create: Write the chunks and remove any other points the resource owns
        - on_observe: Find the points the resource owns
        - on_update: Same as on_create
        - on_delete: Remove every point the resource owns
    """

    async def resolve_knowledge_spec(self) -> KnowledgeSpec:
        """Resolve the knowledge dependency and return its spec.

        Returns:
            The knowledge base specification.

        Raises:
            RuntimeError: If the knowledge dependency has no outputs.
        """
        knowledge = await self.config.knowledge.resolve()

        if knowledge.outputs is None:
            message = "knowledge dependency has no outputs"
            raise RuntimeError(message)

        return knowledge.outputs.spec

    async def locate_knowledge_spec(self) -> KnowledgeSpec | None:
        """Locate the knowledge whose collection holds the resource's points.

        Returns:
            Spec of the resolved knowledge dependency; when it cannot be resolved,
            the knowledge spec recorded in the outputs; None when neither exists.
        """
        try:
            return await self.resolve_knowledge_spec()
        except RuntimeError:
            if self.outputs is None:
                return None

            return self.outputs.spec.knowledge_spec

    def build_spec(self, knowledge_spec: KnowledgeSpec) -> ContentSpec:
        """Build the content specification from config.

        Args:
            knowledge_spec: Spec of the resolved knowledge dependency.

        Returns:
            ContentSpec with all configuration fields.
        """
        return ContentSpec(
            name=self.resolved(self.config.name) if self.config.name else self.name,
            knowledge_spec=knowledge_spec,
            url=self.resolved(self.config.url),
            text_content=self.resolved(self.config.text_content),
            description=self.resolved(self.config.description),
            metadata=self.resolved(self.config.metadata),
            topics=self.resolved(self.config.topics),
        )

    def build_outputs(self, spec: ContentSpec, content_hash: str | None) -> ContentOutputs:
        """Build outputs for a spec and the content hash its points carry.

        Args:
            spec: The content specification.
            content_hash: Content hash stored on the points, or None when they disagree.

        Returns:
            ContentOutputs with spec and content hash.
        """
        return ContentOutputs(spec=spec, content_hash=content_hash)

    async def on_create(self) -> ContentOutputs:
        """Write the content's chunks and delete every other point the resource owns.

        The resource's previous points are deleted only after every new chunk
        carries an embedding of the collection's vector size and Qdrant has
        confirmed the new points were written.

        Returns:
            ContentOutputs with the written content hash.

        Raises:
            RuntimeError: If the knowledge dependency has no outputs,
                downloading the content URL returns a non-success status,
                the collection does not exist or stores named vectors,
                the content cannot be read into documents,
                or a chunk's embedding does not fit the collection.
        """
        knowledge_spec = await self.resolve_knowledge_spec()
        spec = self.build_spec(knowledge_spec)
        content_hash = compute_content_hash(spec)
        vector_database = VectordbQdrant.from_spec(knowledge_spec.vector_db_spec)

        try:
            vector_size = await fetch_vector_size(vector_database.async_client, vector_database.collection)
            documents = await fetch_documents(spec, self.id)
            assign_chunk_payload(documents, spec, self.id)

            with collect_agno_warnings() as embed_errors:
                for document in documents:
                    await document.async_embed(embedder=vector_database.embedder)

            require_embeddings(documents, vector_size, vector_database.collection, embed_errors, self.id)
            await write_points(
                vector_database.async_client, vector_database.collection, build_points(documents, content_hash)
            )
            await vector_database.async_client.delete(
                collection_name=vector_database.collection,
                points_selector=build_stale_points_selector(self.id, content_hash, len(documents)),
                wait=True,
            )
        finally:
            await vector_database.async_close()

        return self.build_outputs(spec, content_hash)

    async def on_observe(self) -> ContentOutputs | None:
        """Find the points the resource owns in the knowledge's Qdrant collection.

        Returns:
            ContentOutputs with the stored content hash, or None if the resource owns
            no points or its knowledge cannot be located.
        """
        knowledge_spec = await self.locate_knowledge_spec()

        if knowledge_spec is None:
            return None

        vector_database_spec = knowledge_spec.vector_db_spec
        client = build_qdrant_client(vector_database_spec)

        try:
            if not await client.collection_exists(vector_database_spec.collection):
                return None

            content_hashes = await fetch_content_hashes(client, vector_database_spec.collection, self.id)
        finally:
            await client.close()

        if not content_hashes:
            return None

        spec = self.build_spec(knowledge_spec)

        if len(content_hashes) > 1:
            return self.build_outputs(spec, None)

        (content_hash,) = content_hashes
        return self.build_outputs(spec, content_hash)

    async def on_update(self, previous_config: ContentConfig | None) -> ContentOutputs:  # noqa: ARG002
        """Rewrite the content into the knowledge's Qdrant collection.

        Args:
            previous_config: The previous configuration, if any.

        Returns:
            ContentOutputs with the written content hash.
        """
        return await self.on_create()

    async def on_delete(self) -> None:
        """Remove every point the resource owns from the knowledge's Qdrant collection.

        Does nothing when the knowledge cannot be located or its collection does not exist.
        """
        knowledge_spec = await self.locate_knowledge_spec()

        if knowledge_spec is None:
            return

        vector_database_spec = knowledge_spec.vector_db_spec
        client = build_qdrant_client(vector_database_spec)

        try:
            if await client.collection_exists(vector_database_spec.collection):
                await client.delete(
                    collection_name=vector_database_spec.collection,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(must=[build_resource_condition(self.id)])
                    ),
                    wait=True,
                )
        finally:
            await client.close()
