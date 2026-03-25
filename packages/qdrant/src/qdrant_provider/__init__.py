"""Qdrant provider for Pragmatiks.

Provides vector storage and similarity search resources
with reactive dependency management.
"""

from pragma_sdk import Provider

from qdrant_provider.resources import (
    Collection,
    CollectionConfig,
    CollectionOutputs,
    Database,
    DatabaseConfig,
    DatabaseOutputs,
    ResourceConfig,
    StorageConfig,
    VectorConfig,
)


qdrant = Provider()

qdrant.resource("collection")(Collection)
qdrant.resource("database")(Database)

__all__ = [
    "qdrant",
    "Collection",
    "CollectionConfig",
    "CollectionOutputs",
    "Database",
    "DatabaseConfig",
    "DatabaseOutputs",
    "ResourceConfig",
    "StorageConfig",
    "VectorConfig",
]
