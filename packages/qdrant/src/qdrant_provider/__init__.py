"""Qdrant provider for Pragmatiks.

Provides vector storage and similarity search resources
with reactive dependency management.
"""

from pragma_sdk import Provider

from qdrant_provider.resources import (
    Collection,
    CollectionConfig,
    CollectionOutputs,
    VectorConfig,
)

qdrant = Provider(name="qdrant")

# Register resources
qdrant.resource("collection")(Collection)

__all__ = [
    "qdrant",
    "Collection",
    "CollectionConfig",
    "CollectionOutputs",
    "VectorConfig",
]
