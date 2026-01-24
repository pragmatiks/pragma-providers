"""Resource definitions for openai provider.

Import and export your Resource classes here for discovery by the runtime.
"""

from openai_provider.resources.chat_completions import (
    ChatCompletions,
    ChatCompletionsConfig,
    ChatCompletionsOutputs,
)
from openai_provider.resources.embeddings import (
    EmbedInput,
    EmbedOutput,
    Embeddings,
    EmbeddingsConfig,
    EmbeddingsOutputs,
)

__all__ = [
    "ChatCompletions",
    "ChatCompletionsConfig",
    "ChatCompletionsOutputs",
    "EmbedInput",
    "EmbedOutput",
    "Embeddings",
    "EmbeddingsConfig",
    "EmbeddingsOutputs",
]
