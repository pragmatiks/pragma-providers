"""OpenAI provider for Pragmatiks.

Provides Chat Completions and Embeddings API resources
with reactive dependency management.
"""

from pragma_sdk import Provider

from openai_provider.resources import (
    ChatCompletions,
    ChatCompletionsConfig,
    ChatCompletionsOutputs,
    EmbedInput,
    EmbedOutput,
    Embeddings,
    EmbeddingsConfig,
    EmbeddingsOutputs,
)

openai = Provider(name="openai")

# Register resources
openai.resource("chat_completions")(ChatCompletions)
openai.resource("embeddings")(Embeddings)

__all__ = [
    "openai",
    "ChatCompletions",
    "ChatCompletionsConfig",
    "ChatCompletionsOutputs",
    "EmbedInput",
    "EmbedOutput",
    "Embeddings",
    "EmbeddingsConfig",
    "EmbeddingsOutputs",
]
