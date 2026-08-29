"""Model adapters, generation cache and keyless replay."""

from .base import (
    Completion,
    DecodeParams,
    LLMClient,
    LLMError,
    Message,
    ModelSpec,
    RetryableError,
    Usage,
    request_key,
)
from .providers import EchoClient, GeminiClient, OpenAICompatClient
from .registry import PROVIDERS, build_client, spec_from_config
from .replay import CachedClient, GenerationCache

__all__ = [
    "CachedClient",
    "Completion",
    "DecodeParams",
    "EchoClient",
    "GeminiClient",
    "GenerationCache",
    "LLMClient",
    "LLMError",
    "Message",
    "ModelSpec",
    "OpenAICompatClient",
    "PROVIDERS",
    "RetryableError",
    "Usage",
    "build_client",
    "request_key",
    "spec_from_config",
]
