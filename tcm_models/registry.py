"""Model factory driven by ``configs/models.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .base import LLMClient, ModelSpec
from .providers import EchoClient, GeminiClient, OpenAICompatClient
from .replay import CachedClient, GenerationCache

PROVIDERS: Dict[str, Any] = {
    "openai_compat": OpenAICompatClient,
    "gemini": GeminiClient,
    "echo": EchoClient,
}


def spec_from_config(key: str, config: Mapping[str, Any]) -> ModelSpec:
    return ModelSpec(
        key=key,
        provider=str(config.get("provider", "openai_compat")),
        model_id=str(config.get("model_id", key)),
        base_url=str(config.get("base_url", "")),
        api_key_env=str(config.get("api_key_env", "")),
        input_usd_per_mtok=float(config.get("input_usd_per_mtok", 0.0)),
        output_usd_per_mtok=float(config.get("output_usd_per_mtok", 0.0)),
        extra_body=dict(config.get("extra_body") or {}),
        timeout_s=float(config.get("timeout_s", 180.0)),
        max_retries=int(config.get("max_retries", 5)),
    )


def build_client(
    spec: ModelSpec,
    *,
    cache_path: Optional[Path] = None,
    replay: bool = False,
    script: Optional[Sequence[str]] = None,
) -> LLMClient:
    """Instantiate a client, optionally wrapped in a cache / replay layer."""
    inner: Optional[LLMClient]
    if replay:
        inner = None  # never construct a provider client in replay mode
    elif spec.provider == "echo":
        inner = EchoClient(spec, script or ['{"action": "answer", "result": {}}'])
    else:
        factory = PROVIDERS.get(spec.provider)
        if factory is None:
            raise ValueError(
                f"unknown provider {spec.provider!r}; known: {sorted(PROVIDERS)}"
            )
        inner = factory(spec)

    if cache_path is None and not replay:
        assert inner is not None
        return inner
    cache = GenerationCache(Path(cache_path) if cache_path else Path("runs/.cache.jsonl"))
    return CachedClient(inner, cache, spec, strict=replay)
