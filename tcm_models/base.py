"""Unified model interface.

``LLMClient.generate`` is the *only* thing that differs between experimental
arms.  Everything else -- prompts, tools, retrieval, budgets, parsing, scoring
-- is frozen by the framework hash, so a score difference between two arms is
attributable to the model rather than to the scaffolding around it.
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class DecodeParams:
    """Decoding settings, identical across models within a run."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: Optional[int] = 20260829
    stop: Sequence[str] = ()

    def fingerprint(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=list)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Completion:
    """One model response plus everything the trace metrics need."""

    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str = ""
    n_retries: int = 0
    from_cache: bool = False
    error: Optional[str] = None
    raw: Optional[Mapping[str, Any]] = None

    def to_dict(self, *, include_raw: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "text": self.text,
            "model": self.model,
            "usage": asdict(self.usage),
            "latency_ms": round(self.latency_ms, 2),
            "finish_reason": self.finish_reason,
            "n_retries": self.n_retries,
            "from_cache": self.from_cache,
        }
        if self.error:
            payload["error"] = self.error
        if include_raw and self.raw is not None:
            payload["raw"] = self.raw
        return payload


@dataclass(frozen=True)
class ModelSpec:
    """Declarative model definition, loaded from ``configs/models.yaml``."""

    key: str
    provider: str
    model_id: str
    base_url: str = ""
    api_key_env: str = ""
    #: USD per million tokens, used for the cost metric
    input_usd_per_mtok: float = 0.0
    output_usd_per_mtok: float = 0.0
    #: provider-specific extras merged into the request body
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    timeout_s: float = 180.0
    max_retries: int = 5

    def cost_usd(self, usage: Usage) -> float:
        return (
            usage.prompt_tokens * self.input_usd_per_mtok
            + usage.completion_tokens * self.output_usd_per_mtok
        ) / 1_000_000


def request_key(
    model_id: str, messages: Sequence[Message], decode: DecodeParams, sample: int = 0
) -> str:
    """Stable hash of a generation request.

    Used both for the on-disk generation cache and for keyless replay: an
    identical request replays byte-for-byte, and a changed prompt necessarily
    misses, so a replayed run can never silently reflect a stale prompt.
    """
    payload = json.dumps(
        {
            "model": model_id,
            "messages": [m.to_dict() for m in messages],
            "decode": decode.fingerprint(),
            "sample": sample,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMError(RuntimeError):
    """Non-retryable model failure."""


class RetryableError(RuntimeError):
    """Transient failure: rate limit, timeout, 5xx."""


class LLMClient(abc.ABC):
    """The one seam between the framework and a provider."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.n_calls = 0
        self.total_usage = Usage()

    @property
    def name(self) -> str:
        return self.spec.key

    @abc.abstractmethod
    def _generate(
        self, messages: Sequence[Message], decode: DecodeParams
    ) -> Completion:  # pragma: no cover - provider specific
        ...

    def generate(
        self,
        messages: Sequence[Message],
        decode: Optional[DecodeParams] = None,
        *,
        sample: int = 0,
    ) -> Completion:
        """Generate with retry, backoff and usage accounting."""
        decode = decode or DecodeParams()
        delay = 2.0
        last_error: Optional[str] = None
        for attempt in range(self.spec.max_retries + 1):
            started = time.perf_counter()
            try:
                completion = self._generate(messages, decode)
                completion.latency_ms = (time.perf_counter() - started) * 1000
                completion.n_retries = attempt
                self.n_calls += 1
                self.total_usage.prompt_tokens += completion.usage.prompt_tokens
                self.total_usage.completion_tokens += completion.usage.completion_tokens
                return completion
            except RetryableError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.spec.max_retries:
                    break
                # full jitter: avoids thundering-herd retries across workers
                time.sleep(random.uniform(0, delay))
                delay = min(delay * 2, 60.0)
            except LLMError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break
        return Completion(
            text="",
            model=self.spec.model_id,
            error=last_error or "generation failed",
            n_retries=self.spec.max_retries,
            finish_reason="error",
        )

    def close(self) -> None:  # pragma: no cover - providers may override
        return None
