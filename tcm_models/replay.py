"""Generation cache and keyless replay.

Borrowed from the DeepSeek-harness eval design: record once with a key, then
replay in CI without one.  Two consequences worth stating in a methods section:

* Re-scoring is free.  Generation and scoring are separate phases, so a scorer
  fix is re-run over recorded outputs instead of re-billing 300 cases x 5
  models x 5 conditions.
* Replay is prompt-sensitive by construction.  The cache key hashes the full
  message list, so editing a prompt necessarily misses the cache; a replayed
  run can never quietly represent a prompt that no longer exists.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .base import (
    Completion,
    DecodeParams,
    LLMClient,
    LLMError,
    Message,
    ModelSpec,
    Usage,
    request_key,
)


class GenerationCache:
    """Append-only JSONL cache of ``request_key -> completion``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._handle = None
        self.load()

    def load(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated final line from a killed run
                key = record.get("key")
                if key:
                    self._entries[key] = record
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def get(self, key: str) -> Optional[Completion]:
        record = self._entries.get(key)
        if record is None:
            return None
        payload = record.get("completion") or {}
        usage = payload.get("usage") or {}
        return Completion(
            text=payload.get("text", ""),
            model=payload.get("model", ""),
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
            ),
            latency_ms=float(payload.get("latency_ms") or 0.0),
            finish_reason=str(payload.get("finish_reason") or ""),
            n_retries=int(payload.get("n_retries") or 0),
            from_cache=True,
            error=payload.get("error"),
        )

    def put(
        self,
        key: str,
        completion: Completion,
        *,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        record = {
            "key": key,
            "completion": completion.to_dict(),
            **({"meta": dict(meta)} if meta else {}),
        }
        with self._lock:
            self._entries[key] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


class CachedClient(LLMClient):
    """Wraps a client with a persistent cache; can also run strictly offline.

    ``strict=True`` is replay mode: a cache miss raises instead of contacting a
    provider, which is what makes a replay run provably free of new generation.
    """

    def __init__(
        self,
        inner: Optional[LLMClient],
        cache: GenerationCache,
        spec: ModelSpec,
        *,
        strict: bool = False,
    ):
        super().__init__(spec)
        self.inner = inner
        self.cache = cache
        self.strict = strict
        self.n_hits = 0
        self.n_misses = 0

    def generate(
        self,
        messages: Sequence[Message],
        decode: Optional[DecodeParams] = None,
        *,
        sample: int = 0,
    ) -> Completion:
        decode = decode or DecodeParams()
        key = request_key(self.spec.model_id, messages, decode, sample)
        hit = self.cache.get(key)
        if hit is not None:
            self.n_hits += 1
            return hit
        self.n_misses += 1
        if self.strict or self.inner is None:
            return Completion(
                text="",
                model=self.spec.model_id,
                finish_reason="replay_miss",
                error=(
                    "replay cache miss: this request was never recorded. "
                    "The prompt, tools or decode settings differ from the "
                    "recorded run, so replay cannot represent it."
                ),
            )
        completion = self.inner.generate(messages, decode, sample=sample)
        if not completion.error:
            self.cache.put(key, completion, meta={"model_key": self.spec.key})
        self.n_calls += 1
        return completion

    def _generate(self, messages: Sequence[Message], decode: DecodeParams) -> Completion:
        raise NotImplementedError  # generate() is overridden above
