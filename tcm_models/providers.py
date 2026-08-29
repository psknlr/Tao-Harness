"""Provider adapters.

Four of the five frontier models under study speak the OpenAI chat-completions
shape (DeepSeek, GLM, MiniMax, GPT); Gemini has its own shape and gets a
dedicated adapter.  Both adapters normalise into the same
:class:`~tcm_models.base.Completion`, so the agent runtime above them is
provider-agnostic.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .base import (
    Completion,
    DecodeParams,
    LLMClient,
    LLMError,
    Message,
    ModelSpec,
    RetryableError,
    Usage,
)
from .http import post_json


def _api_key(spec: ModelSpec) -> str:
    if not spec.api_key_env:
        return ""
    key = os.environ.get(spec.api_key_env, "").strip()
    if not key:
        raise LLMError(
            f"environment variable {spec.api_key_env} is unset; "
            f"export it or run with --replay to use recorded traces"
        )
    return key


class OpenAICompatClient(LLMClient):
    """Chat-completions client for any OpenAI-compatible endpoint."""

    def _generate(self, messages: Sequence[Message], decode: DecodeParams) -> Completion:
        payload: Dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": [m.to_dict() for m in messages],
            "temperature": decode.temperature,
            "top_p": decode.top_p,
            "max_tokens": decode.max_tokens,
        }
        if decode.stop:
            payload["stop"] = list(decode.stop)
        if decode.seed is not None:
            payload["seed"] = decode.seed
        payload.update(self.spec.extra_body)

        url = self.spec.base_url.rstrip("/") + "/chat/completions"
        data = post_json(
            url,
            payload,
            {"Authorization": f"Bearer {_api_key(self.spec)}"},
            self.spec.timeout_s,
        )

        choices = data.get("choices") or []
        if not choices:
            raise RetryableError(f"no choices in response: {str(data)[:300]}")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        # Reasoning models may return an empty content with the answer in a
        # separate field; fall back rather than scoring a spurious blank.
        if not text:
            text = message.get("reasoning_content") or choices[0].get("text") or ""
        usage_raw = data.get("usage") or {}
        return Completion(
            text=str(text),
            model=str(data.get("model") or self.spec.model_id),
            usage=Usage(
                prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
                completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            ),
            finish_reason=str(choices[0].get("finish_reason") or ""),
            raw=data,
        )


class GeminiClient(LLMClient):
    """Google Generative Language ``generateContent`` adapter."""

    def _generate(self, messages: Sequence[Message], decode: DecodeParams) -> Completion:
        system_parts = [m.content for m in messages if m.role == "system"]
        contents: List[Dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            contents.append(
                {
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": [{"text": message.content}],
                }
            )

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": decode.temperature,
                "topP": decode.top_p,
                "maxOutputTokens": decode.max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if decode.stop:
            payload["generationConfig"]["stopSequences"] = list(decode.stop)
        payload.update(self.spec.extra_body)

        base = self.spec.base_url.rstrip("/")
        url = f"{base}/models/{self.spec.model_id}:generateContent"
        data = post_json(
            url, payload, {"x-goog-api-key": _api_key(self.spec)}, self.spec.timeout_s
        )

        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback") or {}
            if feedback.get("blockReason"):
                # a safety block is a real, non-retryable outcome and must be
                # recorded as such rather than retried into a rate limit
                raise LLMError(f"blocked by provider: {feedback}")
            raise RetryableError(f"no candidates in response: {str(data)[:300]}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(p.get("text") or "") for p in parts)
        usage_raw = data.get("usageMetadata") or {}
        return Completion(
            text=text,
            model=self.spec.model_id,
            usage=Usage(
                prompt_tokens=int(usage_raw.get("promptTokenCount") or 0),
                completion_tokens=int(usage_raw.get("candidatesTokenCount") or 0),
            ),
            finish_reason=str(candidates[0].get("finishReason") or ""),
            raw=data,
        )


class EchoClient(LLMClient):
    """Deterministic offline client used by the test suite.

    Responds from a scripted queue keyed by turn index, so a full agent loop --
    tool calls included -- can be exercised with no network and no API key.
    """

    def __init__(self, spec: ModelSpec, script: Sequence[str]):
        super().__init__(spec)
        self.script = list(script)
        self.turn = 0
        self.seen: List[List[Dict[str, str]]] = []

    def _generate(self, messages: Sequence[Message], decode: DecodeParams) -> Completion:
        self.seen.append([m.to_dict() for m in messages])
        text = self.script[self.turn] if self.turn < len(self.script) else self.script[-1]
        self.turn += 1
        return Completion(
            text=text,
            model=self.spec.model_id,
            usage=Usage(
                prompt_tokens=sum(len(m.content) for m in messages) // 4,
                completion_tokens=len(text) // 4,
            ),
            finish_reason="stop",
        )
