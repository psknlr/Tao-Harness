"""Provider adapters.

Most of the models under study speak the OpenAI chat-completions shape
(DeepSeek, GLM, GPT); Gemini has its own. MiniMax and Poe speak the OpenAI
shape but diverge in ways that would otherwise become per-model confounds --
MiniMax reports errors with HTTP 200 and may omit the prompt/completion token
split, and Poe is a gateway whose bot name is not a snapshot identity -- so
each gets a thin adapter that normalises the difference rather than letting it
land in the results.

Every adapter returns the same :class:`~tcm_models.base.Completion`, so the
agent runtime above them is provider-agnostic and no arm can differ because of
how its provider was read.
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
        if decode.seed is not None:
            # The OpenAI-compatible client has always sent the seed; this one
            # did not, so with samples > 1 the two providers were drawing
            # samples under different rules -- one seeded per sample index, one
            # at the backend's discretion. That is a per-provider confound
            # inside self-consistency, not a self-consistency result. Whether
            # the endpoint honours it is verified by `benchmark_runner smoke`
            # rather than assumed here.
            payload["generationConfig"]["seed"] = decode.seed
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


class MiniMaxClient(OpenAICompatClient):
    """MiniMax chat-completions.

    MiniMax speaks the OpenAI shape, so the request side is inherited. Two
    things differ enough to matter for a controlled comparison:

    * **Errors arrive with HTTP 200.** A failed request returns
      ``base_resp.status_code != 0`` in the body rather than a non-2xx status,
      so the shared HTTP layer sees success and the run records an empty
      completion as a legitimate blank answer. That would show up as a lower
      ``answered`` rate for MiniMax and read as a model difference.
    * **Usage may report only ``total_tokens``.** Splitting cost between
      prompt and completion at their published rates would be a guess, so the
      total is recorded as completion tokens and the prompt count left at zero
      -- visibly incomplete rather than plausibly wrong. ``smoke`` reports it.
    """

    #: Documented MiniMax API status codes worth retrying rather than failing.
    _RETRYABLE = frozenset({1002, 1027, 1039, 2013})

    def _generate(self, messages: Sequence[Message], decode: DecodeParams) -> Completion:
        completion = super()._generate(messages, decode)
        body = completion.raw if isinstance(completion.raw, Mapping) else {}

        resp = body.get("base_resp") or {}
        status = int(resp.get("status_code") or 0)
        if status:
            detail = str(resp.get("status_msg") or "")
            message = f"MiniMax error {status}: {detail}"
            if status in self._RETRYABLE:
                raise RetryableError(message)
            raise LLMError(message)

        # Fill in a usage split MiniMax sometimes omits.
        usage_raw = body.get("usage") or {}
        if not completion.usage.prompt_tokens and not completion.usage.completion_tokens:
            total = int(usage_raw.get("total_tokens") or 0)
            if total:
                completion.usage = Usage(prompt_tokens=0, completion_tokens=total)
        return completion


class PoeClient(OpenAICompatClient):
    """Poe's OpenAI-compatible gateway.

    Poe fronts many upstream models behind one endpoint, which makes it useful
    for reaching a model whose first-party API is not available -- and makes
    two things the harness depends on less certain:

    * **The bot name is the model identity.** ``model_id`` is a Poe bot, and a
      bot can be repointed at a different upstream snapshot without its name
      changing. The manifest therefore cannot promise the same reproducibility
      it does for a first-party endpoint, so the completion records
      ``via_gateway`` and the spec is fingerprinted with the gateway marked.
      Prefer a first-party adapter where one exists; a paper that pins a
      snapshot should say which models came through here.
    * **Sampling controls may be ignored.** Poe forwards what the upstream bot
      accepts, so ``seed`` in particular may not be honoured even though the
      request carries it. ``smoke`` reports whether two seeded calls agree, and
      that answer is the one to trust over this docstring.
    """

    def _generate(self, messages: Sequence[Message], decode: DecodeParams) -> Completion:
        completion = super()._generate(messages, decode)
        # Say plainly, in the trace, that this answer came through a gateway.
        if isinstance(completion.raw, dict):
            completion.raw.setdefault("via_gateway", "poe")
        return completion


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
