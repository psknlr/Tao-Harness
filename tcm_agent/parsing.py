"""Tolerant JSON extraction from model output.

Frontier models wrap JSON in prose, in Markdown fences, in full-width
punctuation, or emit a trailing comma.  A brittle ``json.loads`` turns those
into scored failures and silently penalises whichever model happens to be
chattiest -- a formatting confound in what is meant to be a knowledge
comparison.  The ladder below is applied in order and the first success wins;
:func:`extract_json_object` reports which rung succeeded so that formatting
robustness is itself measurable rather than invisible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
#: full-width punctuation that appears inside otherwise valid JSON
_FULLWIDTH_MAP = str.maketrans({"｛": "{", "｝": "}", "［": "[", "］": "]", "：": ":", "，": ","})


@dataclass
class ParseOutcome:
    """Result of a parse attempt, including how it was recovered."""

    value: Optional[Dict[str, Any]]
    strategy: str
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None


def _balanced_objects(text: str) -> List[str]:
    """Every balanced ``{...}`` span, string- and escape-aware."""
    spans: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start : i + 1])
    return spans


def _repair(candidate: str) -> str:
    """Fixes that never change a well-formed document's meaning."""
    repaired = candidate.translate(_FULLWIDTH_MAP)
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    # bare newlines inside string literals
    out: List[str] = []
    in_string = False
    escaped = False
    for char in repaired:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            elif char in "\n\r":
                out.append("\\n")
                continue
        elif char == '"':
            in_string = True
        out.append(char)
    return "".join(out)


def extract_json_object(text: str) -> ParseOutcome:
    """Recover a JSON object from model output, or report failure."""
    if not text or not text.strip():
        return ParseOutcome(None, "empty", text or "")
    stripped = text.strip()

    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return ParseOutcome(value, "direct", stripped)
    except json.JSONDecodeError:
        pass

    for block in _FENCE_RE.findall(stripped):
        try:
            value = json.loads(block.strip())
            if isinstance(value, dict):
                return ParseOutcome(value, "fenced", block.strip())
        except json.JSONDecodeError:
            continue

    spans = _balanced_objects(stripped)
    # prefer the longest span: a model that narrates before answering usually
    # emits small illustrative objects first and the real answer last/largest
    for span in sorted(spans, key=len, reverse=True):
        try:
            value = json.loads(span)
            if isinstance(value, dict):
                return ParseOutcome(value, "balanced_span", span)
        except json.JSONDecodeError:
            continue

    for span in sorted(spans, key=len, reverse=True):
        try:
            value = json.loads(_repair(span))
            if isinstance(value, dict):
                return ParseOutcome(value, "repaired", span)
        except json.JSONDecodeError:
            continue

    try:
        value = json.loads(_repair(stripped))
        if isinstance(value, dict):
            return ParseOutcome(value, "repaired_whole", stripped)
    except json.JSONDecodeError:
        pass

    return ParseOutcome(None, "failed", stripped[:2000])


def coerce_str(value: Any) -> str:
    """Flatten whatever a model put in a string field into a string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "；".join(coerce_str(v) for v in value if v is not None)
    if isinstance(value, Mapping):
        return "；".join(f"{k}: {coerce_str(v)}" for k, v in value.items())
    return str(value)


def coerce_list(value: Any) -> List[str]:
    """Flatten whatever a model put in a list field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\s;、，；/]+", value.strip())
        return [p for p in parts if p]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(coerce_list(item) if not isinstance(item, str) else [item.strip()])
        return [o for o in out if o]
    return [str(value)]
