"""Trace format.

Every run writes one JSON object per case containing the full interaction --
each model turn, each tool call and its coverage verdict, token usage and
latency.  Metrics are then computed *from the trace* rather than accumulated
during the run, which is what makes re-scoring, replay and post-hoc analyses
(tool-selection accuracy, coverage-honesty, context growth) possible without
re-running anything.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tcm_models.base import Completion


@dataclass
class LLMStep:
    index: int
    phase: str  # "reason" | "answer" | "verify_revise"
    prompt_chars: int
    completion: Dict[str, Any]
    parse_strategy: str = ""
    action: str = ""  # "tool" | "answer" | "unparsed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolStep:
    index: int
    tool: str
    arguments: Dict[str, Any]
    coverage: str
    ok: bool
    latency_ms: float
    n_results: int
    error: Optional[str] = None
    result_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Trace:
    """One case, one condition, one model."""

    run_id: str
    case_id: str
    dataset: str
    condition: str
    model_key: str
    framework_hash: str
    sample: int = 0
    llm_steps: List[LLMStep] = field(default_factory=list)
    tool_steps: List[ToolStep] = field(default_factory=list)
    static_context_chars: int = 0
    final_raw: str = ""
    final: Optional[Dict[str, Any]] = None
    parse_strategy: str = ""
    error: Optional[str] = None
    wall_ms: float = 0.0
    started_at: str = ""

    # ---------------------------------------------------------------- metrics
    @property
    def prompt_tokens(self) -> int:
        return sum(int(s.completion.get("usage", {}).get("prompt_tokens", 0)) for s in self.llm_steps)

    @property
    def completion_tokens(self) -> int:
        return sum(
            int(s.completion.get("usage", {}).get("completion_tokens", 0))
            for s in self.llm_steps
        )

    @property
    def n_llm_calls(self) -> int:
        return len(self.llm_steps)

    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_steps)

    @property
    def n_invalid_tool_calls(self) -> int:
        return sum(1 for s in self.tool_steps if not s.ok)

    @property
    def n_retries(self) -> int:
        return sum(int(s.completion.get("n_retries", 0)) for s in self.llm_steps)

    def tools_used(self) -> List[str]:
        seen: List[str] = []
        for step in self.tool_steps:
            if step.tool not in seen:
                seen.append(step.tool)
        return seen

    def coverage_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for step in self.tool_steps:
            counts[step.coverage] = counts.get(step.coverage, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "dataset": self.dataset,
            "condition": self.condition,
            "model_key": self.model_key,
            "framework_hash": self.framework_hash,
            "sample": self.sample,
            "started_at": self.started_at,
            "wall_ms": round(self.wall_ms, 2),
            "static_context_chars": self.static_context_chars,
            "llm_steps": [s.to_dict() for s in self.llm_steps],
            "tool_steps": [s.to_dict() for s in self.tool_steps],
            "final": self.final,
            "final_raw": self.final_raw[:8000],
            "parse_strategy": self.parse_strategy,
            "error": self.error,
            "metrics": {
                "n_llm_calls": self.n_llm_calls,
                "n_tool_calls": self.n_tool_calls,
                "n_invalid_tool_calls": self.n_invalid_tool_calls,
                "n_retries": self.n_retries,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "tools_used": self.tools_used(),
                "coverage_counts": self.coverage_counts(),
            },
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "Trace":
        trace = Trace(
            run_id=str(payload.get("run_id", "")),
            case_id=str(payload.get("case_id", "")),
            dataset=str(payload.get("dataset", "")),
            condition=str(payload.get("condition", "")),
            model_key=str(payload.get("model_key", "")),
            framework_hash=str(payload.get("framework_hash", "")),
            sample=int(payload.get("sample", 0)),
            static_context_chars=int(payload.get("static_context_chars", 0)),
            final=payload.get("final"),
            final_raw=str(payload.get("final_raw", "")),
            parse_strategy=str(payload.get("parse_strategy", "")),
            error=payload.get("error"),
            wall_ms=float(payload.get("wall_ms", 0.0)),
            started_at=str(payload.get("started_at", "")),
        )
        for step in payload.get("llm_steps", []):
            trace.llm_steps.append(LLMStep(**step))
        for step in payload.get("tool_steps", []):
            trace.tool_steps.append(
                ToolStep(
                    index=int(step["index"]),
                    tool=str(step["tool"]),
                    arguments=dict(step.get("arguments") or {}),
                    coverage=str(step.get("coverage", "")),
                    ok=bool(step.get("ok", False)),
                    latency_ms=float(step.get("latency_ms", 0.0)),
                    n_results=int(step.get("n_results", 0)),
                    error=step.get("error"),
                    result_chars=int(step.get("result_chars", 0)),
                )
            )
        return trace


def write_traces(path, traces: Sequence[Trace]) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


def read_traces(path) -> List[Trace]:
    from pathlib import Path

    target = Path(path)
    out: List[Trace] = []
    if not target.exists():
        return out
    with open(target, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(Trace.from_dict(json.loads(line)))
    return out
