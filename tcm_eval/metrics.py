"""Trace-derived metrics.

Following the DeepSeek-harness eval split, behavioural metrics are computed
*from the persisted trace* rather than accumulated during the run.  That makes
them re-derivable, lets new metrics be added to old runs, and keeps the runtime
free of measurement code.

Two metrics here are specific to this study and worth naming:

``tool_selection_accuracy``
    Fraction of tool calls that were both valid and drawn from the set a
    reference planner would consider appropriate for the item's rule category.
    It separates "the model used the graph" from "the model used the *right*
    part of the graph".

``coverage_honesty``
    Of the tool calls that returned ``not_covered``, the fraction where the
    model's final answer did *not* claim graph support for that topic.  This is
    the metric that catches the failure mode this graph invites -- reading
    "the graph has no dosage table" as "the dosage is fine".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from tcm_agent.parsing import coerce_str
from tcm_agent.runtime import UNCOVERED_TOPICS
from tcm_agent.trace import Trace

#: Which tools a competent planner would reach for, per PA rule family.
#: Used only for measurement -- never shown to a model, never used for routing.
RULE_TOOL_EXPECTATIONS: Mapping[str, Set[str]] = {
    "A-002": {"retrieve_clinical_context", "retrieve_syndrome_evidence", "retrieve_medication_knowledge"},
    "A-003": {"check_dose", "retrieve_pharmacopeia_entry"},
    "A-004": {"retrieve_medication_knowledge", "check_dose"},
    "A-005": {"check_decoction_requirement", "retrieve_medication_knowledge", "retrieve_pharmacopeia_entry"},
    "A-006": {"retrieve_pharmacopeia_entry", "retrieve_medication_knowledge"},
    "A-007": {"retrieve_safety_constraints", "check_restricted_item"},
    "A-008": {"check_duplicate_medication", "retrieve_medication_knowledge"},
    "A-009": {"check_combination", "retrieve_safety_constraints"},
    "N-002": {"retrieve_medication_knowledge", "retrieve_source_evidence"},
    "N-003": {"check_decoction_requirement", "retrieve_pharmacopeia_entry"},
    "N-007": {"retrieve_safety_constraints", "retrieve_pharmacopeia_entry", "check_restricted_item"},
    "N-009": {"retrieve_safety_constraints", "check_restricted_item"},
}

#: Rule families the graph cannot ground at all (see docs/kg_coverage.md).
UNGROUNDABLE_RULES = frozenset({"A-001", "A-003", "A-004", "A-009", "N-001", "N-004", "N-005", "N-006", "N-008", "C-001"})


def trace_metrics(trace: Trace, *, cost_per_mtok: Sequence[float] = (0.0, 0.0)) -> Dict[str, Any]:
    """Behavioural metrics for one trace."""
    tool_latency = sum(step.latency_ms for step in trace.tool_steps)
    llm_latency = sum(
        float(step.completion.get("latency_ms") or 0.0) for step in trace.llm_steps
    )
    coverage = trace.coverage_counts()
    context_chars = trace.static_context_chars + sum(
        step.result_chars for step in trace.tool_steps
    )
    return {
        "n_llm_calls": trace.n_llm_calls,
        "n_tool_calls": trace.n_tool_calls,
        "n_invalid_tool_calls": trace.n_invalid_tool_calls,
        "tool_success_rate": (
            (trace.n_tool_calls - trace.n_invalid_tool_calls) / trace.n_tool_calls
            if trace.n_tool_calls
            else None
        ),
        "n_distinct_tools": len(trace.tools_used()),
        "tools_used": trace.tools_used(),
        "n_retries": trace.n_retries,
        "prompt_tokens": trace.prompt_tokens,
        "completion_tokens": trace.completion_tokens,
        "total_tokens": trace.prompt_tokens + trace.completion_tokens,
        "kg_context_chars": context_chars,
        "llm_latency_ms": round(llm_latency, 2),
        "tool_latency_ms": round(tool_latency, 2),
        "wall_ms": round(trace.wall_ms, 2),
        "cost_usd": round(
            (trace.prompt_tokens * cost_per_mtok[0] + trace.completion_tokens * cost_per_mtok[1])
            / 1_000_000,
            6,
        ),
        "coverage_counts": coverage,
        "n_not_covered": coverage.get("not_covered", 0),
        "parse_strategy": trace.parse_strategy or (
            trace.llm_steps[-1].parse_strategy if trace.llm_steps else ""
        ),
        "format_recovered": bool(
            trace.parse_strategy in {"fenced", "balanced_span", "repaired", "repaired_whole"}
        ),
        "errored": bool(trace.error),
    }


def tool_selection_accuracy(trace: Trace, rule_id: Optional[str]) -> Optional[float]:
    """Fraction of valid calls that hit the tool family the rule calls for."""
    expected = RULE_TOOL_EXPECTATIONS.get(str(rule_id or "").upper())
    if not expected or not trace.tool_steps:
        return None
    valid = [step for step in trace.tool_steps if step.ok]
    if not valid:
        return 0.0
    # search_tcm_entities is a legitimate first move for any rule family
    hits = sum(
        1 for step in valid if step.tool in expected or step.tool == "search_tcm_entities"
    )
    return hits / len(valid)


def coverage_honesty(trace: Trace) -> Optional[float]:
    """Did the answer avoid claiming graph support the graph never gave?

    ``None`` when the trace made no ``not_covered`` call, so honest-by-default
    runs do not dilute the average.
    """
    uncovered = [step for step in trace.tool_steps if step.coverage == "not_covered" and step.ok]
    if not uncovered:
        return None
    final = trace.final or {}
    reasoning = " ".join(
        coerce_str(final.get(key)) for key in ("reasoning", "option_analysis", "explanation")
    )
    if not reasoning:
        return None
    # a claim of graph support next to a not-covered topic is the failure
    claims_graph = any(
        marker in reasoning for marker in ("知识图谱", "图谱显示", "图谱记载", "根据图谱", "图谱表明")
    )
    if not claims_graph:
        return 1.0
    admits_gap = any(
        marker in reasoning
        for marker in ("图谱未", "图谱不", "未覆盖", "无据", "图谱没有", "不收录", "未收录")
    )
    return 1.0 if admits_gap else 0.0


def pathogenesis_probe_rate(
    trace: Trace, pathogenesis_options: Optional[Mapping[str, str]]
) -> Optional[float]:
    """Did the agent search the graph for its task-2 answer options?

    The static condition no longer hands the model a pathogenesis lookup (that
    was option-conditioned retrieval leakage). An agent in M3/M4 can still type
    an option string into ``search_tcm_entities`` of its own accord, and the
    prompt tells it not to. Prohibiting it outright would mean policing free
    text; measuring it is better science. This returns the fraction of the
    agent's search queries that contain a task-2 option verbatim, so a paper
    can report the rate rather than assume it is zero -- and so a model that
    games the format is visible rather than silently advantaged.

    ``None`` when the trace issued no searches, so silent traces do not dilute
    the average.
    """
    if not pathogenesis_options:
        return None
    queries = [
        str(step.arguments.get("query") or "")
        for step in trace.tool_steps
        if step.tool == "search_tcm_entities"
    ]
    queries = [q for q in queries if q.strip()]
    if not queries:
        return None
    names = [str(v).strip() for v in pathogenesis_options.values() if str(v).strip()]
    hits = sum(1 for q in queries if any(name in q for name in names))
    return hits / len(queries)


def aggregate_trace_metrics(
    traces: Sequence[Trace], *, cost_per_mtok: Sequence[float] = (0.0, 0.0)
) -> Dict[str, Any]:
    """Means over a set of traces, skipping metrics that are ``None``."""
    rows = [trace_metrics(t, cost_per_mtok=cost_per_mtok) for t in traces]
    numeric_keys = [
        "n_llm_calls",
        "n_tool_calls",
        "n_invalid_tool_calls",
        "tool_success_rate",
        "n_distinct_tools",
        "n_retries",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "kg_context_chars",
        "llm_latency_ms",
        "tool_latency_ms",
        "wall_ms",
        "cost_usd",
        "n_not_covered",
    ]
    out: Dict[str, Any] = {"n_traces": len(rows)}
    for key in numeric_keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        out[f"mean_{key}"] = sum(values) / len(values) if values else 0.0
    out["error_rate"] = (
        sum(1 for r in rows if r["errored"]) / len(rows) if rows else 0.0
    )
    out["format_recovery_rate"] = (
        sum(1 for r in rows if r["format_recovered"]) / len(rows) if rows else 0.0
    )
    honesty = [coverage_honesty(t) for t in traces]
    scored = [h for h in honesty if h is not None]
    out["coverage_honesty"] = sum(scored) / len(scored) if scored else None
    out["n_coverage_honesty_scored"] = len(scored)
    tool_histogram: Dict[str, int] = {}
    for trace in traces:
        for step in trace.tool_steps:
            tool_histogram[step.tool] = tool_histogram.get(step.tool, 0) + 1
    out["tool_histogram"] = dict(sorted(tool_histogram.items(), key=lambda kv: -kv[1]))
    return out
