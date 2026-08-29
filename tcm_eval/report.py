"""Report generation.

Produces the tables the study's four research questions need:

RQ1  base-model spread with no external knowledge (M0 / M1)
RQ2  whether the graph helps, per model and per benchmark (M0 -> M2/M3)
RQ3  static KG-RAG versus dynamic KG-Agent (M2 -> M3), and verification (M3 -> M4)
RQ4  whether the gain differs between knowledge the graph encodes explicitly
     (PA safety/pharmacopoeia rules) and reasoning it only constrains
     indirectly (SDT pathogenesis, which is not a graph entity at all)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .scorers import PA_METRICS, PA_PRIMARY, SDT_METRICS, SDT_PRIMARY, ScoredItem, aggregate
from .stats import PairedResult, holm_bonferroni, mcnemar, paired_bootstrap, spearman

CONDITION_LABELS = {
    "M0": "M0 Base LLM",
    "M1": "M1 Structured",
    "M2": "M2 KG-RAG (static)",
    "M3": "M3 KG-Agent",
    "M4": "M4 KG-Agent + Verify",
}

CONTRASTS = (
    ("M0", "M1", "prompt structure"),
    ("M1", "M2", "static KG retrieval"),
    ("M2", "M3", "agentic tool use"),
    ("M3", "M4", "evidence verification"),
    ("M0", "M3", "total KG-Agent gain"),
)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "–"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def index_items(
    items: Sequence[ScoredItem],
) -> Dict[Tuple[str, str], Dict[str, ScoredItem]]:
    """``(model, condition) -> {case_id: item}`` for paired lookups."""
    out: Dict[Tuple[str, str], Dict[str, ScoredItem]] = defaultdict(dict)
    for item in items:
        out[(item.model_key, item.condition)][item.case_id] = item
    return out


def paired_vectors(
    index: Mapping[Tuple[str, str], Mapping[str, ScoredItem]],
    model: str,
    left: str,
    right: str,
    metric: str,
) -> Tuple[List[float], List[float]]:
    """Aligned metric vectors over the cases both conditions attempted."""
    a = index.get((model, left), {})
    b = index.get((model, right), {})
    shared = sorted(set(a) & set(b))
    xs, ys = [], []
    for case_id in shared:
        left_value = a[case_id].metrics.get(metric)
        right_value = b[case_id].metrics.get(metric)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            xs.append(float(left_value))
            ys.append(float(right_value))
    return xs, ys


def main_table(items: Sequence[ScoredItem], dataset: str, metrics: Sequence[str]) -> str:
    """Model x condition grid of the headline metrics."""
    grid: Dict[Tuple[str, str], List[ScoredItem]] = defaultdict(list)
    for item in items:
        if item.dataset == dataset:
            grid[(item.model_key, item.condition)].append(item)

    models = sorted({m for m, _ in grid})
    conditions = [c for c in CONDITION_LABELS if any((m, c) in grid for m in models)]
    headers = ["model", "condition", "n"] + list(metrics)
    rows: List[List[Any]] = []
    for model in models:
        for condition in conditions:
            bucket = grid.get((model, condition))
            if not bucket:
                continue
            summary = aggregate(bucket, metrics)
            rows.append(
                [model, CONDITION_LABELS[condition], int(summary["n"])]
                + [summary[m] for m in metrics]
            )
    return _md_table(headers, rows)


def contrast_table(
    items: Sequence[ScoredItem], dataset: str, metric: str, *, binary: bool = True
) -> Tuple[str, Dict[str, PairedResult]]:
    """Paired condition contrasts per model, with multiplicity control."""
    scoped = [i for i in items if i.dataset == dataset]
    index = index_items(scoped)
    models = sorted({i.model_key for i in scoped})

    results: Dict[str, PairedResult] = {}
    rows: List[List[Any]] = []
    for model in models:
        for left, right, label in CONTRASTS:
            xs, ys = paired_vectors(index, model, left, right, metric)
            if not xs:
                continue
            result = mcnemar(xs, ys) if binary else paired_bootstrap(xs, ys)
            key = f"{dataset}|{model}|{left}->{right}"
            results[key] = result
            rows.append(
                [
                    model,
                    f"{left}→{right}",
                    label,
                    result.n,
                    result.mean_a,
                    result.mean_b,
                    result.delta,
                    result.wins,
                    result.losses,
                    result.p_value,
                ]
            )

    corrected = holm_bonferroni({k: v.p_value for k, v in results.items()})
    for row, key in zip(rows, results):
        row.append(corrected[key]["p_adjusted"])
        row.append("✓" if corrected[key]["reject"] else "")

    headers = [
        "model",
        "contrast",
        "what it isolates",
        "n",
        "before",
        "after",
        "Δ",
        "win",
        "lose",
        "p",
        "p (Holm)",
        "sig",
    ]
    return _md_table(headers, rows), results


def compensation_table(
    items: Sequence[ScoredItem], dataset: str, metric: str, base: str = "M0", enhanced: str = "M3"
) -> Tuple[str, Dict[str, Any]]:
    """Does the framework help weaker models more?  (the headline RQ)"""
    scoped = [i for i in items if i.dataset == dataset]
    index = index_items(scoped)
    models = sorted({i.model_key for i in scoped})

    rows: List[List[Any]] = []
    bases: List[float] = []
    deltas: List[float] = []
    for model in models:
        xs, ys = paired_vectors(index, model, base, enhanced, metric)
        if not xs:
            continue
        base_score = sum(xs) / len(xs)
        delta = sum(ys) / len(ys) - base_score
        bases.append(base_score)
        deltas.append(delta)
        rows.append([model, len(xs), base_score, sum(ys) / len(ys), delta])

    rho, n = spearman(bases, deltas)
    table = _md_table(["model", "n", base, enhanced, "Δ"], rows)
    stats = {
        "spearman_rho_base_vs_delta": None if rho != rho else round(rho, 4),
        "n_models": n,
        "interpretation": (
            "negative rho means the framework compensates weaker base models"
            if n >= 3
            else "too few models for a rank correlation"
        ),
    }
    return table, stats


def explicit_vs_implicit_table(items: Sequence[ScoredItem]) -> str:
    """RQ4: gain on graph-explicit knowledge versus graph-constrained reasoning.

    PA accuracy depends on knowledge the graph encodes as entities and edges.
    SDT syndrome sits between: the graph names syndromes but not symptoms.
    SDT pathogenesis is the pure case -- the graph has no such entity at all,
    so any gain there comes only from the graph narrowing the reasoning space.
    """
    specs = [
        ("PA", "pa", "exact", "explicit — safety / pharmacopoeia rules are graph entities"),
        ("SDT syndrome", "sdt", "syndrome_exact", "partial — syndromes are entities, symptoms are not"),
        (
            "SDT pathogenesis",
            "sdt",
            "pathogenesis_f1",
            "implicit — pathogenesis is not a graph entity at all",
        ),
        (
            "SDT clinical info",
            "sdt",
            "clinical_information_f1",
            "implicit — extraction from the case, unaided by the graph",
        ),
    ]
    index = index_items(items)
    models = sorted({i.model_key for i in items})
    rows: List[List[Any]] = []
    for label, dataset, metric, kind in specs:
        deltas: List[float] = []
        base_values: List[float] = []
        for model in models:
            scoped_index = {
                k: v for k, v in index.items() if any(i.dataset == dataset for i in v.values())
            }
            xs, ys = paired_vectors(scoped_index, model, "M0", "M3", metric)
            if not xs:
                continue
            base_values.append(sum(xs) / len(xs))
            deltas.append(sum(ys) / len(ys) - sum(xs) / len(xs))
        if not deltas:
            continue
        rows.append(
            [
                label,
                kind,
                len(deltas),
                sum(base_values) / len(base_values),
                sum(deltas) / len(deltas),
            ]
        )
    return _md_table(
        ["target", "graph relationship", "models", "mean M0", "mean Δ (M0→M3)"], rows
    )


def build_report(
    items: Sequence[ScoredItem],
    *,
    trace_summaries: Optional[Mapping[str, Any]] = None,
    framework: Optional[Mapping[str, Any]] = None,
    title: str = "TCM-KG Agent — results",
) -> str:
    """Assemble the full Markdown report."""
    parts: List[str] = [f"# {title}", ""]

    if framework:
        parts.append("## Framework contract")
        parts.append("")
        parts.append(
            "All arms share one frozen framework; only `model.generate()` differs."
        )
        parts.append("")
        parts.append("```json")
        parts.append(json.dumps(dict(framework), ensure_ascii=False, indent=2))
        parts.append("```")
        parts.append("")

    datasets = sorted({i.dataset for i in items})
    for dataset in datasets:
        metrics = SDT_METRICS if dataset == "sdt" else PA_METRICS
        primary = SDT_PRIMARY if dataset == "sdt" else PA_PRIMARY
        parts.append(f"## {dataset.upper()}")
        parts.append("")
        parts.append(f"### Main results (primary metric: `{primary}`)")
        parts.append("")
        parts.append(main_table(items, dataset, metrics))
        parts.append("")
        table, _ = contrast_table(items, dataset, primary, binary=True)
        parts.append("### Paired condition contrasts")
        parts.append("")
        parts.append("Exact McNemar per model; Holm–Bonferroni across the whole family.")
        parts.append("")
        parts.append(table)
        parts.append("")
        comp_table, comp_stats = compensation_table(items, dataset, primary)
        parts.append("### Does the framework compensate weaker models?")
        parts.append("")
        parts.append(comp_table)
        parts.append("")
        parts.append(f"Spearman ρ(base, Δ) = `{comp_stats['spearman_rho_base_vs_delta']}` "
                     f"over {comp_stats['n_models']} models — {comp_stats['interpretation']}.")
        parts.append("")

    if len(datasets) > 1:
        parts.append("## Explicit knowledge vs graph-constrained reasoning")
        parts.append("")
        parts.append(explicit_vs_implicit_table(items))
        parts.append("")

    if trace_summaries:
        parts.append("## Agent behaviour (from traces)")
        parts.append("")
        headers = [
            "arm",
            "traces",
            "LLM calls",
            "tool calls",
            "invalid",
            "tool success",
            "tokens",
            "KG context chars",
            "coverage honesty",
        ]
        rows = []
        for key, summary in sorted(trace_summaries.items()):
            rows.append(
                [
                    key,
                    summary.get("n_traces"),
                    summary.get("mean_n_llm_calls"),
                    summary.get("mean_n_tool_calls"),
                    summary.get("mean_n_invalid_tool_calls"),
                    summary.get("mean_tool_success_rate"),
                    summary.get("mean_total_tokens"),
                    summary.get("mean_kg_context_chars"),
                    summary.get("coverage_honesty"),
                ]
            )
        parts.append(_md_table(headers, rows))
        parts.append("")

    return "\n".join(parts)
