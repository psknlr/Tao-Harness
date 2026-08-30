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

from .scorers import (
    CP_METRICS,
    CP_PRIMARY,
    PA_METRICS,
    PA_PRIMARY,
    SDT_METRICS,
    SDT_PRIMARY,
    ScoredItem,
    aggregate,
)
from .stats import (
    PairedResult,
    cluster_paired_bootstrap,
    holm_bonferroni,
    is_binary,
    mcnemar,
    paired_bootstrap,
    paired_test,
    spearman,
    wilcoxon_signed_rank,
)

#: Multiplicity is controlled within prespecified hypothesis families, not
#: across the whole experiment. SDT tests an effectiveness claim (does the
#: graph improve clinical reasoning) and PA a safety claim (does it reduce
#: rule and knowledge errors); they are separate questions asked of separate
#: instruments, and pooling them would cost power on both to control an error
#: rate no reviewer is asking about. The family is named in every table so the
#: correction and the prose cannot drift apart.
HYPOTHESIS_FAMILIES: Mapping[str, str] = {
    "sdt": "SDT effectiveness family",
    "pa": "PA safety family",
    "cp": "Clinical-pathway family",
}

CONDITION_LABELS = {
    "M0": "M0 Base LLM",
    "M1": "M1 Structured",
    "M2": "M2 KG-RAG (static)",
    "M2C": "M2C Iterative, static KG, no tools (control)",
    "M3": "M3 KG-Agent",
    "M3C": "M3C Sham revision (control)",
    "M4": "M4 KG-Agent + Verify",
}

#: Each contrast names what it *isolates*, not what it is convenient to call it.
#: In particular M0→M3 confounds prompt structure, retrieval and agency, so it
#: is labelled as the whole-scaffold effect rather than a KG-Agent effect; the
#: interpretable KG and agency terms are M1→M2 and M2C→M3.
CONTRASTS = (
    ("M0", "M1", "prompt structure alone"),
    ("M1", "M2", "static KG evidence"),
    ("M2", "M3", "agency + extra compute (confounded)"),
    ("M2C", "M3", "agentic retrieval over the same KG, compute-matched"),
    ("M3", "M4", "verification + extra revision (confounded)"),
    ("M3C", "M4", "verification content, compute-matched"),
    ("M0", "M3", "whole scaffold (not a KG-only effect)"),
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


def paired_clusters(
    index: Mapping[Tuple[str, str], Mapping[str, ScoredItem]],
    model: str,
    left: str,
    right: str,
    metric: str,
    cluster_field: str,
) -> List[str]:
    """Cluster label per paired observation, aligned with ``paired_vectors``.

    Alignment matters more than it looks: the bootstrap resamples clusters and
    indexes into the value vectors positionally, so a label list built from a
    different filter than the values silently mislabels every observation
    after the first drop.
    """
    return [
        str(li.metrics.get(cluster_field) or case_id)
        for case_id, li, _ in _usable_pairs(index, model, left, right, metric)
    ]


def _usable_pairs(
    index: Mapping[Tuple[str, str], Mapping[str, ScoredItem]],
    model: str,
    left: str,
    right: str,
    metric: str,
    *,
    stratum: Optional[str] = None,
) -> List[Tuple[str, ScoredItem, ScoredItem]]:
    """Case ids both conditions scored, minus the ones that are not comparable.

    A case whose branches broke compute parity is dropped rather than scored.
    The compute-matched contrasts exist precisely to hold test-time compute
    fixed; including a pair where it was not fixed reintroduces the confound
    the contrast was built to remove, and does it invisibly.  ``stratum``
    additionally restricts to cases where the M4 verification pass reached a
    given depth, so the gain can be read separately over items a checker
    actually adjudicated and items where the pass had nothing to say.
    """
    a = index.get((model, left), {})
    b = index.get((model, right), {})
    out: List[Tuple[str, ScoredItem, ScoredItem]] = []
    for case_id in sorted(set(a) & set(b)):
        li, ri = a[case_id], b[case_id]
        if li.trace_metrics.get("parity_error") or ri.trace_metrics.get("parity_error"):
            continue
        if stratum is not None:
            observed = ri.trace_metrics.get("verification_stratum") or li.trace_metrics.get(
                "verification_stratum"
            )
            if observed != stratum:
                continue
        left_value = li.metrics.get(metric)
        right_value = ri.metrics.get(metric)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            out.append((case_id, li, ri))
    return out


def dropped_for_parity(
    items: Sequence[ScoredItem], left: str, right: str
) -> Dict[str, int]:
    """How many pairs each model lost to a compute-parity break."""
    index = index_items(items)
    out: Dict[str, int] = {}
    for model in sorted({i.model_key for i in items}):
        a = index.get((model, left), {})
        b = index.get((model, right), {})
        out[model] = sum(
            1
            for case_id in set(a) & set(b)
            if a[case_id].trace_metrics.get("parity_error")
            or b[case_id].trace_metrics.get("parity_error")
        )
    return out


def paired_vectors(
    index: Mapping[Tuple[str, str], Mapping[str, ScoredItem]],
    model: str,
    left: str,
    right: str,
    metric: str,
    *,
    stratum: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """Aligned metric vectors over the cases both conditions comparably attempted."""
    pairs = _usable_pairs(index, model, left, right, metric, stratum=stratum)
    xs = [float(li.metrics[metric]) for _, li, _ in pairs]
    ys = [float(ri.metrics[metric]) for _, _, ri in pairs]
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
    items: Sequence[ScoredItem],
    dataset: str,
    metric: str,
    *,
    binary: Optional[bool] = None,
    contrasts: Sequence[Tuple[str, str, str]] = CONTRASTS,
    clusters: Optional[str] = None,
) -> Tuple[str, Dict[str, PairedResult]]:
    """Paired condition contrasts per model, with multiplicity control.

    The test is chosen from the data, not asserted by the caller: exact McNemar
    for genuinely binary outcomes, paired bootstrap for continuous ones, with
    Wilcoxon reported beside the bootstrap as a distribution-free check.
    Forcing McNemar onto a continuous score -- as an earlier version of this
    file did to the SDT composite -- throws away the magnitude of every paired
    difference and tests a hypothesis the metric does not express.

    ``binary`` remains available to override the detection, but the default of
    ``None`` (detect) is what the report uses.
    """
    scoped = [i for i in items if i.dataset == dataset]
    index = index_items(scoped)
    models = sorted({i.model_key for i in scoped})

    results: Dict[str, PairedResult] = {}
    rows: List[List[Any]] = []
    for model in models:
        for left, right, label in contrasts:
            xs, ys = paired_vectors(index, model, left, right, metric)
            if not xs:
                continue
            if clusters is not None:
                # CP items cluster by disease; see cluster_paired_bootstrap
                labels = paired_clusters(index, model, left, right, metric, clusters)
                result = cluster_paired_bootstrap(xs, ys, labels)
            elif binary is None:
                result = paired_test(xs, ys)
            else:
                result = mcnemar(xs, ys) if binary else paired_bootstrap(xs, ys)
            key = f"{dataset}|{model}|{left}->{right}"
            results[key] = result
            interval = (
                f"[{result.ci_low:+.3f}, {result.ci_high:+.3f}]"
                if result.ci_low == result.ci_low
                else "–"
            )
            secondary = (
                wilcoxon_signed_rank(xs, ys).p_value
                if result.test.startswith("paired_bootstrap")
                else None
            )
            rows.append(
                [
                    model,
                    f"{left}→{right}",
                    label,
                    result.n,
                    result.mean_a,
                    result.mean_b,
                    result.delta,
                    interval,
                    result.test,
                    result.p_value,
                    secondary,
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
        "95% CI",
        "test",
        "p",
        "p (Wilcoxon)",
        "p (Holm)",
        "sig",
    ]
    return _md_table(headers, rows), results


def leakage_table(items: Sequence[ScoredItem]) -> str:
    """How often agents probed the graph for their task-2 answer options.

    The static context no longer offers a pathogenesis lookup, but an agent can
    type an option into the search tool. The prompt forbids it; this reports
    whether it happened anyway, per model and arm. Publishing the rate is what
    lets a reader judge the leakage claim instead of taking it on trust -- and
    if the rate is non-trivial, the sensitivity row below says what the result
    looks like with those cases removed.
    """
    scoped = [i for i in items if i.dataset == "sdt"]
    rows: List[List[Any]] = []
    for model in sorted({i.model_key for i in scoped}):
        for condition in [c for c in CONDITION_LABELS if any(
            i.model_key == model and i.condition == c for i in scoped
        )]:
            bucket = [
                i for i in scoped if i.model_key == model and i.condition == condition
            ]
            rates = [
                float(i.trace_metrics["pathogenesis_probe_rate"])
                for i in bucket
                if isinstance(i.trace_metrics.get("pathogenesis_probe_rate"), (int, float))
            ]
            if not rates:
                continue
            probed = [r for r in rates if r > 0]
            clean = [
                float(i.metrics.get("sdt_composite", 0.0))
                for i in bucket
                if not i.trace_metrics.get("pathogenesis_probe_rate")
            ]
            everything = [float(i.metrics.get("sdt_composite", 0.0)) for i in bucket]
            rows.append(
                [
                    model,
                    condition,
                    len(bucket),
                    len(probed),
                    len(probed) / len(rates) if rates else None,
                    sum(everything) / len(everything) if everything else None,
                    sum(clean) / len(clean) if clean else None,
                ]
            )
    if not rows:
        return ""
    return _md_table(
        [
            "model", "arm", "cases", "cases that probed", "probe rate",
            "composite (all)", "composite (non-probing only)",
        ],
        rows,
    )


#: Below this many paired cases a stratum is described but not tested: a
#: significant-looking p over eight items is noise dressed as a finding.
MIN_STRATUM_N = 20

#: How deep the M4 verification pass got on an item, worst-supported last.
VERIFICATION_STRATA: Tuple[Tuple[str, str], ...] = (
    ("deterministic", "a rule checker adjudicated the answer"),
    ("audit_only", "no checker applied; the prose was audited for over-claiming"),
    ("not_applicable", "nothing to check; the revision turn ran for parity only"),
)


def verification_stratum_table(
    items: Sequence[ScoredItem], dataset: str, metric: str
) -> str:
    """M3C→M4 split by how much the verification pass could actually check.

    This is the test that keeps the M4 claim honest.  M4 always takes a
    revision turn now, even on items no deterministic checker adjudicates, so
    the arm contains two different treatments wearing one label: real
    verification evidence, and a bare prompt to look again.  If the M4 gain is
    concentrated in the ``not_applicable`` stratum then what the experiment
    measured is a second turn, and the verification story does not survive.
    Reporting the split makes that visible instead of leaving it for a
    reviewer to suspect.

    Strata with fewer than ``MIN_STRATUM_N`` pairs are shown with their n and
    no test: an underpowered stratum tempts exactly the over-reading this
    table exists to prevent.
    """
    scoped = [i for i in items if i.dataset == dataset]
    index = index_items(scoped)
    rows: List[List[Any]] = []
    for model in sorted({i.model_key for i in scoped}):
        for stratum, gloss in VERIFICATION_STRATA:
            xs, ys = paired_vectors(index, model, "M3C", "M4", metric, stratum=stratum)
            if not xs:
                continue
            mean_a = sum(xs) / len(xs)
            mean_b = sum(ys) / len(ys)
            if len(xs) >= MIN_STRATUM_N:
                result = paired_test(xs, ys)
                interval = (
                    f"[{result.ci_low:+.3f}, {result.ci_high:+.3f}]"
                    if result.ci_low == result.ci_low
                    else "–"
                )
                p_value: Any = result.p_value
            else:
                interval, p_value = "–", None
            rows.append(
                [model, stratum, gloss, len(xs), mean_a, mean_b, mean_b - mean_a,
                 interval, p_value]
            )
    if not rows:
        return ""
    return _md_table(
        ["model", "stratum", "what the pass could do", "n", "M3C", "M4", "Δ", "95% CI", "p"],
        rows,
    )


def compute_parity_table(items: Sequence[ScoredItem]) -> str:
    """Did each control actually spend what the arm it matches spent?

    A compute-matched control only licenses its conclusion if the match held in
    practice. If M2C used half as many model calls as M3, then M2C→M3 is still
    partly a compute contrast and must be reported as such. This table is the
    evidence for -- or against -- the matching claim, and belongs in the paper
    beside the contrasts that depend on it.
    """
    pairs = [("M2C", "M3"), ("M3C", "M4")]
    rows: List[List[Any]] = []
    for control, arm in pairs:
        for model in sorted({i.model_key for i in items}):
            for dataset in sorted({i.dataset for i in items}):
                def _mean(condition: str, key: str) -> Optional[float]:
                    values = [
                        float(i.trace_metrics[key])
                        for i in items
                        if i.model_key == model
                        and i.dataset == dataset
                        and i.condition == condition
                        and isinstance(i.trace_metrics.get(key), (int, float))
                    ]
                    return sum(values) / len(values) if values else None

                control_calls = _mean(control, "n_llm_calls")
                arm_calls = _mean(arm, "n_llm_calls")
                if control_calls is None or arm_calls is None:
                    continue
                control_tokens = _mean(control, "total_tokens")
                arm_tokens = _mean(arm, "total_tokens")
                ratio = control_calls / arm_calls if arm_calls else None
                # Means can match while individual pairs do not, so the
                # per-case count is the binding check; the ratio is only a
                # readable summary beside it.
                scope = [i for i in items if i.model_key == model and i.dataset == dataset]
                broken = dropped_for_parity(scope, control, arm).get(model, 0)
                verdict = (
                    "MISMATCH"
                    if broken
                    else ("ok" if ratio is not None and 0.8 <= ratio <= 1.25 else "MISMATCH")
                )
                rows.append(
                    [
                        dataset,
                        model,
                        f"{control} vs {arm}",
                        control_calls,
                        arm_calls,
                        ratio,
                        control_tokens,
                        arm_tokens,
                        broken,
                        verdict,
                    ]
                )
    if not rows:
        return ""
    return _md_table(
        [
            "dataset", "model", "pair", "control calls", "arm calls",
            "call ratio", "control tokens", "arm tokens",
            "per-case breaks (dropped)", "parity",
        ],
        rows,
    )


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


#: Groundability verdict per PA rule family, from ``scripts/kg_coverage.py``.
#: Kept here so the report can split PA accuracy by whether the graph could
#: possibly have helped -- pooling grounded and ungroundable families dilutes
#: any real effect and invites the wrong conclusion.
RULE_GROUNDABILITY: Mapping[str, str] = {
    "A-002": "grounded", "A-006": "grounded", "N-003": "grounded",
    "A-004": "partial", "A-005": "partial", "A-007": "partial",
    "A-008": "partial", "N-007": "partial", "N-009": "partial",
    "A-001": "not grounded", "A-003": "not grounded", "A-009": "not grounded",
    "N-001": "not grounded", "N-002": "not grounded", "N-004": "not grounded",
    "N-005": "not grounded", "N-006": "not grounded", "N-008": "not grounded",
    "C-001": "not grounded",
}


def pa_rule_table(items: Sequence[ScoredItem], metric: str = "exact") -> str:
    """PA accuracy per rule family, per condition, with the graph's verdict."""
    scoped = [i for i in items if i.dataset == "pa"]
    if not scoped:
        return ""
    conditions = [c for c in CONDITION_LABELS if any(i.condition == c for i in scoped)]
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    for item in scoped:
        rule = str(item.metrics.get("rule_id") or "?").upper()
        value = item.metrics.get(metric)
        if isinstance(value, (int, float)):
            buckets[(rule, item.condition)].append(float(value))
        if item.condition == conditions[0]:
            counts[rule] += 1

    rows: List[List[Any]] = []
    for rule in sorted({r for r, _ in buckets}):
        row: List[Any] = [
            rule,
            RULE_GROUNDABILITY.get(rule, "?"),
            counts.get(rule) or len(buckets.get((rule, conditions[0]), [])),
        ]
        for condition in conditions:
            values = buckets.get((rule, condition), [])
            row.append(sum(values) / len(values) if values else None)
        rows.append(row)
    return _md_table(["rule", "graph verdict", "n"] + conditions, rows)


def pa_groundability_table(items: Sequence[ScoredItem], metric: str = "exact") -> str:
    """PA accuracy pooled by whether the graph can ground the rule family."""
    scoped = [i for i in items if i.dataset == "pa"]
    if not scoped:
        return ""
    conditions = [c for c in CONDITION_LABELS if any(i.condition == c for i in scoped)]
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    sizes: Dict[str, set] = defaultdict(set)
    for item in scoped:
        rule = str(item.metrics.get("rule_id") or "?").upper()
        verdict = RULE_GROUNDABILITY.get(rule, "unknown")
        value = item.metrics.get(metric)
        if isinstance(value, (int, float)):
            buckets[(verdict, item.condition)].append(float(value))
        sizes[verdict].add(item.case_id)

    rows: List[List[Any]] = []
    for verdict in ("grounded", "partial", "not grounded", "unknown"):
        if verdict not in sizes:
            continue
        row: List[Any] = [verdict, len(sizes[verdict])]
        for condition in conditions:
            values = buckets.get((verdict, condition), [])
            row.append(sum(values) / len(values) if values else None)
        first = buckets.get((verdict, conditions[0]), [])
        last = buckets.get((verdict, conditions[-1]), [])
        row.append(
            (sum(last) / len(last)) - (sum(first) / len(first)) if first and last else None
        )
        rows.append(row)
    return _md_table(
        ["graph verdict", "items"] + conditions + [f"Δ {conditions[0]}→{conditions[-1]}"], rows
    )


def explicit_vs_implicit_table(items: Sequence[ScoredItem]) -> str:
    """RQ4: gain on graph-explicit knowledge versus graph-constrained reasoning.

    PA accuracy depends on knowledge the graph encodes as entities and edges.
    SDT syndrome sits between: the graph names syndromes but not symptoms.
    SDT pathogenesis is the pure case -- the graph has no such entity at all,
    so any gain there comes only from the graph narrowing the reasoning space.
    """
    specs = [
        ("PA (all rules)", "pa", "exact", "explicit where the rule is grounded at all"),
        (
            "SDT task 3 syndrome",
            "sdt",
            "task3_syndrome",
            "partial — syndromes are entities, symptoms are not (~32% of options in graph)",
        ),
        (
            "SDT task 2 pathogenesis",
            "sdt",
            "task2_pathogenesis",
            "implicit — pathogenesis is not a graph entity at all",
        ),
        (
            "SDT task 1 clinical info",
            "sdt",
            "task1_clinical_information",
            "unaided — extraction from the case text alone",
        ),
        (
            "SDT task 4 explanation",
            "sdt",
            "task4_explanation",
            "unaided — free-text ROUGE-L against the case's own commentary",
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


def _framework_blocks(
    framework: Optional[Mapping[str, Any]]
) -> Dict[str, Mapping[str, Any]]:
    """Normalise the provenance argument to ``{dataset: block}``."""
    if not framework:
        return {}
    if all(isinstance(v, Mapping) for v in framework.values()):
        return {str(k): v for k, v in framework.items()}
    return {str(framework.get("task") or "run"): framework}


def build_report(
    items: Sequence[ScoredItem],
    *,
    trace_summaries: Optional[Mapping[str, Any]] = None,
    framework: Optional[Mapping[str, Any]] = None,
    title: str = "TCM-KG Agent — results",
) -> str:
    """Assemble the full Markdown report."""
    parts: List[str] = [f"# {title}", ""]

    # ``framework`` is ``{dataset: provenance}``; a single flat block is still
    # accepted so older callers and tests keep working.
    per_dataset = _framework_blocks(framework)
    if framework:
        parts.append("## Framework contract")
        parts.append("")
        parts.append(
            "All arms share one frozen framework; only `model.generate()` differs. "
            "Each dataset carries its own provenance: one block cannot attest to "
            "a run it did not describe."
        )
        parts.append("")
        forced = sorted(k for k, v in per_dataset.items() if v.get("scored_with_allow_drift"))
        if forced:
            parts.append(
                f"> **Scored with `--allow-drift`: {', '.join(k.upper() for k in forced)}.** "
                f"The traces did not match the frozen manifest for these datasets. "
                f"See the `drift` list in the block below; the numbers are not "
                f"reproducible from the recorded provenance."
            )
            parts.append("")
        parts.append("```json")
        parts.append(json.dumps(dict(framework), ensure_ascii=False, indent=2))
        parts.append("```")
        parts.append("")

    datasets = sorted({i.dataset for i in items})
    for dataset in datasets:
        metrics = {"sdt": SDT_METRICS, "pa": PA_METRICS, "cp": CP_METRICS}.get(
            dataset, PA_METRICS
        )
        primary = {"sdt": SDT_PRIMARY, "pa": PA_PRIMARY, "cp": CP_PRIMARY}.get(
            dataset, PA_PRIMARY
        )
        parts.append(f"## {dataset.upper()}")
        parts.append("")
        if dataset == "cp":
            parts.append(
                "> **Instrument-capability benchmark.** TCM-CP's gold answers are "
                "derived from the same knowledge graph the KG arms consult, so a "
                "KG advantage here is expected by construction and demonstrates "
                "faithful pathway execution, not clinical effectiveness. Its "
                "contrasts must not be pooled with, or reported alongside, the "
                "SDT and PA effectiveness results."
            )
            parts.append("")
        parts.append(f"### Main results (primary metric: `{primary}`)")
        parts.append("")
        parts.append(main_table(items, dataset, metrics))
        parts.append("")
        table, results = contrast_table(
            items, dataset, primary, clusters="disease" if dataset == "cp" else None
        )
        tests_used = sorted({r.test for r in results.values()}) or ["–"]
        family = HYPOTHESIS_FAMILIES.get(dataset, f"{dataset} family")
        parts.append("### Paired condition contrasts")
        parts.append("")
        parts.append(
            f"Test chosen from the data: {', '.join(tests_used)}. "
            f"Multiplicity controlled by Holm–Bonferroni within the "
            f"**{family}** ({len(results)} comparisons); the SDT and PA families "
            f"are corrected separately because they test different, "
            f"prespecified claims."
        )
        parts.append("")
        parts.append(table)
        parts.append("")
        strata = verification_stratum_table(items, dataset, primary)
        if strata:
            parts.append("### Is the M4 gain verification, or just a second turn?")
            parts.append("")
            parts.append(
                "M4 takes its revision turn on **every** case, including the ones "
                "no deterministic checker adjudicates — otherwise it would spend "
                "one model call fewer than M3C on exactly those cases and the "
                "compute match would fail where it matters most. The price is "
                "that the arm mixes two treatments, so the gain is split by how "
                "much the verification pass could actually check. A gain "
                "concentrated in `not_applicable` is a second-turn effect and "
                "must be reported as one."
            )
            parts.append("")
            parts.append(strata)
            parts.append("")
        comp_table, comp_stats = compensation_table(items, dataset, primary)
        parts.append("### Does the framework compensate weaker models?")
        parts.append("")
        parts.append(comp_table)
        parts.append("")
        parts.append(f"Spearman ρ(base, Δ) = `{comp_stats['spearman_rho_base_vs_delta']}` "
                     f"over {comp_stats['n_models']} models — {comp_stats['interpretation']}.")
        parts.append("")

    leakage = leakage_table(items)
    if leakage:
        parts.append("## Option-probing leakage check (SDT)")
        parts.append("")
        parts.append(
            "Task-2 pathogenesis options are never looked up for the model, and "
            "the agent prompt forbids searching them. This table reports whether "
            "agents did so anyway, and what the headline metric looks like with "
            "the probing cases removed — the sensitivity analysis a reviewer "
            "would otherwise have to ask for."
        )
        parts.append("")
        parts.append(leakage)
        parts.append("")

    parity = compute_parity_table(items)
    if parity:
        parts.append("## Compute parity of the control arms")
        parts.append("")
        parts.append(
            "M2C matches M3's turn budget while keeping M2's static KG context "
            "and withholding the tools -- so M2C→M3 isolates *agentic retrieval*, "
            "not the graph, which both arms have. M3C matches M4's turn count "
            "without the verification evidence. These contrasts are only "
            "interpretable if the match actually held, so the realised call and "
            "token counts are reported here alongside the number of pairs where "
            "parity broke per case. Any per-case break drops that pair from the "
            "contrast and marks the row `MISMATCH`; means alone can agree while "
            "individual pairs do not."
        )
        parts.append("")
        parts.append(parity)
        parts.append("")

    if any(i.dataset == "pa" for i in items):
        parts.append("## PA by rule family")
        parts.append("")
        parts.append(
            "Half the released PA items fall in rule families this graph cannot "
            "ground (A-003 dosage alone is 87 of 328). Pooling them with the "
            "grounded families dilutes any real effect, so both splits are shown."
        )
        parts.append("")
        parts.append(pa_groundability_table(items))
        parts.append("")
        parts.append(pa_rule_table(items))
        parts.append("")

    if len(datasets) > 1:
        parts.append("## Explicit knowledge vs graph-constrained reasoning")
        parts.append("")
        parts.append(explicit_vs_implicit_table(items))
        parts.append("")

    if trace_summaries:
        parts.append("## Agent behaviour (from traces)")
        parts.append("")
        parts.append(
            "`agent calls` are the tool calls the model chose; `verifier calls` "
            "are the ones the M4 verification pass made for it. Only the former "
            "reflect tool-use skill — the verifier calls the right checker by "
            "construction."
        )
        parts.append("")
        headers = [
            "dataset",
            "model",
            "condition",
            "traces",
            "LLM calls",
            "agent calls",
            "verifier calls",
            "invalid",
            "tool success",
            "tokens",
            "KG context chars",
            "coverage honesty",
        ]
        rows = []
        for key, summary in sorted(trace_summaries.items()):
            # keys are "dataset/model/condition"; older files wrote
            # "model/condition" and are still rendered rather than dropped
            parts_of_key = key.split("/")
            if len(parts_of_key) == 3:
                dataset, model, condition = parts_of_key
            elif len(parts_of_key) == 2:
                dataset, (model, condition) = "–", parts_of_key
            else:
                dataset, model, condition = "–", key, "–"
            rows.append(
                [
                    dataset,
                    model,
                    condition,
                    summary.get("n_traces"),
                    summary.get("mean_n_llm_calls"),
                    summary.get("mean_n_agent_tool_calls", summary.get("mean_n_tool_calls")),
                    summary.get("mean_n_verification_tool_calls"),
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
