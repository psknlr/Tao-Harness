"""Scoring.

Generation and scoring are separate phases: a scorer can be fixed and re-run
over recorded traces without re-billing a single token.  Every scorer therefore
takes a trace's ``final`` object plus the gold record, and returns a flat dict
of named metrics so new metrics can be added without invalidating old runs.

Two decisions are worth defending explicitly.

**Syndrome partial credit.**  Chinese syndrome names compose additively:
``痰阻血瘀，湿郁化热证`` is two conjuncts.  An answer recovering one of two is
genuinely closer than one recovering neither, and collapsing both to "wrong"
throws away the signal that most distinguishes a KG-grounded run from a
free-running one.  The headline metric stays strict exact match; atom-F1 is
reported beside it.

**Multi-select credit.**  The primary PA metric is strict set equality, as the
benchmark intends.  Pharmacist-exam partial credit (any wrong option scores
zero, otherwise credit is proportional to options recovered) is reported
alongside because with only 31 multiple-choice items strict accuracy is very
noisy, and the two together say whether a model is under-selecting or guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from tcm_agent.parsing import coerce_str
from tcm_agent.tasks import normalise_options
from tcm_kg.normalize import canonical_syndrome, char_ngrams, normalize_text, syndrome_atoms


def _bigrams(text: str) -> Set[str]:
    return set(char_ngrams(text, (2,)))


def char_f1(prediction: str, reference: str) -> float:
    """Character-bigram F1 -- a deterministic, segmenter-free text overlap score.

    Used as the auditable secondary metric for the free-text SDT steps; the
    LLM judge in :mod:`tcm_eval.judge` is the primary one.
    """
    pred, ref = _bigrams(prediction), _bigrams(reference)
    if not pred or not ref:
        return 1.0 if not pred and not ref else 0.0
    overlap = len(pred & ref)
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def set_f1(prediction: Iterable[str], reference: Iterable[str]) -> float:
    pred, ref = set(prediction), set(reference)
    if not pred or not ref:
        return 1.0 if not pred and not ref else 0.0
    overlap = len(pred & ref)
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# SDT
# --------------------------------------------------------------------------- #

#: Modifier characters that a model may add or drop without changing meaning.
_SYNDROME_TRIM_RE = re.compile(r"(证候|证型|证$|型$)")


def normalise_syndrome(name: str) -> str:
    text = canonical_syndrome(coerce_str(name))
    text = normalize_text(text)
    text = _SYNDROME_TRIM_RE.sub("", text)
    return re.sub(r"[\s,.;:()\[\]]", "", text)


def score_sdt(
    prediction: Optional[Mapping[str, Any]],
    gold: Mapping[str, Any],
    *,
    kg=None,
) -> Dict[str, Any]:
    """Score one SDT case."""
    if not prediction:
        empty: Dict[str, Any] = {
            "answered": 0.0,
            "syndrome_exact": 0.0,
            "syndrome_atom_f1": 0.0,
            "syndrome_alias_match": 0.0,
        }
        for step in ("clinical_information", "pathogenesis", "explanation"):
            if coerce_str(gold.get(step)):
                empty[f"{step}_f1"] = 0.0
        return empty

    pred_syndrome = coerce_str(prediction.get("syndrome"))
    gold_syndrome = coerce_str(gold.get("syndrome"))
    exact = float(
        bool(pred_syndrome)
        and normalise_syndrome(pred_syndrome) == normalise_syndrome(gold_syndrome)
    )

    pred_atoms = {normalise_syndrome(a) for a in syndrome_atoms(pred_syndrome)}
    gold_atoms = {normalise_syndrome(a) for a in syndrome_atoms(gold_syndrome)}
    pred_atoms.discard("")
    gold_atoms.discard("")
    atom_f1 = set_f1(pred_atoms, gold_atoms)

    alias = exact
    if not alias and kg is not None and pred_syndrome and gold_syndrome:
        alias = float(_alias_equivalent(kg, pred_syndrome, gold_syndrome))

    scores: Dict[str, Any] = {
        "answered": 1.0,
        "syndrome_exact": exact,
        "syndrome_atom_f1": atom_f1,
        "syndrome_alias_match": max(exact, alias),
        "predicted_syndrome": pred_syndrome,
        "gold_syndrome": gold_syndrome,
    }
    # A free-text step is scored only when the gold record actually annotates
    # it.  Scoring an absent reference as a perfect match would silently
    # inflate every model's average on whichever steps the split omits.
    for step in ("clinical_information", "pathogenesis", "explanation"):
        reference = coerce_str(gold.get(step))
        if reference:
            scores[f"{step}_f1"] = char_f1(coerce_str(prediction.get(step)), reference)
    return scores


def _alias_equivalent(kg, left: str, right: str) -> bool:
    """Two syndrome names that the graph itself treats as one entity."""
    left_nodes = kg.find_by_name(canonical_syndrome(left), ["Syndrome"])
    right_nodes = kg.find_by_name(canonical_syndrome(right), ["Syndrome"])
    if not left_nodes or not right_nodes:
        return False
    left_ids = {kg.canonical_id(n.id) for n in left_nodes}
    right_ids = {kg.canonical_id(n.id) for n in right_nodes}
    return bool(left_ids & right_ids)


SDT_PRIMARY = "syndrome_exact"
SDT_METRICS = (
    "answered",
    "syndrome_exact",
    "syndrome_alias_match",
    "syndrome_atom_f1",
    "clinical_information_f1",
    "pathogenesis_f1",
    "explanation_f1",
)


# --------------------------------------------------------------------------- #
# PA
# --------------------------------------------------------------------------- #


def score_pa(
    prediction: Optional[Mapping[str, Any]], gold: Mapping[str, Any]
) -> Dict[str, Any]:
    """Score one PA item."""
    gold_options = set(normalise_options(gold.get("answer")))
    is_multi = len(gold_options) > 1

    if not prediction:
        return {
            "answered": 0.0,
            "exact": 0.0,
            "partial_credit": 0.0,
            "jaccard": 0.0,
            "is_multi": float(is_multi),
            "n_predicted": 0.0,
            "gold_answer": sorted(gold_options),
        }

    pred_options = set(normalise_options(prediction.get("answer")))
    exact = float(bool(pred_options) and pred_options == gold_options)

    if not pred_options or (pred_options - gold_options):
        partial = 0.0  # any incorrect option forfeits credit, as in the exam rule
    else:
        partial = len(pred_options & gold_options) / len(gold_options)

    union = pred_options | gold_options
    jaccard = len(pred_options & gold_options) / len(union) if union else 1.0

    return {
        "answered": 1.0,
        "exact": exact,
        "partial_credit": partial,
        "jaccard": jaccard,
        "is_multi": float(is_multi),
        "n_predicted": float(len(pred_options)),
        "predicted_answer": sorted(pred_options),
        "gold_answer": sorted(gold_options),
        "rule_id": coerce_str(gold.get("rule_id")) or None,
    }


PA_PRIMARY = "exact"
PA_METRICS = ("answered", "exact", "partial_credit", "jaccard")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass
class ScoredItem:
    case_id: str
    dataset: str
    condition: str
    model_key: str
    sample: int
    metrics: Dict[str, Any]
    trace_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "condition": self.condition,
            "model_key": self.model_key,
            "sample": self.sample,
            "metrics": self.metrics,
            "trace_metrics": self.trace_metrics,
        }


def aggregate(items: Sequence[ScoredItem], keys: Sequence[str]) -> Dict[str, Optional[float]]:
    """Mean of each named metric over the items that carry it.

    A metric no item carries returns ``None`` rather than ``0.0``: a split that
    does not annotate a step is not the same as every model scoring zero on it,
    and reports render the two differently.
    """
    out: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [
            float(i.metrics[key])
            for i in items
            if key in i.metrics and isinstance(i.metrics[key], (int, float))
        ]
        out[key] = sum(values) / len(values) if values else None
    out["n"] = float(len(items))
    return out


def group_by(items: Sequence[ScoredItem], field_name: str) -> Dict[str, List[ScoredItem]]:
    groups: Dict[str, List[ScoredItem]] = {}
    for item in items:
        key = str(item.metrics.get(field_name) or getattr(item, field_name, "") or "unknown")
        groups.setdefault(key, []).append(item)
    return groups


def majority_vote(predictions: Sequence[Mapping[str, Any]], field_name: str) -> Optional[str]:
    """Self-consistency vote over ``n>1`` samples of the same case."""
    counts: Dict[str, int] = {}
    for prediction in predictions:
        value = coerce_str(prediction.get(field_name))
        if not value:
            continue
        key = (
            normalise_syndrome(value)
            if field_name == "syndrome"
            else ",".join(normalise_options(value))
        )
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda k: counts[k])
