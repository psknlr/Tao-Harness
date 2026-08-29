"""Scoring.

Generation and scoring are separate phases: a scorer can be fixed and re-run
over recorded traces without re-billing a single token.  Every scorer therefore
takes a trace's ``final`` object plus the gold record, and returns a flat dict
of named metrics so new metrics can be added without invalidating old runs.

Two decisions are worth defending explicitly.

**SDT follows the benchmark, not our preferences.**  TCMEval-SDT is a
four-task benchmark whose official ``evaluate.py`` weights clinical-information
recall 0.2, pathogenesis options 0.3, syndrome options 0.4 and a ROUGE-L
explanation 0.1.  Those rules -- including their quirks -- are implemented in
:mod:`tcm_eval.official_sdt` and verified against the vendored original.  Any
metric this module adds on top is diagnostic and named so it cannot be mistaken
for an official score.

**Multi-select credit differs between the two benchmarks, deliberately.**  SDT's
own rule dilutes on a wrong pick (``correct / (|gold| + n_wrong)``); PA has no
official scorer shipped with it, so it uses strict set equality as the primary
metric with exam-style partial credit reported alongside.  Mixing the two would
misreport one benchmark or the other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from tcm_agent.parsing import coerce_list, coerce_str
from tcm_agent.tasks import normalise_options
from tcm_kg.normalize import canonical_syndrome, char_ngrams, normalize_text, syndrome_atoms

from .official_sdt import score_case as official_score_case


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
# SDT -- scored by the benchmark's own rules
# --------------------------------------------------------------------------- #

#: Modifier characters a model may add or drop without changing meaning.
_SYNDROME_TRIM_RE = re.compile(r"(证候|证型|证$|型$)")


def normalise_syndrome(name: str) -> str:
    """Canonical surface form of a syndrome name (used by the KG-alias check)."""
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
    """Score one SDT case with the official four-task rules.

    The headline numbers come from :mod:`tcm_eval.official_sdt`, which agrees
    with the benchmark's own ``evaluate.py`` (see
    ``tests/test_official_sdt.py``). Everything added on top is diagnostic and
    named so it can never be confused with an official metric:

    ``n_pathogenesis_selected`` / ``n_syndrome_selected``
        How many options the model picked. The official rule dilutes rather
        than forfeits on a wrong pick, so selecting everything scores
        ``|gold|/10`` instead of zero. Reporting selection counts is what
        distinguishes a model that reasoned from one that hedged.
    ``syndrome_exact_set`` / ``pathogenesis_exact_set``
        Strict set equality, for readers who want an accuracy-shaped number
        beside the official proportional score.
    """
    scores: Dict[str, Any] = dict(official_score_case(prediction, gold))
    gold_pathogenesis = set(gold.get("pathogenesis_letters") or [])
    gold_syndrome = set(gold.get("syndrome_letters") or [])

    if not prediction:
        scores.update(
            {
                "n_pathogenesis_selected": 0.0,
                "n_syndrome_selected": 0.0,
                "pathogenesis_exact_set": 0.0,
                "syndrome_exact_set": 0.0,
            }
        )
        return scores

    predicted_pathogenesis = set(normalise_options(prediction.get("pathogenesis_answer")))
    predicted_syndrome = set(normalise_options(prediction.get("syndrome_answer")))
    scores.update(
        {
            "n_pathogenesis_selected": float(len(predicted_pathogenesis)),
            "n_syndrome_selected": float(len(predicted_syndrome)),
            "pathogenesis_exact_set": float(
                bool(predicted_pathogenesis) and predicted_pathogenesis == gold_pathogenesis
            ),
            "syndrome_exact_set": float(
                bool(predicted_syndrome) and predicted_syndrome == gold_syndrome
            ),
            "n_clinical_information": float(
                len(coerce_list(prediction.get("clinical_information")))
            ),
            "predicted_syndrome_letters": sorted(predicted_syndrome),
            "gold_syndrome_letters": sorted(gold_syndrome),
        }
    )
    return scores


SDT_PRIMARY = "sdt_composite"
SDT_METRICS = (
    "answered",
    "sdt_composite",
    "task1_clinical_information",
    "task2_pathogenesis",
    "task3_syndrome",
    "task4_explanation",
    "syndrome_exact_set",
    "n_syndrome_selected",
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
