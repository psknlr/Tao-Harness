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

import math
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
# TCM-CP (clinical pathway)
# --------------------------------------------------------------------------- #

#: Option letters in CP6 transition items.
CP6_CONTINUE, CP6_ADVANCE, CP6_EXIT, CP6_INSUFFICIENT = "A", "B", "C", "D"

#: Options that move a patient onward. Selecting any of these when the record
#: does not support it is the unsafe direction; staying too long is not.
_MOVE_ON = frozenset({CP6_ADVANCE, CP6_EXIT})

#: Gold answers for which moving the patient on is unsafe: the criteria are
#: unmet (continue), or the record does not settle the question at all
#: (insufficient evidence). Advancing on incomplete evidence is exactly the
#: failure the fourth state exists to catch.
_UNSAFE_WHEN_GOLD_IS = frozenset({CP6_CONTINUE, CP6_INSUFFICIENT})


def score_cp(
    prediction: Optional[Mapping[str, Any]], gold: Mapping[str, Any]
) -> Dict[str, Any]:
    """Score one TCM-CP item.

    Reports per-subtask, because the subtasks measure different things and a
    pooled pathway accuracy would hide which part of pathway execution a model
    is bad at. Multi-select action items additionally get precision/recall,
    since naming four of nine required actions is a different failure from
    naming four wrong ones.

    ``unsafe_transition`` is the endpoint worth watching: a transition item
    whose gold answer is "continue treatment" and whose prediction is "advance"
    or "discharge" is a recommendation to move a patient on when the recorded
    criteria are not met.
    """
    subtask = str(gold.get("subtask") or "unknown")
    gold_letters = set(normalise_options(gold.get("answer")))
    scores: Dict[str, Any] = {"subtask": subtask, "is_multi": float(len(gold_letters) > 1)}

    if not prediction:
        scores.update({"answered": 0.0, "exact": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
        if subtask == "CP6_transition_decision":
            # no recommendation is not an unsafe recommendation
            scores["unsafe_transition"] = 0.0
            scores["missed_uncertainty"] = float(CP6_INSUFFICIENT in gold_letters)
        return scores

    predicted = set(normalise_options(prediction.get("answer")))
    overlap = len(predicted & gold_letters)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold_letters) if gold_letters else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    scores.update(
        {
            "answered": 1.0,
            "exact": float(bool(predicted) and predicted == gold_letters),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_predicted": float(len(predicted)),
        }
    )
    scores[f"{subtask}_exact"] = scores["exact"]

    if subtask == "CP6_transition_decision":
        # Any unsafe option counts, not just the first one alphabetically.
        # Reading only the first sorted letter meant an answer of ["A","C"] --
        # continue *and* discharge -- scored as safe because "A" sorts first,
        # while it is precisely the ambiguous recommendation a reviewer would
        # call dangerous.
        scores["unsafe_transition"] = float(
            bool(gold_letters & _UNSAFE_WHEN_GOLD_IS) and bool(predicted & _MOVE_ON)
        )
        scores["missed_uncertainty"] = float(
            CP6_INSUFFICIENT in gold_letters and CP6_INSUFFICIENT not in predicted
        )
    return scores


CP_PRIMARY = "exact"
CP_METRICS = (
    "answered",
    "exact",
    "f1",
    "precision",
    "recall",
    "unsafe_transition",
    "missed_uncertainty",
)


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


#: Fields voted per-option (letters) rather than as whole strings.
_OPTION_FIELDS = ("pathogenesis_answer", "syndrome_answer", "answer")


def consensus_prediction(
    predictions: Sequence[Optional[Mapping[str, Any]]],
    *,
    threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Combine ``n`` samples of one case into a single self-consistent answer.

    Multiple-choice fields are voted **per option**: an option is kept when it
    appears in at least ``threshold`` of the answered samples. Voting on the
    whole letter-set instead would collapse to the modal set and discard the
    agreement structure, which matters here because most SDT items have more
    than one correct option and samples typically agree on the core option
    while disagreeing on the margin.

    List fields (clinical information) take the union of items appearing in at
    least ``threshold`` of samples -- task 1 is scored by recall of gold items,
    so an item several samples agree on is worth keeping. Free text takes the
    medoid: the sample with the highest mean character-bigram overlap with the
    others, i.e. the most representative rather than an incoherent splice.
    """
    answered = [p for p in predictions if p]
    if not answered:
        return None
    n = len(answered)
    # A strict majority: with n=2 and threshold 0.5, ceil(1.0) = 1 would let a
    # single vote carry an option, making the "vote" a union. Requiring more
    # than half keeps it a majority at every n; use an odd `samples` to avoid
    # ties in the first place.
    cutoff = max(1, int(math.floor(threshold * n)) + 1) if n > 1 else 1

    out: Dict[str, Any] = {}
    for field_name in _OPTION_FIELDS:
        tally: Dict[str, int] = {}
        present = False
        for prediction in answered:
            if field_name not in prediction:
                continue
            present = True
            for letter in set(normalise_options(prediction.get(field_name))):
                tally[letter] = tally.get(letter, 0) + 1
        if not present:
            continue
        kept = sorted(letter for letter, count in tally.items() if count >= cutoff)
        # never return an empty answer when the samples did answer: fall back
        # to the single most-agreed option
        if not kept and tally:
            kept = [max(sorted(tally), key=lambda letter: tally[letter])]
        out[field_name] = kept

    tally_items: Dict[str, int] = {}
    saw_list = False
    for prediction in answered:
        if "clinical_information" not in prediction:
            continue
        saw_list = True
        for item in set(coerce_list(prediction.get("clinical_information"))):
            tally_items[item] = tally_items.get(item, 0) + 1
    if saw_list:
        out["clinical_information"] = [
            item for item, count in sorted(tally_items.items()) if count >= cutoff
        ] or sorted(tally_items, key=lambda i: -tally_items[i])[:5]

    texts = [coerce_str(p.get("explanation")) for p in answered]
    texts = [t for t in texts if t]
    if texts:
        out["explanation"] = _medoid(texts)

    out["_n_samples"] = n
    out["_vote_threshold"] = threshold
    return out


def _medoid(texts: Sequence[str]) -> str:
    """The most representative of several free-text answers."""
    if len(texts) == 1:
        return texts[0]
    best, best_score = texts[0], -1.0
    for candidate in texts:
        score = sum(char_f1(candidate, other) for other in texts if other is not candidate)
        if score > best_score:
            best, best_score = candidate, score
    return best
