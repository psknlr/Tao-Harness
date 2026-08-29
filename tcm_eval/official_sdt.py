"""Re-implementation of the official TCMEval-SDT scoring functions.

Ported line-for-line from the ``scripts/evaluate.py`` shipped with the dataset
(vendored at ``vendor/tcmeval_sdt/official_evaluate.py``), so that numbers this
harness reports are the numbers the benchmark's own evaluator would produce.
``tests/test_official_sdt.py`` asserts agreement against the vendored script on
the released answer files.

Three details of the official scorer are easy to get wrong and are preserved
deliberately, quirks included:

* **Task 1** de-duplicates the model's list before counting, and divides by the
  length of the *gold* list. It is recall of gold items, uncapped by precision:
  a model cannot be penalised for extracting extra findings, only for missing
  annotated ones.
* **Tasks 2 and 3** use ``max_score * correct / (len(gold) + n_wrong)``. This is
  *not* the exam rule where any wrong option forfeits credit -- a wrong pick
  dilutes rather than zeroes. It also means selecting all ten options scores
  ``len(gold)/10``, not zero, so guess-everything is weakly rewarded and worth
  reporting as a behavioural statistic.
* **Task 4** ROUGE-L is computed over raw characters (the official code takes
  ``len()`` of the strings directly), which for Chinese text is a
  character-level LCS.

The composite is ``0.2*T1 + 0.3*T2 + 0.4*T3 + 0.1*T4``.

One further quirk is reproduced because it changes the reported number. The
official evaluator reads both files with ``readlines()`` and never strips the
line terminator, so task 4 always compares ``prediction + "\n"`` against
``reference + "\n"``. The shared newline contributes 1 to the LCS, which means
an **empty explanation does not score zero** -- it scores ``2/(len(ref)+1)``,
around 0.009 on these files. Small, but a harness that claims to agree with the
official evaluator has to model it, so :func:`score_case` emulates the file
round-trip by default. Pass ``emulate_official_io=False`` for the clean
arithmetic.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

TASK_WEIGHTS: Mapping[str, float] = {
    "task1_clinical_information": 0.2,
    "task2_pathogenesis": 0.3,
    "task3_syndrome": 0.4,
    "task4_explanation": 0.1,
}


def clinical_info_extraction_eval(
    predicted: Sequence[str] | str, gold: Sequence[str]
) -> float:
    """Task 1: recall of gold clinical-information items, exact string match."""
    gold_list = list(gold)
    if not gold_list:
        return 0.0
    if isinstance(predicted, str):
        predicted_list = predicted.split(";")
    else:
        predicted_list = list(predicted)
    predicted_set = set(predicted_list)  # official code de-duplicates
    matched = sum(1 for item in gold_list if item in predicted_set)
    if matched == 0:
        return 0.0
    return matched / len(gold_list)


def score_proportional(
    predicted: Iterable[str], gold: Iterable[str], max_score: float = 1.0
) -> float:
    """Tasks 2 and 3: proportional credit over selected options.

    ``max_score * |correct ∩ selected| / (|correct| + |selected \\ correct|)``
    """
    gold_set = set(gold)
    predicted_set = set(predicted)
    correct = len(predicted_set & gold_set)
    wrong = len(predicted_set - gold_set)
    total = len(gold_set) + wrong
    if total == 0:
        return 0.0
    return max_score * correct / total


def lcs_length(a: str, b: str) -> int:
    """Longest common subsequence length, iterative with a rolling row."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        a_char = a[i - 1]
        for j in range(1, len(b) + 1):
            if a_char == b[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = current[j - 1] if current[j - 1] >= previous[j] else previous[j]
        previous = current
    return previous[len(b)]


#: Line terminator the official evaluator leaves attached to every field.
OFFICIAL_LINE_TERMINATOR = "\n"


def rouge_l(reference: str, candidate: str, beta: float = 1.0) -> float:
    """Task 4: character-level ROUGE-L F-measure.

    Symmetric at ``beta=1``, so the official code's swapped argument order is
    immaterial.
    """
    if not reference or not candidate:
        return 0.0
    length = lcs_length(reference, candidate)
    if length == 0:
        return 0.0
    recall = length / len(reference)
    precision = length / len(candidate)
    if recall + precision == 0:
        return 0.0
    return ((1 + beta**2) * recall * precision) / (recall + beta**2 * precision)


def score_case(
    prediction: Optional[Mapping[str, object]],
    gold: Mapping[str, object],
    *,
    emulate_official_io: bool = True,
) -> Dict[str, float]:
    """Score one SDT case across all four tasks plus the weighted composite.

    ``prediction`` uses the harness's normalised answer keys; an absent
    prediction scores zero on every task, matching the official evaluator's
    treatment of an empty submission field.
    """
    if not prediction:
        scores = {name: 0.0 for name in TASK_WEIGHTS}
        scores["answered"] = 0.0
        scores["sdt_composite"] = 0.0
        return scores

    task1 = clinical_info_extraction_eval(
        _as_list(prediction.get("clinical_information")),
        _as_list(gold.get("clinical_information_list")),
    )
    task2 = score_proportional(
        _as_list(prediction.get("pathogenesis_answer")),
        _as_list(gold.get("pathogenesis_letters")),
    )
    task3 = score_proportional(
        _as_list(prediction.get("syndrome_answer")),
        _as_list(gold.get("syndrome_letters")),
    )
    reference = str(gold.get("explanation_reference") or "")
    candidate = str(prediction.get("explanation") or "")
    if emulate_official_io:
        # the official evaluator compares the raw file lines, terminator included
        reference += OFFICIAL_LINE_TERMINATOR
        candidate += OFFICIAL_LINE_TERMINATOR
    task4 = rouge_l(reference, candidate)

    scores = {
        "answered": 1.0,
        "task1_clinical_information": task1,
        "task2_pathogenesis": task2,
        "task3_syndrome": task3,
        "task4_explanation": task4,
    }
    scores["sdt_composite"] = sum(TASK_WEIGHTS[name] * scores[name] for name in TASK_WEIGHTS)
    return scores


def _as_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


# --------------------------------------------------------------------------- #
# Official submission format
# --------------------------------------------------------------------------- #


def to_submission_line(case_id: str, prediction: Optional[Mapping[str, object]]) -> str:
    """Render one case in the official ``case@t1@t2@t3@t4`` format.

    Emitting this lets a run be scored by the benchmark's own evaluator without
    trusting this harness's arithmetic -- the point of vendoring the official
    script and of ``benchmark_runner submit``.
    """
    prediction = prediction or {}
    task1 = ";".join(_as_list(prediction.get("clinical_information")))
    task2 = ";".join(_as_list(prediction.get("pathogenesis_answer")))
    task3 = ";".join(_as_list(prediction.get("syndrome_answer")))
    task4 = str(prediction.get("explanation") or "").replace("@", " ").replace("\n", " ")
    return f"{case_id}@{task1}@{task2}@{task3}@{task4}"


def write_submission(path, rows: Sequence[tuple]) -> None:
    """Write an official submission file. ``rows`` is ``(case_id, prediction)``."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for case_id, prediction in rows:
            handle.write(to_submission_line(case_id, prediction) + "\n")
