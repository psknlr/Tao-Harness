"""LLM-as-judge, kept as a layer of its own.

Separating the judge from the automatic metrics matters for three reasons:

* it runs over recorded traces, so re-judging costs no generation;
* the judge model is pinned independently of the models under test, and the
  same judge scores every arm, so judge quality is a constant, not a variable;
* a judge failure degrades to "unscored" rather than to a silent zero, which
  would otherwise be indistinguishable from a genuinely bad answer.

The rubric lives in ``tcm_agent/prompts/judge_sdt.txt`` and is folded into the
prompt fingerprint, so changing it invalidates comparability exactly as
changing a task prompt does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tcm_agent.parsing import coerce_str, extract_json_object
from tcm_agent.prompts import load_prompt
from tcm_models.base import DecodeParams, LLMClient, Message

JUDGE_FIELDS = ("clinical_information", "pathogenesis", "explanation")
JUDGE_MAX = 4.0


@dataclass
class JudgeScore:
    case_id: str
    scores: Dict[str, Optional[float]]
    hallucination: Optional[bool]
    comment: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"case_id": self.case_id, "comment": self.comment}
        for key, value in self.scores.items():
            payload[f"judge_{key}"] = value
            if value is not None:
                payload[f"judge_{key}_norm"] = value / JUDGE_MAX
        if self.hallucination is not None:
            payload["judge_hallucination"] = float(self.hallucination)
        if self.error:
            payload["judge_error"] = self.error
        return payload


class SDTJudge:
    """Scores the three free-text SDT steps with a pinned judge model."""

    def __init__(self, model: LLMClient, *, decode: Optional[DecodeParams] = None):
        self.model = model
        # temperature 0 and a low token budget: the judge emits a fixed-shape
        # object, and sampling variance in the judge would leak straight into
        # the between-model comparison it is meant to arbitrate
        self.decode = decode or DecodeParams(temperature=0.0, max_tokens=512)
        self.system = load_prompt("judge_sdt")

    def score(
        self,
        case_id: str,
        prediction: Optional[Mapping[str, Any]],
        gold: Mapping[str, Any],
    ) -> JudgeScore:
        if not prediction:
            return JudgeScore(case_id, {k: 0.0 for k in JUDGE_FIELDS}, hallucination=None)

        payload = {
            "标准答案": {
                "clinical_information": coerce_str(gold.get("clinical_information")),
                "pathogenesis": coerce_str(gold.get("pathogenesis")),
                "syndrome": coerce_str(gold.get("syndrome")),
                "explanation": coerce_str(gold.get("explanation")),
            },
            "模型作答": {
                key: coerce_str(prediction.get(key))
                for key in ("clinical_information", "pathogenesis", "syndrome", "explanation")
            },
            "病例原文": coerce_str(gold.get("clinical_data"))[:2000],
        }
        messages = [
            Message("system", self.system),
            Message("user", json.dumps(payload, ensure_ascii=False)),
        ]
        completion = self.model.generate(messages, self.decode)
        if completion.error:
            return JudgeScore(
                case_id, {k: None for k in JUDGE_FIELDS}, None, error=completion.error
            )
        outcome = extract_json_object(completion.text)
        if outcome.value is None:
            return JudgeScore(
                case_id,
                {k: None for k in JUDGE_FIELDS},
                None,
                error="judge output unparseable",
            )
        scores: Dict[str, Optional[float]] = {}
        for key in JUDGE_FIELDS:
            raw = outcome.value.get(key)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                scores[key] = None
                continue
            scores[key] = max(0.0, min(JUDGE_MAX, value))
        hallucination = outcome.value.get("hallucination")
        return JudgeScore(
            case_id=case_id,
            scores=scores,
            hallucination=bool(hallucination) if isinstance(hallucination, bool) else None,
            comment=coerce_str(outcome.value.get("comment"))[:300],
        )

    def score_many(
        self,
        pairs: Sequence[tuple],
    ) -> List[JudgeScore]:
        return [self.score(case_id, prediction, gold) for case_id, prediction, gold in pairs]


def aggregate_judge(scores: Sequence[JudgeScore]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_judged": len(scores)}
    for key in JUDGE_FIELDS:
        values = [s.scores.get(key) for s in scores]
        numeric = [v for v in values if isinstance(v, (int, float))]
        out[f"judge_{key}"] = sum(numeric) / len(numeric) if numeric else None
        out[f"judge_{key}_norm"] = (
            (sum(numeric) / len(numeric)) / JUDGE_MAX if numeric else None
        )
    flags = [s.hallucination for s in scores if s.hallucination is not None]
    out["hallucination_rate"] = sum(1 for f in flags if f) / len(flags) if flags else None
    out["n_judge_errors"] = sum(1 for s in scores if s.error)
    return out
