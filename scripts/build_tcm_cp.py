"""Build TCM-CP: a clinical-pathway benchmark derived from the knowledge graph.

**Read this before using the numbers it produces.**

There is no published TCM clinical-pathway benchmark, so this builds one from
the pathway layer of the knowledge graph itself. That makes it *circular by
construction* with respect to the KG conditions: gold answers come from the
same graph the KG arms may consult, so a KG arm scoring higher than a
no-KG arm demonstrates that the agent can retrieve and apply pathway knowledge
correctly -- not that the knowledge graph makes a model clinically better.

TCM-CP is therefore an **instrument-capability benchmark**, and the harness
labels it as one everywhere it appears. It answers: can an agent execute a
staged pathway faithfully -- identify the stage, name the actions due, plan
treatment for the syndrome, and decide transitions against recorded criteria?
That is a real and separately interesting question, and it is the question the
SDT and PA benchmarks cannot ask. It is not evidence for the effectiveness
claims, and no contrast from TCM-CP belongs in the same table as those.

Two design choices reduce (but cannot remove) the circularity:

* Distractors for stage and action tasks are drawn from **other stages of the
  same disease**, so an agent cannot separate them by retrieving the disease
  alone; it has to land on the right stage.
* Case vignettes are rendered from stage content with the answer-bearing
  fields withheld, so the item is not a verbatim lookup of its own answer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcm_kg import load_kg
from tcm_kg.schema import EdgeType, NodeType
from tcm_kg.store import KGStore

#: A stage must carry enough content to make a fair item.
MIN_ACTIONS = 4
MIN_STAGES_PER_DISEASE = 2
N_OPTIONS = 6


def _stages_by_disease(kg: KGStore) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = defaultdict(list)
    for disease in kg.of_type(NodeType.DISEASE.value):
        stages = [t for _e, t in kg.neighbours(disease.id, {EdgeType.HAS_PATHWAY_STAGE.value})]
        stages = [s for s in stages if len(s.get("day_actions") or []) >= MIN_ACTIONS]
        if len(stages) >= MIN_STAGES_PER_DISEASE:
            out[disease.id] = sorted(stages, key=lambda s: (str(s.get("variant") or ""), s.get("order") or 0))
    return out


def _vignette(kg: KGStore, disease: Any, stage: Any, syndrome: Optional[Any]) -> str:
    """Render a case vignette from stage content, withholding the answers.

    Monitoring items describe what is being watched at this point in the
    admission, which situates the patient in time without naming the stage;
    the stage name, its day actions and its criteria are all withheld because
    they are what the item asks for.
    """
    parts = [f"患者因{disease.get('tcm_name') or disease.name}入院，进入该病种中医临床路径。"]
    if stage.get("variant"):
        parts.append(f"路径分支：{stage.get('variant')}。")
    if syndrome is not None:
        parts.append(f"入院辨证为{syndrome.name}。")
        if syndrome.sentence():
            parts.append(f"四诊所见：{syndrome.sentence()[:160]}")
    monitoring = [str(m) for m in (stage.get("monitoring_items") or [])][:4]
    if monitoring:
        parts.append("本次查房重点观察：" + "；".join(monitoring) + "。")
    return "".join(parts)


def _options(correct: str, pool: Sequence[str], rng: random.Random, n: int = N_OPTIONS) -> Tuple[Dict[str, str], List[str]]:
    distractors = [p for p in dict.fromkeys(pool) if p != correct]
    rng.shuffle(distractors)
    chosen = [correct] + distractors[: n - 1]
    rng.shuffle(chosen)
    options = {chr(ord("A") + i): text for i, text in enumerate(chosen)}
    answer = [k for k, v in options.items() if v == correct]
    return options, answer


def _multi_options(
    correct: Sequence[str], pool: Sequence[str], rng: random.Random, n: int = 8
) -> Tuple[Dict[str, str], List[str]]:
    correct = list(dict.fromkeys(correct))[: n // 2]
    distractors = [p for p in dict.fromkeys(pool) if p not in correct]
    rng.shuffle(distractors)
    chosen = correct + distractors[: max(0, n - len(correct))]
    rng.shuffle(chosen)
    options = {chr(ord("A") + i): text for i, text in enumerate(chosen)}
    answer = sorted(k for k, v in options.items() if v in correct)
    return options, answer


def build(kg: KGStore, *, seed: int = 20260829, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_disease = _stages_by_disease(kg)
    items: List[Dict[str, Any]] = []

    for disease_id, stages in sorted(by_disease.items()):
        disease = kg.node(disease_id)
        syndromes = [t for _e, t in kg.neighbours(disease_id, {EdgeType.HAS_SYNDROME.value})]
        syndrome = syndromes[0] if syndromes else None
        stage_names = [s.name for s in stages]

        for stage in stages:
            actions = [str(a) for a in (stage.get("day_actions") or [])]
            # distractors from *other stages of the same disease*: retrieving
            # the disease is not enough, the agent must land on the right stage
            other_actions = [
                str(a)
                for other in stages
                if other.id != stage.id
                for a in (other.get("day_actions") or [])
                if str(a) not in actions
            ]
            if len(other_actions) < 3:
                continue

            base = {
                "disease": disease.name,
                "stage_id": stage.id,
                "stage_name": stage.name,
                "variant": stage.get("variant"),
                "vignette": _vignette(kg, disease, stage, syndrome),
                "syndrome": syndrome.name if syndrome else None,
            }

            stage_options, stage_answer = _options(stage.name, stage_names, rng)
            items.append(
                {
                    **base,
                    "id": f"cp2::{stage.id}",
                    "subtask": "CP2_stage_identification",
                    "question": "根据上述病情描述，患者当前处于该临床路径的哪个阶段？",
                    "options": stage_options,
                    "answer": stage_answer,
                }
            )

            action_options, action_answer = _multi_options(actions, other_actions, rng)
            items.append(
                {
                    **base,
                    "id": f"cp3::{stage.id}",
                    "subtask": "CP3_stage_actions",
                    "question": (
                        f"患者处于「{stage.name}」阶段。以下哪些是本阶段临床路径规定"
                        f"应当完成的诊疗行为？（多选）"
                    ),
                    "options": action_options,
                    "answer": action_answer,
                }
            )

            if syndrome is not None:
                principles = [
                    t.name for _e, t in kg.neighbours(syndrome.id, {EdgeType.TREATED_BY_PRINCIPLE.value})
                ]
                if principles:
                    pool = [
                        t.name
                        for other in syndromes
                        if other.id != syndrome.id
                        for _e, t in kg.neighbours(other.id, {EdgeType.TREATED_BY_PRINCIPLE.value})
                    ] or [p.name for p in kg.of_type(NodeType.TREATMENT_PRINCIPLE.value)[:60]]
                    options, answer = _options(principles[0], pool, rng)
                    items.append(
                        {
                            **base,
                            "id": f"cp4::{stage.id}",
                            "subtask": "CP4_treatment_principle",
                            "question": f"针对{syndrome.name}，本路径推荐的治法是？",
                            "options": options,
                            "answer": answer,
                        }
                    )

            exit_criteria = [str(c) for c in (stage.get("exit_criteria") or [])]
            successors = [t for _e, t in kg.neighbours(stage.id, {EdgeType.NEXT_STAGE.value})]
            if exit_criteria:
                for met, label in ((True, "exit"), (False, "continue")):
                    findings = (
                        exit_criteria
                        if met
                        else ["症状无明显缓解", "各项指标未见改善", "病情反复"]
                    )
                    gold = ("advance" if successors else "exit") if met else "continue"
                    options = {
                        "A": "继续本阶段治疗（continue）",
                        "B": "进入下一阶段（advance）",
                        "C": "达到出径标准，可以出径（exit）",
                        "D": "证据不足，无法判断（insufficient_evidence）",
                    }
                    answer = {"continue": ["A"], "advance": ["B"], "exit": ["C"]}[gold]
                    items.append(
                        {
                            **base,
                            "id": f"cp6::{stage.id}::{label}",
                            "subtask": "CP6_transition_decision",
                            "question": (
                                f"患者处于「{stage.name}」阶段。本次随访发现："
                                + "；".join(findings)
                                + "。根据临床路径，下一步应当？"
                            ),
                            "options": options,
                            "answer": answer,
                            "followup_findings": findings,
                        }
                    )

    rng.shuffle(items)
    if limit:
        items = items[:limit]
    return items


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/cp/TCM-CP.json")
    parser.add_argument("--kg", default=None)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    kg = load_kg(args.kg)
    items = build(kg, seed=args.seed, limit=args.limit)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    from collections import Counter

    counts = Counter(i["subtask"] for i in items)
    print(f"wrote {len(items)} items to {target}")
    for subtask, n in sorted(counts.items()):
        print(f"  {subtask:28s} {n}")
    print(f"  diseases covered: {len({i['disease'] for i in items})}")
    print(
        "\nNOTE: gold answers are derived from the same knowledge graph the KG "
        "arms may consult. TCM-CP measures whether an agent can execute a "
        "pathway faithfully, not whether the graph improves clinical ability. "
        "Do not pool its contrasts with the SDT or PA effectiveness results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
