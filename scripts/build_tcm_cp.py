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

Discriminability
----------------

The first build of CP2 was **underdetermined**: measured over all 1,210 items,
*zero* could be uniquely resolved from what the vignette exposed, and the
median item was compatible with five stages of its own pathway. Clinical
pathway stages within one disease share their monitoring items almost
verbatim, so "which stage is this?" had no answer derivable from the question.
Scoring a model on such an item measures nothing.

Items are now emitted only when the vignette **discriminates**: the gold
stage's exposed signature must differ from every distractor's, and where the
graph records stage-specific findings the vignette carries them plus a
treatment-history line locating the patient in time. Everything else is
dropped, and the build reports how many were dropped and why -- a smaller
benchmark that measures something beats a large one that does not.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
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


def _stage_signature(stage: Any) -> frozenset:
    """What a vignette can expose about a stage without naming its answer.

    Monitoring items plus entry criteria: observable, and not the day actions
    or the stage name that CP2 and CP3 ask for.
    """
    return frozenset(
        [str(m).strip() for m in (stage.get("monitoring_items") or [])]
        + [str(c).strip() for c in (stage.get("entry_criteria") or [])]
    )


def stage_label(stage: Any, stages: Sequence[Any]) -> str:
    """A display label that names exactly one stage of this pathway.

    Several diseases run parallel pathway variants whose stages share names
    (``第4～7天`` exists once per variant). An option list built from bare
    names would then offer the same label twice, or offer the gold label as a
    distractor -- the item would be unanswerable for a reason that has nothing
    to do with clinical reasoning. Prefixing the variant makes each label
    denote one stage.
    """
    duplicated = sum(1 for other in stages if other.name == stage.name) > 1
    variant = str(stage.get("variant") or "").strip()
    return f"{variant}·{stage.name}" if duplicated and variant else stage.name


def distinguishable_from(stage: Any, stages: Sequence[Any]) -> List[Any]:
    """Siblings whose observable signature differs from ``stage``'s.

    These are the only admissible distractors: an option sharing the gold
    stage's entire signature is not a wrong answer the vignette rules out, it
    is a second correct answer the item pretends does not exist.
    """
    signature = _stage_signature(stage)
    return [
        other
        for other in stages
        if other.id != stage.id and _stage_signature(other) != signature
    ]


def discriminable_stages(stages: Sequence[Any], *, min_distractors: int = 2) -> List[Any]:
    """Stages a vignette can actually pin down.

    A stage qualifies when it has an observable signature of its own and at
    least ``min_distractors`` siblings that differ from it -- enough to build
    an item whose wrong options really are wrong. Requiring instead that a
    stage be *globally unique* within its pathway was too strict: it discarded
    stages that were perfectly distinguishable from the particular siblings an
    item would show.
    """
    return [
        stage
        for stage in stages
        if _stage_signature(stage)
        and len(distinguishable_from(stage, stages)) >= min_distractors
    ]


def _history_line(stages: Sequence[Any], stage: Any) -> str:
    """A treatment-history sentence locating the patient in the pathway.

    Without it a vignette describes a patient at no particular time, and every
    stage of the admission fits. It states what has already happened, never
    what is due now -- that is what the item asks for.
    """
    ordered = sorted(stages, key=lambda s: (str(s.get("variant") or ""), s.get("order") or 0))
    index = next((i for i, s in enumerate(ordered) if s.id == stage.id), 0)
    if index == 0:
        return "患者今日入院，尚未开始本路径的治疗。"
    previous = ordered[index - 1]
    done = [str(a) for a in (previous.get("day_actions") or [])][:3]
    line = f"此前已完成「{previous.name}」阶段的诊疗，"
    if done:
        line += "包括" + "、".join(done) + "，"
    return line + "现进入下一次查房。"


def _vignette(
    kg: KGStore,
    disease: Any,
    stage: Any,
    syndrome: Optional[Any],
    all_stages: Sequence[Any] = (),
) -> str:
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
    if all_stages:
        parts.append(_history_line(all_stages, stage))
    entry = [str(c) for c in (stage.get("entry_criteria") or [])][:3]
    if entry:
        parts.append("当前状况符合：" + "；".join(entry) + "。")
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


def build(
    kg: KGStore, *, seed: int = 20260829, limit: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate the benchmark and a report of what was emitted and dropped."""
    rng = random.Random(seed)
    by_disease = _stages_by_disease(kg)
    items: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = defaultdict(int)
    #: every monitoring item in the graph, for cross-disease CP5 distractors
    all_monitoring = sorted(
        {
            str(m)
            for stage in kg.of_type(NodeType.PATHWAY_STAGE.value)
            for m in (stage.get("monitoring_items") or [])
            if str(m).strip()
        }
    )

    for disease_id, stages in sorted(by_disease.items()):
        disease = kg.node(disease_id)
        syndromes = [t for _e, t in kg.neighbours(disease_id, {EdgeType.HAS_SYNDROME.value})]
        syndrome = syndromes[0] if syndromes else None
        usable = discriminable_stages(stages)
        dropped["stage_not_discriminable"] += len(stages) - len(usable)
        disease_monitoring = {
            str(m) for st in stages for m in (st.get("monitoring_items") or [])
        }
        monitoring_pool = [m for m in all_monitoring if m not in disease_monitoring]

        for stage in stages:
            actions = [str(a) for a in (stage.get("day_actions") or [])]
            other_actions = [
                str(a)
                for other in stages
                if other.id != stage.id
                for a in (other.get("day_actions") or [])
                if str(a) not in actions
            ]
            if len(other_actions) < 3:
                dropped["too_few_distractor_actions"] += 1
                continue

            base = {
                "disease": disease.name,
                "stage_id": stage.id,
                "stage_name": stage.name,
                "variant": stage.get("variant"),
                "vignette": _vignette(kg, disease, stage, syndrome, stages),
                "syndrome": syndrome.name if syndrome else None,
            }

            # ---- CP2: stage identification, distractors that really differ ----
            if stage in usable:
                gold_label = stage_label(stage, stages)
                distractors = [
                    stage_label(s, stages)
                    for s in distinguishable_from(stage, stages)
                    if stage_label(s, stages) != gold_label
                ]
                if len(distractors) < 2:
                    dropped["ambiguous_stage_labels"] += 1
                    continue
                options, answer = _options(gold_label, distractors, rng, n=min(6, len(distractors) + 1))
                items.append(
                    {
                        **base,
                        "id": f"cp2::{stage.id}",
                        "subtask": "CP2_stage_identification",
                        "question": "根据上述病情描述与治疗经过，患者当前处于该临床路径的哪个阶段？",
                        "options": options,
                        "answer": answer,
                        "n_distractors": len(distractors),
                    }
                )

            # ---- CP3: actions due at this stage ----
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

            # ---- CP5: monitoring and safety ----
            monitoring = [str(m) for m in (stage.get("monitoring_items") or [])]
            # Distractors come from *other diseases*. Within one pathway the
            # monitoring lists repeat almost verbatim between stages, so
            # same-disease distractors are mostly also-correct answers; the
            # discriminating question for CP5 is which items this disease's
            # pathway requires at all.
            other_monitoring = [m for m in monitoring_pool if m not in monitoring]
            if monitoring and len(other_monitoring) >= 3:
                options, answer = _multi_options(monitoring, other_monitoring, rng)
                items.append(
                    {
                        **base,
                        "id": f"cp5m::{stage.id}",
                        "subtask": "CP5_monitoring",
                        "question": (
                            f"患者处于「{stage.name}」阶段。本阶段临床路径要求监测哪些内容？（多选）"
                        ),
                        "options": options,
                        "answer": answer,
                    }
                )

            # ---- CP6: transition decisions ----
            exit_criteria = [str(c) for c in (stage.get("exit_criteria") or [])]
            successors = [t for _e, t in kg.neighbours(stage.id, {EdgeType.NEXT_STAGE.value})]
            if exit_criteria:
                items.extend(_transition_items(base, stage, exit_criteria, successors))
            else:
                dropped["stage_without_exit_criteria"] += 1

        # ---- CP1: pathway eligibility ----
        items.extend(_eligibility_items(kg, disease, stages, by_disease, rng, base_syndrome=syndrome))

        # ---- CP4: treatment planning, beyond the principle alone ----
        for syn in syndromes[:2]:
            items.extend(_treatment_items(kg, disease, syn, syndromes, rng))

    rng.shuffle(items)
    if limit:
        items = items[:limit]
    report = {
        "n_items": len(items),
        "by_subtask": dict(sorted(Counter(i["subtask"] for i in items).items())),
        "n_diseases": len({i["disease"] for i in items}),
        "dropped": dict(sorted(dropped.items())),
    }
    return items, report


def _transition_items(
    base: Mapping[str, Any],
    stage: Any,
    exit_criteria: Sequence[str],
    successors: Sequence[Any],
) -> List[Dict[str, Any]]:
    """continue / advance / exit / insufficient_evidence items.

    The fourth state is the one that matters clinically and the first build
    never produced. A pathway agent that must always choose an action will
    choose one when the record does not support any, which in a discharge
    decision is the dangerous direction. Generating cases whose correct answer
    is "the evidence does not settle this" is what lets the benchmark reward
    saying so.
    """
    options = {
        "A": "继续本阶段治疗（continue）",
        "B": "进入下一阶段（advance）",
        "C": "达到出径标准，可以出径（exit）",
        "D": "证据不足，无法判断（insufficient_evidence）",
    }
    advance_or_exit = "B" if successors else "C"
    out: List[Dict[str, Any]] = []

    def _item(suffix: str, findings: Sequence[str], answer: str, note: str = "") -> Dict[str, Any]:
        return {
            **base,
            "id": f"cp6::{stage.id}::{suffix}",
            "subtask": "CP6_transition_decision",
            "question": (
                f"患者处于「{stage.name}」阶段。本次随访发现："
                + "；".join(findings)
                + "。根据临床路径，下一步应当？"
            ),
            "options": dict(options),
            "answer": [answer],
            "followup_findings": list(findings),
            **({"rationale": note} if note else {}),
        }

    out.append(_item("met", list(exit_criteria), advance_or_exit, "全部出径标准均有对应发现"))
    out.append(
        _item(
            "unmet",
            ["症状无明显缓解", "各项指标未见改善", "病情反复"],
            "A",
            "出径标准未满足",
        )
    )
    # partial evidence: some criteria met, the rest simply not assessed
    if len(exit_criteria) >= 2:
        half = list(exit_criteria[: max(1, len(exit_criteria) // 2)])
        unchecked = list(exit_criteria[max(1, len(exit_criteria) // 2) :])
        out.append(
            _item(
                "partial",
                half + [f"{c}尚未复查" for c in unchecked[:2]],
                "D",
                "部分出径标准未评估，证据不足以支持转阶段",
            )
        )
    return out


def _eligibility_items(
    kg: KGStore,
    disease: Any,
    stages: Sequence[Any],
    by_disease: Mapping[str, Sequence[Any]],
    rng: random.Random,
    *,
    base_syndrome: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """CP1: does this patient belong in this pathway at all?

    Entry criteria for the first stage carry both inclusion and exclusion
    clauses; the negative case is generated from an exclusion clause, so the
    correct answer is "not eligible" for a reason the pathway itself states.
    """
    ordered = sorted(stages, key=lambda s: (str(s.get("variant") or ""), s.get("order") or 0))
    first = next((s for s in ordered if s.get("entry_criteria")), None)
    if first is None:
        return []
    criteria = [str(c) for c in (first.get("entry_criteria") or [])]
    inclusion = [c for c in criteria if not any(w in c for w in _EXCLUSION_WORDS)]
    exclusion = [c for c in criteria if any(w in c for w in _EXCLUSION_WORDS)]
    if not inclusion:
        return []

    options = {
        "A": "符合入径标准，可以进入该临床路径",
        "B": "不符合入径标准，不应进入该临床路径",
        "C": "信息不足，无法判断是否入径",
    }
    out = [
        {
            "disease": disease.name,
            "stage_id": first.id,
            "stage_name": first.name,
            "syndrome": base_syndrome.name if base_syndrome is not None else None,
            "vignette": (
                f"患者拟以{disease.get('tcm_name') or disease.name}收入院。"
                f"入院评估：" + "；".join(inclusion[:3]) + "。"
            ),
            "id": f"cp1::{disease.id}::eligible",
            "subtask": "CP1_pathway_eligibility",
            "question": f"该患者是否符合「{disease.name}」中医临床路径的入径标准？",
            "options": dict(options),
            "answer": ["A"],
        }
    ]
    if exclusion:
        out.append(
            {
                "disease": disease.name,
                "stage_id": first.id,
                "stage_name": first.name,
                "syndrome": base_syndrome.name if base_syndrome is not None else None,
                "vignette": (
                    f"患者拟以{disease.get('tcm_name') or disease.name}收入院。"
                    f"入院评估：" + "；".join(inclusion[:2]) + "。"
                    f"另注意：该患者{exclusion[0][:60]}。"
                ),
                "id": f"cp1::{disease.id}::excluded",
                "subtask": "CP1_pathway_eligibility",
                "question": f"该患者是否符合「{disease.name}」中医临床路径的入径标准？",
                "options": dict(options),
                "answer": ["B"],
            }
        )
    return out


#: Clause markers that make an entry criterion an *exclusion*.
_EXCLUSION_WORDS = ("不进入", "不宜", "除外", "禁忌", "不能进入", "排除")


def _treatment_items(
    kg: KGStore,
    disease: Any,
    syndrome: Any,
    siblings: Sequence[Any],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """CP4: treatment planning across principle, formula, patent and external.

    The first build asked only for the treatment principle, which left the
    formula, patent-medicine and external-therapy edges -- the bulk of the
    treatment sub-graph -- untested.
    """
    base = {
        "disease": disease.name,
        "stage_id": None,
        "stage_name": None,
        "syndrome": syndrome.name,
        "vignette": (
            f"患者因{disease.get('tcm_name') or disease.name}入院，辨证为{syndrome.name}。"
            + (f"四诊所见：{syndrome.sentence()[:160]}" if syndrome.sentence() else "")
        ),
    }
    specs = [
        ("principle", EdgeType.TREATED_BY_PRINCIPLE.value, "本证的治法是？", "CP4_treatment_principle"),
        ("formula", EdgeType.USES_FORMULA.value, "本证推荐使用的方剂是？", "CP4_formula"),
        ("patent", EdgeType.USES_PATENT_MEDICINE.value, "本证可选用的中成药是？", "CP4_patent_medicine"),
        ("external", EdgeType.USES_EXTERNAL_THERAPY.value, "本证可采用的非药物外治疗法是？", "CP4_external_therapy"),
    ]
    out: List[Dict[str, Any]] = []
    for suffix, edge_type, question, subtask in specs:
        correct = [t.name for _e, t in kg.neighbours(syndrome.id, {edge_type})]
        if not correct:
            continue
        pool = [
            t.name
            for other in siblings
            if other.id != syndrome.id
            for _e, t in kg.neighbours(other.id, {edge_type})
            if t.name not in correct
        ]
        if len(pool) < 3:
            continue
        options, answer = _options(correct[0], pool, rng)
        out.append(
            {
                **base,
                "id": f"cp4{suffix}::{syndrome.id}",
                "subtask": subtask,
                "question": question,
                "options": options,
                "answer": answer,
            }
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/cp/TCM-CP.json")
    parser.add_argument("--kg", default=None)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    kg = load_kg(args.kg)
    items, report = build(kg, seed=args.seed, limit=args.limit)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(str(target) + ".build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"wrote {report['n_items']} items to {target}")
    for subtask, n in report["by_subtask"].items():
        print(f"  {subtask:30s} {n}")
    print(f"  diseases covered: {report['n_diseases']}")
    print("  dropped as unusable:")
    for reason, n in report["dropped"].items():
        print(f"    {reason:34s} {n}")
    print(
        "\nNOTE: gold answers are derived from the same knowledge graph the KG "
        "arms may consult. TCM-CP measures whether an agent can execute a "
        "pathway faithfully, not whether the graph improves clinical ability. "
        "Do not pool its contrasts with the SDT or PA effectiveness results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
