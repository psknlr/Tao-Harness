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
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcm_kg import load_kg
from tcm_kg.schema import EdgeType, NodeType
from tcm_kg.store import KGStore

#: Fraction of first-stage CP2 items to keep. First stages are the easiest to
#: identify, and letting them dominate turns stage identification into
#: admission spotting.
FIRST_STAGE_KEEP_RATE = 0.25

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


def _stage_signature(stage: Any, stages: Sequence[Any] = ()) -> frozenset:
    """What a CP2 vignette exposes about a stage, without naming the answer.

    Monitoring items and entry criteria, **plus the treatment already
    completed** when the stage set is supplied. The cumulative prior-action set
    is what the history line states, and it differs by position, so including
    it makes mid-pathway stages identifiable from treatment progress rather
    than from a date.

    Without it only first stages were discriminable -- they alone carry entry
    criteria -- and 87% of CP2 golds were first stages, which made the task
    "spot the admission" rather than "locate the patient in the pathway".
    """
    base = [str(m).strip() for m in (stage.get("monitoring_items") or [])] + [
        str(c).strip() for c in (stage.get("entry_criteria") or [])
    ]
    if stages:
        ordered = sorted(
            stages, key=lambda st: (str(st.get("variant") or ""), st.get("order") or 0)
        )
        index = next((i for i, st in enumerate(ordered) if st.id == stage.id), 0)
        prior = [
            f"done::{a}"
            for earlier in ordered[:index]
            for a in (earlier.get("day_actions") or [])
        ]
        base.extend(dict.fromkeys(prior))
    return frozenset(base)


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
    signature = _stage_signature(stage, stages)
    return [
        other
        for other in stages
        if other.id != stage.id and _stage_signature(other, stages) != signature
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
        if _stage_signature(stage, stages)
        and len(distinguishable_from(stage, stages)) >= min_distractors
    ]


def _history_line(
    stages: Sequence[Any], stage: Any, forbidden: Optional[Set[str]] = None
) -> str:
    """A treatment-history sentence locating the patient in the pathway.

    Without it a vignette describes a patient at no particular time, and every
    stage of the admission fits. It states what has already happened, never
    what is due now -- that is what the item asks for.
    """
    ordered = sorted(stages, key=lambda s: (str(s.get("variant") or ""), s.get("order") or 0))
    index = next((i for i, s in enumerate(ordered) if s.id == stage.id), 0)
    if index == 0:
        # No "今日入院": naming the admission day makes "第1天" readable off the
        # question. State the clinical fact -- no pathway treatment has been
        # given yet -- and let the model infer the stage from it.
        return "本次住院尚未开始路径内的诊疗措施。"
    # Prior treatment, without naming the stage it belonged to: the model has
    # to map treatment progress onto a stage rather than read a label.
    blocked = {b for b in (forbidden or set()) if b}

    done: List[str] = []
    for earlier in ordered[:index]:
        done.extend(_without_leaks([str(a) for a in (earlier.get("day_actions") or [])], blocked))
    unique = list(dict.fromkeys(done))[:5]
    line = "至本次查房，已完成的诊疗包括：" + "、".join(unique) if unique else "已按路径开始治疗"
    return line + "。"


#: Which stage fields each subtask's vignette may expose.
#:
#: A single shared vignette is unsafe: CP5 asks which items the pathway
#: requires monitoring, and the shared vignette printed exactly those items
#: under "本次查房重点观察". Measured on the previous build, **1,210 of 1,210
#: CP5 items (100%)** had their full gold answer present verbatim in their own
#: question, so the task was string matching and no knowledge was needed. Each
#: subtask now gets a vignette that withholds what it asks for.
#: The stage field each subtask asks about -- and therefore must never leak,
#: whether directly or through the treatment-history line.
ANSWER_FIELD: Mapping[str, str] = {
    "CP3": "day_actions",
    "CP5": "monitoring_items",
}

VIGNETTE_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "CP2": ("history", "entry_criteria", "monitoring"),   # asks: which stage
    "CP3": ("history", "monitoring"),                     # asks: which actions
    "CP5": ("history", "progress"),                       # asks: what to monitor
    "CP6": ("history", "monitoring"),                     # asks: transition
}


def _vignette(
    kg: KGStore,
    disease: Any,
    stage: Any,
    syndrome: Optional[Any],
    all_stages: Sequence[Any] = (),
    subtask: str = "CP2",
) -> str:
    """Render a case vignette from stage content, withholding the answers.

    Monitoring items describe what is being watched at this point in the
    admission, which situates the patient in time without naming the stage;
    the stage name, its day actions and its criteria are all withheld because
    they are what the item asks for.
    """
    allowed = VIGNETTE_FIELDS.get(subtask, VIGNETTE_FIELDS["CP2"])
    # Everything this subtask asks for, withheld from every part of the
    # vignette rather than from one field at a time. Patching fields
    # individually kept re-surfacing the same leak somewhere else: pathway
    # text is repetitive, so an item withheld from the history reappears in
    # the monitoring list, and vice versa.
    blocked = {
        str(x).strip()
        for x in (stage.get(ANSWER_FIELD[subtask]) or ())
        if str(x).strip()
    } if subtask in ANSWER_FIELD else set()
    parts = [f"患者因{disease.get('tcm_name') or disease.name}入院，进入该病种中医临床路径。"]
    if stage.get("variant"):
        parts.append(f"路径分支：{stage.get('variant')}。")
    if syndrome is not None:
        parts.append(f"入院辨证为{syndrome.name}。")
        # this disease's own presentation, never the syndrome's global first
        # mention, which for most edges describes a different disease
        presentation = kg.syndrome_presentation(syndrome.id, disease.id)
        if presentation["scope"] == "disease_specific" and presentation["sentence"]:
            parts.append(f"四诊所见：{presentation['sentence'][:160]}")
    if "history" in allowed and all_stages:
        parts.append(_history_line(all_stages, stage, blocked))
    if "progress" in allowed:
        parts.append(_progress_line(all_stages, stage))
    if "entry_criteria" in allowed:
        entry = _without_leaks(
            [str(c) for c in (stage.get("entry_criteria") or [])], blocked
        )[:3]
        if entry:
            parts.append("当前状况符合：" + "；".join(entry) + "。")
    if "monitoring" in allowed:
        monitoring = _without_leaks(
            [str(m) for m in (stage.get("monitoring_items") or [])], blocked
        )[:4]
        if monitoring:
            parts.append("本次查房重点观察：" + "；".join(monitoring) + "。")
    return "".join(parts)


def _without_leaks(values: Sequence[str], blocked: Set[str]) -> List[str]:
    """Drop entries that would reveal a withheld answer.

    Substring in both directions: a gold item hides inside a longer line
    (``中医证候判断`` within ``完成皮损PRSS评分及中医证候判断``) as readily as a
    longer gold item contains a shorter line.
    """
    if not blocked:
        return [v.strip() for v in values if v.strip()]
    return [
        v.strip()
        for v in values
        if v.strip() and not any(b in v or v in b for b in blocked)
    ]


def _progress_line(stages: Sequence[Any], stage: Any) -> str:
    """Clinical progress, with no monitoring content and no stage label.

    Used by CP5, which asks what the pathway requires monitoring: naming any
    monitoring item here would hand over the answer.
    """
    ordered = sorted(stages, key=lambda s: (str(s.get("variant") or ""), s.get("order") or 0))
    index = next((i for i, s in enumerate(ordered) if s.id == stage.id), 0)
    if index == 0:
        return "患者尚未开始本路径的治疗，现按路径规定安排本阶段诊疗。"
    if index >= len(ordered) - 1:
        return "经前期治疗后病情稳定，现进入本路径的收尾阶段。"
    return "经前期治疗后症状有所缓解，病情平稳，现按路径进入下一诊疗节点。"


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

            def _base(subtask: str) -> Dict[str, Any]:
                return {
                    "disease": disease.name,
                    "stage_id": stage.id,
                    "stage_name": stage.name,
                    "variant": stage.get("variant"),
                    "vignette": _vignette(kg, disease, stage, syndrome, stages, subtask),
                    "syndrome": syndrome.name if syndrome else None,
                }

            ordered_stages = sorted(
                stages, key=lambda st: (str(st.get("variant") or ""), st.get("order") or 0)
            )
            is_first = bool(ordered_stages) and ordered_stages[0].id == stage.id

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
                # Cap first-stage items: they are the most easily identified
                # (only first stages carry entry criteria), and in the previous
                # build 87% of CP2 golds were first stages, turning the task
                # into "spot the admission".
                if is_first and rng.random() > FIRST_STAGE_KEEP_RATE:
                    dropped["first_stage_rebalanced"] += 1
                    continue
                options, answer = _options(gold_label, distractors, rng, n=min(6, len(distractors) + 1))
                items.append(
                    {
                        **_base("CP2"),
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
                    **_base("CP3"),
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
                        **_base("CP5"),
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
                items.extend(_transition_items(_base("CP6"), stage, exit_criteria, successors))
            else:
                dropped["stage_without_exit_criteria"] += 1

        # ---- CP1: pathway eligibility ----
        items.extend(
            _eligibility_items(kg, disease, stages, by_disease, rng, base_syndrome=syndrome)
        )

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
    for clause in exclusion:
        fact = exclusion_to_patient_fact(clause)
        if not fact:
            continue
        out.append(
            {
                "disease": disease.name,
                "stage_id": first.id,
                "stage_name": first.name,
                "syndrome": base_syndrome.name if base_syndrome is not None else None,
                "vignette": (
                    f"患者拟以{disease.get('tcm_name') or disease.name}收入院。"
                    f"入院评估：" + "；".join(inclusion[:2]) + "。"
                    f"既往及现症：该患者合并{fact}。"
                ),
                "id": f"cp1::{disease.id}::excluded",
                "subtask": "CP1_pathway_eligibility",
                "question": f"该患者是否符合「{disease.name}」中医临床路径的入径标准？",
                "options": dict(options),
                "answer": ["B"],
                "excluded_because": fact,
            }
        )
        break
    return out


#: Clause markers that make an entry criterion an *exclusion*.
_EXCLUSION_WORDS = ("不进入", "不宜", "除外", "禁忌", "不能进入", "排除")

#: Policy language to strip when turning a rule into a patient fact.
_POLICY_PHRASES = (
    "不进入本路径", "不能进入本路径", "不进入该路径", "不进入临床路径",
    "者不进入", "不进入", "不宜进入", "不宜", "予以除外", "除外",
    "应排除", "排除", "禁忌进入", "属禁忌",
)
_LIST_SPLIT_RE = re.compile(r"[、，,；;]")


def exclusion_to_patient_fact(criterion: str) -> Optional[str]:
    """Turn an exclusion rule into a statement about this patient.

    An item that quotes the rule -- "合并严重心功能不全者不进入本路径" -- is
    answerable by spotting the words 不进入 and needs no pathway knowledge. The
    item has to state a *fact* ("患者既往有严重心功能不全") and let the model
    decide whether it violates the entry criteria.

    Returns ``None`` when the clause cannot be reduced to a clean fact, in
    which case no negative item is generated: a malformed vignette is worse
    than a missing one.
    """
    text = str(criterion or "").strip()
    if not text:
        return None
    for phrase in _POLICY_PHRASES:
        text = text.replace(phrase, "")
    text = text.strip(" 。.，,、；;：:的")
    if not text or len(text) < 3:
        return None
    # keep the first listed condition; a five-item list reads as a rule again
    parts = [p.strip() for p in _LIST_SPLIT_RE.split(text) if len(p.strip()) >= 3]
    condition = parts[0] if parts else text
    condition = condition.strip(" 者等的")
    if len(condition) < 3 or any(w in condition for w in ("本路径", "临床路径")):
        return None
    return condition


def _treatment_vignette(kg: KGStore, disease: Any, syndrome: Any) -> str:
    presentation = kg.syndrome_presentation(syndrome.id, disease.id)
    text = f"患者因{disease.get('tcm_name') or disease.name}入院，辨证为{syndrome.name}。"
    if presentation["scope"] == "disease_specific" and presentation["sentence"]:
        text += f"四诊所见：{presentation['sentence'][:160]}"
    return text


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
        "vignette": _treatment_vignette(kg, disease, syndrome),
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
