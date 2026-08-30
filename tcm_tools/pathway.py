"""Clinical-pathway tools.

The knowledge graph carries a substantial pathway layer -- 1,440
``PathwayStage`` nodes joined by 1,118 ``NEXT_STAGE`` edges, with entry and
exit criteria, monitoring items, outcome indicators, per-day actions and
nursing items -- and until now the agent could not reach most of it.
``retrieve_clinical_context`` returned a stage's criteria but not its
``day_actions`` (present on 89% of stages) or ``nursing_items`` (64%), which
are the part of a pathway that actually says what to *do*, and ``NEXT_STAGE``
was never traversed at all, so stages were static context rather than a state
machine.

These three tools close that gap:

``retrieve_pathway_stage``      the full content of one stage
``evaluate_pathway_transition`` deterministic criteria evaluation + traversal
``retrieve_treatment_plan``     syndrome -> principle -> formula/patent/external

They live in the pathway domain only. SDT cannot reach them, which keeps its
treatment isolation intact.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tcm_kg.normalize import char_ngrams, normalize_text
from tcm_kg.schema import Domain, EdgeType, NodeType

from ._common import documents_block, edge_evidence, node_brief, resolve_entity
from .base import REGISTRY, Coverage, ToolContext, ToolPhase, ToolResult, ToolSpec

_PATHWAY_DOMAINS = (Domain.PATHWAY, Domain.FULL)

#: Every stage field worth returning, with the share of stages that carry it.
STAGE_FIELDS: Tuple[str, ...] = (
    "order",
    "variant",
    "disease",
    "entry_criteria",
    "exit_criteria",
    "monitoring_items",
    "outcome_indicators",
    "day_actions",
    "nursing_items",
)


# --------------------------------------------------------------------------- #
# retrieve_pathway_stage
# --------------------------------------------------------------------------- #

_STAGE_SPEC = ToolSpec(
    name="retrieve_pathway_stage",
    description=(
        "查询临床路径阶段的完整内容：阶段序号与变异分支、入径标准、出径标准、"
        "本阶段应完成的诊疗行为（day_actions）、护理项目（nursing_items）、"
        "监测项目（monitoring_items）、疗效评价指标（outcome_indicators），"
        "以及该阶段的前后阶段。"
        "用于判断患者当前处于哪个阶段、本阶段应当做什么。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "disease": {"type": "string", "description": "疾病名，用于定位该疾病的路径。"},
            "stage": {
                "type": "string",
                "description": "可选，阶段名（如“第1天”“入院第2～3天”）或阶段 id。留空返回该疾病的全部阶段概览。",
            },
            "variant": {"type": "string", "description": "可选，路径分支（如“急性期”）。"},
        },
        "required": ["disease"],
    },
    domains=_PATHWAY_DOMAINS,
)


@REGISTRY.register(_STAGE_SPEC)
def retrieve_pathway_stage(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    disease_name = str(args.get("disease", "")).strip()
    stage_name = str(args.get("stage") or "").strip()
    variant = str(args.get("variant") or "").strip()

    node = kg.node(disease_name)
    diseases = [node] if node and node.type == NodeType.DISEASE.value else resolve_entity(
        kg, disease_name, [NodeType.DISEASE.value], retriever=ctx.retriever, domain=ctx.domain
    )
    if not diseases:
        return ToolResult(
            tool=_STAGE_SPEC.name,
            ok=True,
            coverage=Coverage.EMPTY,
            data={"disease": disease_name, "stages": []},
            caveats=[f"图谱中没有名为 {disease_name!r} 的疾病，因而没有对应的临床路径。"],
        )

    disease = diseases[0]
    stages = [
        target
        for _e, target in kg.neighbours(disease.id, {EdgeType.HAS_PATHWAY_STAGE.value})
    ]
    if variant:
        stages = [s for s in stages if variant in str(s.get("variant") or "")]
    if not stages:
        return ToolResult(
            tool=_STAGE_SPEC.name,
            ok=True,
            coverage=Coverage.EMPTY,
            data={"disease": disease.name, "stages": []},
            caveats=[
                f"{disease.name} 在图谱中没有临床路径阶段"
                + (f"（分支过滤：{variant}）。" if variant else "。")
            ],
        )

    if stage_name:
        matched = [s for s in stages if s.id == stage_name or stage_name in s.name]
        if not matched:
            return ToolResult(
                tool=_STAGE_SPEC.name,
                ok=True,
                coverage=Coverage.PARTIAL,
                data={
                    "disease": disease.name,
                    "requested_stage": stage_name,
                    "available_stages": [s.name for s in _ordered(stages)],
                },
                caveats=[f"未找到阶段 {stage_name!r}；可用阶段见 available_stages。"],
            )
        stages = matched

    payload = {
        "disease": node_brief(kg, disease, with_sentence=False),
        "n_stages_total": len(
            kg.out_edges(disease.id, {EdgeType.HAS_PATHWAY_STAGE.value})
        ),
        "stages": [_stage_detail(ctx, s) for s in _ordered(stages)[:8]],
    }
    payload["documents"] = documents_block(kg, [disease.id])
    return ToolResult(
        tool=_STAGE_SPEC.name, ok=True, coverage=Coverage.SUPPORTED, data=payload
    )


def _ordered(stages: Sequence[Any]) -> List[Any]:
    return sorted(stages, key=lambda s: (str(s.get("variant") or ""), s.get("order") or 0))


def _stage_detail(ctx: ToolContext, stage: Any) -> Dict[str, Any]:
    kg = ctx.kg
    detail = node_brief(kg, stage, with_attrs=STAGE_FIELDS)
    detail["next_stages"] = [
        {"id": target.id, "name": target.name, "order": target.get("order")}
        for _e, target in kg.neighbours(stage.id, {EdgeType.NEXT_STAGE.value})
    ]
    detail["previous_stages"] = [
        {"id": kg.nodes[edge.source].id, "name": kg.nodes[edge.source].name}
        for edge in kg.in_edges(stage.id, {EdgeType.NEXT_STAGE.value})
    ]
    missing = [f for f in ("entry_criteria", "exit_criteria") if not stage.get(f)]
    if missing:
        detail["absent_fields"] = missing
    return detail


# --------------------------------------------------------------------------- #
# evaluate_pathway_transition
# --------------------------------------------------------------------------- #

_TRANSITION_SPEC = ToolSpec(
    name="evaluate_pathway_transition",
    description=(
        "确定性路径状态判定（不调用大模型）。给定患者当前所处阶段与本次随访观察到的"
        "临床发现，逐条比对该阶段的出径标准与下一阶段的入径标准，"
        "并沿 NEXT_STAGE 给出可达的下一阶段。"
        "返回逐条标准的匹配情况与建议动作：continue（继续本阶段）、"
        "advance（进入下一阶段）、exit（出径）、insufficient_evidence（证据不足）。"
        "注意：判定只依据图谱记载的标准文本与你提供的发现之间的词面匹配，"
        "不构成临床决策，最终判断仍需你结合病情作出。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "stage_id": {"type": "string", "description": "当前阶段的实体 id 或阶段名。"},
            "disease": {"type": "string", "description": "疾病名（当 stage_id 为阶段名时用于定位）。"},
            "findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "本次观察到的临床发现列表，如 [\"体温正常\", \"疼痛明显减轻\"]。",
            },
        },
        "required": ["stage_id"],
    },
    domains=_PATHWAY_DOMAINS,
    deterministic=True,
    phase=ToolPhase.BOTH,
)


@REGISTRY.register(_TRANSITION_SPEC)
def evaluate_pathway_transition(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    stage_ref = str(args.get("stage_id", "")).strip()
    disease_name = str(args.get("disease") or "").strip()
    findings = [str(f) for f in (args.get("findings") or []) if str(f).strip()]

    stage = kg.node(stage_ref)
    if stage is None or stage.type != NodeType.PATHWAY_STAGE.value:
        stage = _find_stage(ctx, disease_name, stage_ref)
    if stage is None:
        return ToolResult(
            tool=_TRANSITION_SPEC.name,
            ok=True,
            coverage=Coverage.EMPTY,
            data={"stage": stage_ref, "recommendation": "insufficient_evidence"},
            caveats=[f"未能在图谱中定位阶段 {stage_ref!r}。"],
        )

    exit_criteria = [str(c) for c in (stage.get("exit_criteria") or [])]
    outcome_indicators = [str(c) for c in (stage.get("outcome_indicators") or [])]
    exit_eval = [_match_criterion(c, findings) for c in exit_criteria]

    successors = [t for _e, t in kg.neighbours(stage.id, {EdgeType.NEXT_STAGE.value})]
    successor_eval: List[Dict[str, Any]] = []
    for successor in successors:
        entry = [str(c) for c in (successor.get("entry_criteria") or [])]
        successor_eval.append(
            {
                "stage": {"id": successor.id, "name": successor.name, "order": successor.get("order")},
                "entry_criteria": [_match_criterion(c, findings) for c in entry],
                "entry_criteria_recorded": bool(entry),
            }
        )

    recommendation, rationale = _recommend(
        findings, exit_criteria, exit_eval, successors, successor_eval
    )

    payload = {
        "stage": {
            "id": stage.id,
            "name": stage.name,
            "order": stage.get("order"),
            "variant": stage.get("variant"),
            "disease": stage.get("disease"),
        },
        "findings": findings,
        "exit_criteria_recorded": bool(exit_criteria),
        "exit_criteria_evaluation": exit_eval,
        "outcome_indicators": outcome_indicators,
        "successors": successor_eval,
        "is_terminal": not successors,
        "recommendation": recommendation,
        "rationale": rationale,
    }
    coverage = (
        Coverage.SUPPORTED
        if exit_criteria or successors
        else Coverage.NOT_COVERED
    )
    caveats = [
        "标准匹配为词面比对，不是临床判断；标准未记载时不能推断为“已达标”。"
    ]
    if not exit_criteria:
        caveats.append(
            "该阶段在图谱中没有记载出径标准（图谱中仅 12% 的阶段有 exit_criteria），"
            "因此无法判定是否可以出径。"
        )
    return ToolResult(
        tool=_TRANSITION_SPEC.name,
        ok=True,
        coverage=coverage,
        data=payload,
        caveats=caveats,
    )


def _find_stage(ctx: ToolContext, disease_name: str, stage_ref: str) -> Optional[Any]:
    kg = ctx.kg
    if not disease_name:
        matches = kg.find_by_name(stage_ref, [NodeType.PATHWAY_STAGE.value])
        return matches[0] if matches else None
    diseases = resolve_entity(
        kg, disease_name, [NodeType.DISEASE.value], retriever=ctx.retriever, domain=ctx.domain
    )
    if not diseases:
        return None
    for _e, stage in kg.neighbours(diseases[0].id, {EdgeType.HAS_PATHWAY_STAGE.value}):
        if stage_ref and (stage_ref in stage.name or stage.id == stage_ref):
            return stage
    return None


#: A criterion counts as met when the findings cover enough of its content
#: words. Deliberately conservative: partial overlap reports `partial`, never
#: `met`, because "probably discharged" is not a discharge decision.
_MET_THRESHOLD = 0.6
_PARTIAL_THRESHOLD = 0.3
_NEGATION = ("未", "无", "不", "非", "尚未", "没有")


def _match_criterion(criterion: str, findings: Sequence[str]) -> Dict[str, Any]:
    """Word-level match of one criterion against the observed findings."""
    if not findings:
        return {"criterion": criterion, "status": "no_findings_supplied", "score": 0.0}
    criterion_tokens = set(char_ngrams(criterion, (2,)))
    if not criterion_tokens:
        return {"criterion": criterion, "status": "unparseable", "score": 0.0}

    best_score, best_finding = 0.0, ""
    for finding in findings:
        tokens = set(char_ngrams(finding, (2,)))
        if not tokens:
            continue
        score = len(criterion_tokens & tokens) / len(criterion_tokens)
        if score > best_score:
            best_score, best_finding = score, finding

    if best_score >= _MET_THRESHOLD:
        status = "met"
    elif best_score >= _PARTIAL_THRESHOLD:
        status = "partial"
    else:
        status = "not_evidenced"
    # a finding that negates the criterion is not evidence for it
    if status != "not_evidenced" and any(
        marker in best_finding for marker in _NEGATION
    ) and not any(marker in criterion for marker in _NEGATION):
        status = "contradicted"
    return {
        "criterion": criterion,
        "status": status,
        "score": round(best_score, 3),
        "matched_finding": best_finding or None,
    }


def _recommend(
    findings: Sequence[str],
    exit_criteria: Sequence[str],
    exit_eval: Sequence[Mapping[str, Any]],
    successors: Sequence[Any],
    successor_eval: Sequence[Mapping[str, Any]],
) -> Tuple[str, str]:
    if not findings:
        return "insufficient_evidence", "未提供随访发现，无法评估任何标准。"
    if exit_criteria:
        met = sum(1 for e in exit_eval if e["status"] == "met")
        contradicted = sum(1 for e in exit_eval if e["status"] == "contradicted")
        if contradicted:
            return "continue", f"{contradicted} 条出径标准被观察结果否定。"
        if met == len(exit_criteria):
            return ("exit" if not successors else "advance"), "全部出径标准均有对应发现支持。"
        if met:
            return "continue", f"仅 {met}/{len(exit_criteria)} 条出径标准得到支持。"
    for candidate in successor_eval:
        criteria = candidate["entry_criteria"]
        if criteria and all(c["status"] == "met" for c in criteria):
            return "advance", f"下一阶段 {candidate['stage']['name']} 的入径标准全部满足。"
    if not exit_criteria and not any(c["entry_criteria"] for c in successor_eval):
        return (
            "insufficient_evidence",
            "图谱未记载本阶段出径标准与后继阶段入径标准，无法作出转阶段判定。",
        )
    return "continue", "尚无足够证据支持转阶段或出径。"


# --------------------------------------------------------------------------- #
# retrieve_treatment_plan
# --------------------------------------------------------------------------- #

_PLAN_SPEC = ToolSpec(
    name="retrieve_treatment_plan",
    description=(
        "查询证候对应的完整治疗方案：治法（TreatmentPrinciple）、"
        "推荐方剂及其组成、中成药、非药物外治疗法，以及各自的原文依据。"
        "用于临床路径中“本阶段应给予什么治疗”的决策。"
        "仅在临床路径任务中可用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "syndrome": {"type": "string", "description": "证候名或实体 id。"},
            "include_composition": {
                "type": "boolean",
                "description": "是否返回方剂的饮片组成，默认 true。",
            },
        },
        "required": ["syndrome"],
    },
    domains=_PATHWAY_DOMAINS,
)


@REGISTRY.register(_PLAN_SPEC)
def retrieve_treatment_plan(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    name = str(args.get("syndrome", "")).strip()
    include_composition = bool(args.get("include_composition", True))

    node = kg.node(name)
    matches = [node] if node and node.type == NodeType.SYNDROME.value else resolve_entity(
        kg, name, [NodeType.SYNDROME.value], retriever=ctx.retriever, domain=ctx.domain
    )
    if not matches:
        return ToolResult(
            tool=_PLAN_SPEC.name,
            ok=True,
            coverage=Coverage.EMPTY,
            data={"syndrome": name},
            caveats=[f"图谱未收录证候 {name!r}，无法给出其治疗方案。"],
        )

    syndrome = matches[0]
    principles = [
        node_brief(kg, t, with_sentence=False)
        for _e, t in kg.neighbours(syndrome.id, {EdgeType.TREATED_BY_PRINCIPLE.value})
    ]
    formulas: List[Dict[str, Any]] = []
    for edge, formula in kg.neighbours(syndrome.id, {EdgeType.USES_FORMULA.value}):
        entry = node_brief(kg, formula, with_sentence=False)
        if include_composition:
            entry["composition"] = [
                herb.name
                for _e, herb in kg.neighbours(formula.id, {EdgeType.CONTAINS_HERB.value})
            ][:40]
        entry["provenance"] = edge_evidence(edge, max_items=1)
        formulas.append(entry)

    payload = {
        "syndrome": node_brief(kg, syndrome),
        "treatment_principles": principles,
        "formulas": formulas[:6],
        "patent_medicines": [
            node_brief(kg, t, with_sentence=False)
            for _e, t in kg.neighbours(syndrome.id, {EdgeType.USES_PATENT_MEDICINE.value})
        ][:10],
        "external_therapies": [
            node_brief(kg, t, with_sentence=False)
            for _e, t in kg.neighbours(syndrome.id, {EdgeType.USES_EXTERNAL_THERAPY.value})
        ][:10],
        "direct_herbs": [
            node_brief(kg, t, with_sentence=False)
            for _e, t in kg.neighbours(syndrome.id, {EdgeType.USES_HERB_DIRECT.value})
        ][:10],
    }
    payload["documents"] = documents_block(kg, [syndrome.id])
    covered = any(
        payload[key]
        for key in ("treatment_principles", "formulas", "patent_medicines", "external_therapies")
    )
    return ToolResult(
        tool=_PLAN_SPEC.name,
        ok=True,
        coverage=Coverage.SUPPORTED if covered else Coverage.EMPTY,
        data=payload,
        caveats=[] if covered else ["该证候在图谱中没有关联任何治疗方案。"],
    )
