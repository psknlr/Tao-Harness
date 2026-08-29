"""Safety-constraint tool (tool 6).

The safety sub-graph in this KG is narrower than the name suggests, and the
tool is written to say so.  ``SafetyContext`` nodes are disease- and
syndrome-scoped patient cohorts (``阳气亏虚血瘀证心衰患者``,
``孕妇及哺乳期妇女``) drawn from the nursing and dietary sections of clinical
pathways, and ``RestrictedItem`` nodes are predominantly dietary restrictions
(``生冷厚腻之品``) plus a handful of procedure restrictions.  There is no
herb-herb incompatibility table.  Every result states which of those it is, so
the model can tell a real dietary contraindication from a missing drug-drug
rule instead of treating an empty result as a clean bill of health.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from tcm_kg.schema import Domain, EdgeType, NodeType

from ._common import documents_block, edge_evidence, node_brief, resolve_entity
from .base import REGISTRY, Coverage, ToolContext, ToolResult, ToolSpec

_SAFETY_DOMAINS = (Domain.SAFETY, Domain.PATHWAY, Domain.FULL)

SAFETY_SCOPE_NOTE = (
    "图谱的安全知识来源于诊疗方案与临床路径中的调护、饮食与操作禁忌段落，"
    "主要覆盖：饮食禁忌、操作/疗法禁忌、特定证候或特定人群的注意事项。"
    "图谱【不包含】中药十八反十九畏配伍禁忌表、妊娠禁忌药分级表、"
    "中西药相互作用表与剂量上限表。上述四类问题不能依赖本工具作答。"
)


_CONSTRAINT_SPEC = ToolSpec(
    name="retrieve_safety_constraints",
    description=(
        "查询安全约束：给定药物、疗法、证候、疾病或人群情境，返回图谱中记载的"
        "禁忌项（CONTRAINDICATED_FOR）与慎用项（CAUTION_FOR），并给出原文依据句。"
        "同时返回该约束适用的安全情境（SafetyContext，如特定证候患者、孕妇、过敏体质）。"
        "注意：图谱不含十八反十九畏、妊娠禁忌药分级与剂量上限，这类问题本工具会明确返回未覆盖。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "要查询的药物/疗法/饮食项名称，或证候名、疾病名。",
            },
            "context": {
                "type": "string",
                "description": "可选，人群或情境关键词，例如 \"孕妇\"、\"儿童\"、\"过敏\"、\"肝肾功能\"。",
            },
            "top_k": {"type": "integer", "description": "返回条数上限，默认 15。"},
        },
        "required": [],
    },
    domains=_SAFETY_DOMAINS,
)


@REGISTRY.register(_CONSTRAINT_SPEC)
def retrieve_safety_constraints(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    entity = str(args.get("entity") or "").strip()
    context = str(args.get("context") or "").strip()
    limit = max(1, min(int(args.get("top_k") or 15), 40))

    if not entity and not context:
        return ToolResult(
            tool=_CONSTRAINT_SPEC.name,
            ok=False,
            coverage=Coverage.NOT_COVERED,
            error="必须提供 entity 或 context 之一。",
            caveats=[SAFETY_SCOPE_NOTE],
        )

    seeds: List[Any] = []
    if entity:
        node = kg.node(entity)
        seeds = [node] if node else resolve_entity(
            kg, entity, ctx.visible_types(), retriever=ctx.retriever, domain=ctx.domain
        )

    contexts: List[Any] = []
    if context:
        for hit in ctx.retriever.search(
            context,
            domain=ctx.domain,
            node_types=[NodeType.SAFETY_CONTEXT.value],
            top_k=limit,
        ):
            node = kg.node(hit.node_id)
            if node is not None and context in node.name:
                contexts.append(node)

    findings: List[Dict[str, Any]] = []
    for seed in seeds[:3]:
        findings.extend(_constraints_for(ctx, seed))
    for safety_context in contexts[:limit]:
        findings.extend(_constraints_of_context(ctx, safety_context))

    findings = _dedupe(findings)[:limit]
    payload: Dict[str, Any] = {
        "query": {"entity": entity or None, "context": context or None},
        "resolved": [node_brief(kg, s, with_sentence=False) for s in seeds[:3]],
        "matched_safety_contexts": [node_brief(kg, c, with_sentence=False) for c in contexts[:limit]],
        "findings": findings,
        "n_findings": len(findings),
    }
    if findings:
        payload["documents"] = documents_block(kg, [seed.id for seed in seeds[:2]])

    if not seeds and not contexts:
        return ToolResult(
            tool=_CONSTRAINT_SPEC.name,
            ok=True,
            coverage=Coverage.NOT_COVERED,
            data=payload,
            caveats=[
                f"图谱中没有与 {entity or context!r} 对应的安全条目。",
                SAFETY_SCOPE_NOTE,
            ],
        )
    if not findings:
        return ToolResult(
            tool=_CONSTRAINT_SPEC.name,
            ok=True,
            coverage=Coverage.EMPTY,
            data=payload,
            caveats=[
                "已在图谱中定位到该实体，但没有与之关联的禁忌/慎用记录。"
                "这只说明本图谱的来源文献未记载，不能推断为“无禁忌”。",
                SAFETY_SCOPE_NOTE,
            ],
        )
    return ToolResult(
        tool=_CONSTRAINT_SPEC.name,
        ok=True,
        coverage=Coverage.SUPPORTED,
        data=payload,
        caveats=[SAFETY_SCOPE_NOTE],
    )


_CONSTRAINT_EDGES = {EdgeType.CONTRAINDICATED_FOR.value, EdgeType.CAUTION_FOR.value}


def _constraints_for(ctx: ToolContext, seed: Any) -> List[Dict[str, Any]]:
    """Constraints where ``seed`` is the restricted item, or the thing restricted for."""
    kg = ctx.kg
    out: List[Dict[str, Any]] = []
    for member in kg.cluster(seed.id):
        for edge in kg.out_edges(member.id, _CONSTRAINT_EDGES):
            out.append(_finding(ctx, edge, kg.nodes[edge.source], kg.nodes[edge.target]))
        for edge in kg.in_edges(member.id, _CONSTRAINT_EDGES):
            out.append(_finding(ctx, edge, kg.nodes[edge.source], kg.nodes[edge.target]))
    return out


def _constraints_of_context(ctx: ToolContext, safety_context: Any) -> List[Dict[str, Any]]:
    kg = ctx.kg
    return [
        _finding(ctx, edge, kg.nodes[edge.source], kg.nodes[edge.target])
        for edge in kg.in_edges(safety_context.id, _CONSTRAINT_EDGES)
    ]


def _finding(ctx: ToolContext, edge: Any, source: Any, target: Any) -> Dict[str, Any]:
    kg = ctx.kg
    severity = "contraindicated" if edge.type == EdgeType.CONTRAINDICATED_FOR.value else "caution"
    return {
        "severity": severity,
        "restricted": node_brief(kg, source, with_sentence=False),
        "applies_to": node_brief(kg, target, with_sentence=False, with_attrs=("disease",)),
        "restriction_kind": _classify(source, target),
        "provenance": edge_evidence(edge, max_items=2),
    }


#: Heuristic classification so a model can tell a dietary rule from a drug rule.
_DIET_HINTS = ("食", "饮", "酒", "茶", "咖啡", "辛辣", "生冷", "油腻", "肥甘", "腥", "烟", "厚腻", "寒凉")
_PROCEDURE_HINTS = ("疗法", "针", "灸", "推拿", "熏蒸", "置换", "拔罐", "刮痧", "康复", "运动", "手术")


def _classify(source: Any, target: Any) -> str:
    if source.type == NodeType.HERB.value:
        return "herb"
    name = source.name
    if any(hint in name for hint in _PROCEDURE_HINTS):
        return "procedure"
    if any(hint in name for hint in _DIET_HINTS):
        return "diet"
    if source.type == NodeType.RESTRICTED_ITEM.value:
        return "other_restriction"
    return "unclassified"


def _dedupe(findings: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for finding in findings:
        key = (
            finding["severity"],
            finding["restricted"]["id"],
            finding["applies_to"]["id"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(finding))
    return out
