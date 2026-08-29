"""Medication and pharmacopoeia tools (tools 5 and 7).

Available in the prescription-safety domain only.  The clinical-reasoning
domain cannot reach them, which is what stops an SDT agent from inverting the
syndrome -> formula mapping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from tcm_kg.schema import Domain, EdgeType, NodeType

from ._common import documents_block, edge_evidence, node_brief, resolve_entity
from .base import REGISTRY, Coverage, ToolContext, ToolResult, ToolSpec

_SAFETY_DOMAINS = (Domain.SAFETY, Domain.PATHWAY, Domain.FULL)

_DRUG_TYPES = (
    NodeType.FORMULA.value,
    NodeType.HERB.value,
    NodeType.PATENT_MEDICINE.value,
    NodeType.EXTERNAL_THERAPY.value,
)

#: Facts the pharmacopoeia sub-graph does *not* carry.  Stated once, here, and
#: echoed in every result so a model cannot mistake silence for permission.
PHARMACOPOEIA_GAPS = (
    "本图谱的药典条目不包含【用法用量/剂量范围】字段，无法据此判断剂量是否超量；",
    "不包含【十八反/十九畏配伍禁忌表】，无法据此判断配伍禁忌；",
    "不包含【妊娠禁忌药分级表】，妊娠相关信息仅来自诊疗方案原文中的零散表述。",
)


# --------------------------------------------------------------------------- #
# Tool 5: retrieve_medication_knowledge
# --------------------------------------------------------------------------- #

_MED_SPEC = ToolSpec(
    name="retrieve_medication_knowledge",
    description=(
        "查询方剂、中药、中成药或非药物疗法的图谱知识。"
        "对方剂返回其组成饮片（含原文标注的特殊煎法，如先煎/后下/烊化/冲服）；"
        "对中药返回含有该药的方剂、直接使用该药的证候、以及药典登记情况；"
        "对中成药返回使用它的证候与疾病。"
        "用于处方审核中的品种、组成、重复用药与用法判断。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "方剂名、中药名、中成药名或实体 id。"},
            "kind": {
                "type": "string",
                "description": "可选，限定类型：Formula / Herb / PatentMedicine / ExternalTherapy。留空自动判断。",
            },
            "max_related": {"type": "integer", "description": "每类关联条目的上限，默认 15。"},
        },
        "required": ["name"],
    },
    domains=_SAFETY_DOMAINS,
)


@REGISTRY.register(_MED_SPEC)
def retrieve_medication_knowledge(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    name = str(args.get("name", "")).strip()
    kind = str(args.get("kind") or "").strip() or None
    limit = max(1, min(int(args.get("max_related") or 15), 40))
    types = [kind] if kind else list(_DRUG_TYPES)

    node = kg.node(name)
    matches = [node] if node else resolve_entity(
        kg, name, types, retriever=ctx.retriever, domain=ctx.domain
    )
    if not matches:
        return ToolResult(
            tool=_MED_SPEC.name,
            ok=True,
            coverage=Coverage.EMPTY,
            data={"name": name, "resolved": None},
            caveats=[
                f"图谱未收录 {name!r}。图谱的药物知识来自国家中医诊疗方案与临床路径，"
                "覆盖范围有限，未收录不代表该药不存在或不可用。"
            ],
        )

    target = matches[0]
    ctx.assert_visible(target.type)
    payload: Dict[str, Any] = {
        "resolved": node_brief(kg, target),
        "aliases": [n.name for n in kg.cluster(target.id) if n.id != target.id],
        "alternatives": [node_brief(kg, n, with_sentence=False) for n in matches[1:4]],
    }

    if target.type == NodeType.FORMULA.value:
        components: List[Dict[str, Any]] = []
        for edge, herb in kg.neighbours(target.id, {EdgeType.CONTAINS_HERB.value}):
            entry = node_brief(kg, herb, with_sentence=False)
            markers = kg.preparation_markers(herb.id)
            if markers:
                entry["preparation"] = {k: v[:1] for k, v in markers.items()}
            entry["provenance"] = edge_evidence(edge, max_items=1)
            components.append(entry)
        payload["composition"] = components[:60]
        payload["n_components"] = len(components)
        payload["used_by_syndromes"] = [
            node_brief(kg, syn, with_sentence=False)
            for _e, syn in kg.neighbours(target.id, {EdgeType.USES_FORMULA.value}, direction="in")
        ][:limit]

    elif target.type == NodeType.HERB.value:
        payload["preparation"] = kg.preparation_markers(target.id)
        payload["in_formulas"] = [
            node_brief(kg, formula, with_sentence=False)
            for _e, formula in kg.neighbours(
                target.id, {EdgeType.CONTAINS_HERB.value}, direction="in"
            )
        ][:limit]
        payload["used_directly_by_syndromes"] = [
            node_brief(kg, syn, with_sentence=False)
            for _e, syn in kg.neighbours(
                target.id, {EdgeType.USES_HERB_DIRECT.value}, direction="in"
            )
        ][:limit]
        entry = target.attrs.get("pharmacopoeia_entry")
        payload["pharmacopoeia_registered"] = bool(entry)
        if isinstance(entry, Mapping):
            payload["pharmacopoeia_summary"] = {
                k: entry.get(k)
                for k in ("canonical_name", "nature_taste_meridian", "pharmacopoeial_functions")
                if entry.get(k)
            }

    elif target.type == NodeType.PATENT_MEDICINE.value:
        users = kg.neighbours(target.id, {EdgeType.USES_PATENT_MEDICINE.value}, direction="in")
        payload["used_by_syndromes"] = [node_brief(kg, syn) for _e, syn in users][:limit]
        diseases: List[Dict[str, Any]] = []
        for _e, syn in users[:6]:
            for edge in kg.in_edges(syn.id, {EdgeType.HAS_SYNDROME.value}):
                diseases.append(node_brief(kg, kg.nodes[edge.source], with_sentence=False))
        payload["indicated_diseases"] = _dedupe_records(diseases)[:limit]
        payload["composition_available"] = False

    elif target.type == NodeType.EXTERNAL_THERAPY.value:
        payload["used_by_syndromes"] = [
            node_brief(kg, syn, with_sentence=False)
            for _e, syn in kg.neighbours(
                target.id, {EdgeType.USES_EXTERNAL_THERAPY.value}, direction="in"
            )
        ][:limit]

    payload["documents"] = documents_block(kg, [target.id])

    caveats = list(PHARMACOPOEIA_GAPS)
    if target.type == NodeType.PATENT_MEDICINE.value:
        caveats.append(
            "图谱不收录中成药的处方组成，因此无法用图谱判断中成药之间或"
            "中成药与汤剂之间的成分重复。"
        )
    return ToolResult(
        tool=_MED_SPEC.name,
        ok=True,
        coverage=Coverage.SUPPORTED,
        data=payload,
        caveats=caveats,
    )


def _dedupe_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for record in records:
        key = record.get("id")
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(record))
    return out


# --------------------------------------------------------------------------- #
# Tool 7: retrieve_pharmacopeia_entry
# --------------------------------------------------------------------------- #

_PHARMA_SPEC = ToolSpec(
    name="retrieve_pharmacopeia_entry",
    description=(
        "查询《中国药典》登记条目：药材基原部位、性味归经（含有毒标注）、"
        "功能主治、药典页码与常用别名。"
        "用于品种鉴别、别名核对、有毒药材识别与功能主治核对。"
        "重要限制：本图谱的药典条目【不含用法用量字段】，因此不能用于判断剂量是否超量。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "herb": {"type": "string", "description": "中药名、别名或药典条目 id。"},
            "include_related": {
                "type": "boolean",
                "description": "是否返回同一基原的炮制品/别名条目，默认 true。",
            },
        },
        "required": ["herb"],
    },
    domains=_SAFETY_DOMAINS,
)

#: Toxicity wording used by the pharmacopoeia, strongest first.
TOXICITY_MARKERS = ("大毒", "有大毒", "有毒", "小毒", "有小毒")


@REGISTRY.register(_PHARMA_SPEC)
def retrieve_pharmacopeia_entry(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    name = str(args.get("herb", "")).strip()
    include_related = bool(args.get("include_related", True))

    node = kg.node(name)
    matches = [node] if node else resolve_entity(
        kg,
        name,
        [NodeType.PHARMACOPOEIA_ENTRY.value, NodeType.HERB.value],
        retriever=ctx.retriever,
        domain=ctx.domain,
    )
    if not matches:
        return ToolResult(
            tool=_PHARMA_SPEC.name,
            ok=True,
            coverage=Coverage.EMPTY,
            data={"herb": name, "entry": None},
            caveats=[
                f"图谱未收录 {name!r} 的药典条目。图谱共收录 "
                f"{len(kg.of_type(NodeType.PHARMACOPOEIA_ENTRY.value))} 条药典条目，"
                "远少于药典全本，未收录不等于该药材未被药典收载。"
            ],
        )

    entries = _collect_entries(ctx, matches, include_related)
    if not entries:
        herb = matches[0]
        return ToolResult(
            tool=_PHARMA_SPEC.name,
            ok=True,
            coverage=Coverage.PARTIAL,
            data={"herb": name, "resolved": node_brief(kg, herb, with_sentence=False), "entry": None},
            caveats=[
                f"{herb.name} 在图谱中存在，但没有关联到药典条目，"
                "无法提供性味归经与功能主治。"
            ],
        )

    payload = {
        "resolved": node_brief(kg, matches[0], with_sentence=False),
        "entries": entries,
        "toxicity_flagged": [e["name"] for e in entries if e.get("toxicity")],
    }
    return ToolResult(
        tool=_PHARMA_SPEC.name,
        ok=True,
        coverage=Coverage.SUPPORTED,
        data=payload,
        caveats=list(PHARMACOPOEIA_GAPS),
    )


def _collect_entries(
    ctx: ToolContext, matches: Sequence[Any], include_related: bool
) -> List[Dict[str, Any]]:
    kg = ctx.kg
    seen: set = set()
    out: List[Dict[str, Any]] = []
    roots = list(matches)
    if include_related:
        for match in list(matches):
            roots.extend(kg.cluster(match.id))
    for node in roots:
        targets = []
        if node.type == NodeType.PHARMACOPOEIA_ENTRY.value:
            targets.append(node)
        else:
            targets.extend(
                target
                for _e, target in kg.neighbours(
                    node.id, {EdgeType.REGISTERED_IN_PHARMACOPOEIA.value}
                )
            )
        for target in targets:
            if target.id in seen:
                continue
            seen.add(target.id)
            nature = str(target.get("nature_taste_meridian") or "")
            record = {
                "id": target.id,
                "name": target.name,
                "source": target.get("source"),
                "pharmacopoeia_page": target.get("pharmacopoeia_page"),
                "part_used": target.get("part_used"),
                "nature_taste_meridian": nature,
                "pharmacopoeial_functions": target.get("pharmacopoeial_functions"),
                "common_aliases": list(target.get("common_aliases") or []),
                "dosage_range": None,
                "dosage_available": False,
            }
            toxicity = next((m for m in TOXICITY_MARKERS if m in nature), None)
            if toxicity:
                record["toxicity"] = toxicity
            out.append(record)
    return out
