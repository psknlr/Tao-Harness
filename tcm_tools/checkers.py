"""Deterministic prescription-audit checkers.

These are the tools that must not be answered by an LLM: whether two drugs
duplicate each other, whether an item is restricted for a cohort, and whether a
herb needs special preparation are all decidable from the graph, and a rule
engine decides them identically for every model under test.

The design rule that matters most here is the honest ``NOT_COVERED`` verdict.
Two of the five checks below are **not** groundable in this knowledge graph:

* **Dose** -- the pharmacopoeia sub-graph carries basionym, part used, nature /
  taste / meridian and functions, but no 用法用量 field.  Nothing in the graph
  can decide whether 30 g of a herb is excessive.
* **Herb-herb incompatibility** -- there is no 十八反 / 十九畏 table, and the
  only ``CONTRAINDICATED_FOR`` edge with a Herb source is a single dietary
  instruction (薄荷 for 吐酸病).

Returning a confident "no problem found" for either would be a false negative
in a safety system.  Both therefore return ``Coverage.NOT_COVERED`` and say
what the graph *can* contribute instead.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from tcm_kg.normalize import DECOCTION_MARKERS, split_herb_annotation
from tcm_kg.schema import Domain, EdgeType, NodeType

from ._common import edge_evidence, node_brief, resolve_entity
from .base import REGISTRY, Coverage, ToolContext, ToolPhase, ToolResult, ToolSpec
from .medication import TOXICITY_MARKERS
from .safety import SAFETY_SCOPE_NOTE

_PA_DOMAINS = (Domain.SAFETY, Domain.PATHWAY, Domain.FULL)


def _resolve_drugs(ctx: ToolContext, names: Sequence[str], types: Sequence[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Resolve drug names to nodes, returning the unresolved ones separately."""
    resolved: Dict[str, Any] = {}
    unresolved: List[str] = []
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        base, _markers = split_herb_annotation(name)
        node = ctx.kg.node(name)
        matches = [node] if node else resolve_entity(ctx.kg, name, types)
        if not matches and base != name:
            matches = resolve_entity(ctx.kg, base, types)
        if matches:
            resolved[name] = matches[0]
        else:
            unresolved.append(name)
    return resolved, unresolved


# --------------------------------------------------------------------------- #
# check_dose  -- deliberately reports NOT_COVERED
# --------------------------------------------------------------------------- #

_DOSE_SPEC = ToolSpec(
    name="check_dose",
    description=(
        "剂量核查。重要：本知识图谱的药典条目不包含用法用量字段，"
        "因此本工具无法判断剂量是否超量，一律返回 not_covered。"
        "它会返回图谱确实拥有的、与剂量判断相关的旁证："
        "该药材的药典有毒标注（大毒/有毒/小毒）、基原部位、"
        "以及原文标注的特殊用法（如冲服、另煎）。"
        "遇到剂量类题目时，应依据你自身的药典知识作答，并明确说明图谱未提供剂量依据。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object"},
                "description": "待核查项，每项形如 {\"name\": \"附子\", \"dose\": \"30g\"}。dose 可省略。",
            }
        },
        "required": ["items"],
    },
    domains=_PA_DOMAINS,
    deterministic=True,
    phase=ToolPhase.VERIFICATION,
)


@REGISTRY.register(_DOSE_SPEC)
def check_dose(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    items = list(args.get("items") or [])
    names = [str(i.get("name") if isinstance(i, Mapping) else i) for i in items]
    resolved, unresolved = _resolve_drugs(
        ctx, names, [NodeType.HERB.value, NodeType.PHARMACOPOEIA_ENTRY.value]
    )

    findings: List[Dict[str, Any]] = []
    for raw, node in resolved.items():
        entry = node.attrs.get("pharmacopoeia_entry")
        nature = ""
        if isinstance(entry, Mapping):
            nature = str(entry.get("nature_taste_meridian") or "")
        if not nature:
            for _e, target in ctx.kg.neighbours(
                node.id, {EdgeType.REGISTERED_IN_PHARMACOPOEIA.value}
            ):
                nature = str(target.get("nature_taste_meridian") or "")
                break
        toxicity = next((m for m in TOXICITY_MARKERS if m in nature), None)
        findings.append(
            {
                "input": raw,
                "resolved": node_brief(ctx.kg, node, with_sentence=False),
                "dose_range_in_graph": None,
                "dose_verdict": "not_covered",
                "toxicity_flag": toxicity,
                "nature_taste_meridian": nature or None,
                "preparation_markers": sorted(ctx.kg.preparation_markers(node.id)),
            }
        )

    return ToolResult(
        tool=_DOSE_SPEC.name,
        ok=True,
        coverage=Coverage.NOT_COVERED,
        data={"findings": findings, "unresolved": unresolved},
        caveats=[
            "本图谱不含用法用量数据，无法判定任何剂量是否超量或不足。",
            "有毒标注可提示需谨慎，但不能替代剂量上限。",
            "作答时请明确区分“图谱未覆盖”与“图谱认为无问题”。",
        ],
    )


# --------------------------------------------------------------------------- #
# check_duplicate_medication
# --------------------------------------------------------------------------- #

_DUP_SPEC = ToolSpec(
    name="check_duplicate_medication",
    description=(
        "重复用药核查（确定性）。输入一张处方中的全部药品名（饮片、方剂、中成药皆可），"
        "返回：①同一药材的别名/炮制品重复（如 瓜蒌 与 全瓜蒌）；"
        "②多个方剂之间共有的饮片及其重叠程度。"
        "限制：图谱不收录中成药的处方组成，因此涉及中成药成分重复的判断会标记为未覆盖。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "处方中的药品名列表。",
            }
        },
        "required": ["items"],
    },
    domains=_PA_DOMAINS,
    deterministic=True,
    phase=ToolPhase.VERIFICATION,
)


@REGISTRY.register(_DUP_SPEC)
def check_duplicate_medication(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    names = [str(n) for n in (args.get("items") or [])]
    resolved, unresolved = _resolve_drugs(
        ctx,
        names,
        [
            NodeType.HERB.value,
            NodeType.FORMULA.value,
            NodeType.PATENT_MEDICINE.value,
        ],
    )

    # 1) same canonical entity listed twice under different surface forms
    clusters: Dict[str, List[str]] = defaultdict(list)
    for raw, node in resolved.items():
        clusters[kg.canonical_id(node.id)].append(raw)
    alias_duplicates = [
        {
            "canonical": node_brief(kg, kg.node(head), with_sentence=False),
            "listed_as": sorted(raws),
            "reason": "同一药材的别名或炮制品在同一处方中重复出现",
        }
        for head, raws in clusters.items()
        if len(raws) > 1 and kg.node(head) is not None
    ]

    # 2) herb overlap between prescribed formulas
    formulas = {
        raw: node for raw, node in resolved.items() if node.type == NodeType.FORMULA.value
    }
    formula_herbs: Dict[str, Set[str]] = {}
    for raw, node in formulas.items():
        formula_herbs[raw] = {
            kg.canonical_id(herb.id)
            for _e, herb in kg.neighbours(node.id, {EdgeType.CONTAINS_HERB.value})
        }
    shared: List[Dict[str, Any]] = []
    formula_names = sorted(formula_herbs)
    for i, a in enumerate(formula_names):
        for b in formula_names[i + 1 :]:
            common = formula_herbs[a] & formula_herbs[b]
            if not common:
                continue
            union = formula_herbs[a] | formula_herbs[b]
            shared.append(
                {
                    "formulas": [a, b],
                    "shared_herbs": sorted(
                        kg.node(h).name for h in common if kg.node(h)
                    ),
                    "n_shared": len(common),
                    "jaccard": round(len(common) / len(union), 3) if union else 0.0,
                }
            )

    # 3) loose herbs that also appear inside a prescribed formula
    loose = {
        raw: node for raw, node in resolved.items() if node.type == NodeType.HERB.value
    }
    herb_in_formula: List[Dict[str, Any]] = []
    for raw, node in loose.items():
        canonical = kg.canonical_id(node.id)
        for formula_raw, herbs in formula_herbs.items():
            if canonical in herbs:
                herb_in_formula.append(
                    {
                        "herb": raw,
                        "already_in_formula": formula_raw,
                        "reason": "单味饮片与所开方剂的组成重复",
                    }
                )

    patents = [raw for raw, node in resolved.items() if node.type == NodeType.PATENT_MEDICINE.value]
    findings = alias_duplicates + shared + herb_in_formula
    coverage = Coverage.SUPPORTED if resolved else Coverage.EMPTY
    caveats: List[str] = []
    if patents:
        coverage = Coverage.PARTIAL
        caveats.append(
            f"处方含中成药 {patents}；图谱不收录中成药组成，"
            "无法判断其与汤剂或其他中成药的成分重复。"
        )
    if unresolved:
        caveats.append(f"以下药品未在图谱中找到，未参与判定：{unresolved}")

    return ToolResult(
        tool=_DUP_SPEC.name,
        ok=True,
        coverage=coverage,
        data={
            "n_inputs": len(names),
            "alias_duplicates": alias_duplicates,
            "formula_overlap": shared,
            "herb_inside_formula": herb_in_formula,
            "n_findings": len(findings),
            "unresolved": unresolved,
        },
        caveats=caveats,
    )


# --------------------------------------------------------------------------- #
# check_restricted_item
# --------------------------------------------------------------------------- #

_RESTRICT_SPEC = ToolSpec(
    name="check_restricted_item",
    description=(
        "禁忌项核查（确定性）。输入药品/饮食/疗法项与患者情境"
        "（证候、疾病或人群，如“孕妇”“脾胃虚弱证”），"
        "返回图谱中记载的禁忌与慎用条目及原文依据。"
        "图谱的禁忌知识以饮食调护与操作禁忌为主，不含配伍禁忌表与妊娠禁忌药分级表。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "待核查的药品、饮食或疗法项。",
            },
            "patient_context": {
                "type": "string",
                "description": "患者情境：证候名、疾病名或人群（孕妇/儿童/过敏体质等）。",
            },
        },
        "required": ["items"],
    },
    domains=_PA_DOMAINS,
    deterministic=True,
    phase=ToolPhase.VERIFICATION,
)


@REGISTRY.register(_RESTRICT_SPEC)
def check_restricted_item(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    names = [str(n) for n in (args.get("items") or [])]
    patient_context = str(args.get("patient_context") or "").strip()

    context_nodes: List[Any] = []
    if patient_context:
        context_nodes = resolve_entity(
            kg,
            patient_context,
            [NodeType.SYNDROME.value, NodeType.SAFETY_CONTEXT.value, NodeType.DISEASE.value],
        )
        if not context_nodes:
            for hit in ctx.retriever.search(
                patient_context,
                domain=ctx.domain,
                node_types=[NodeType.SAFETY_CONTEXT.value, NodeType.SYNDROME.value],
                top_k=8,
            ):
                node = kg.node(hit.node_id)
                if node is not None and patient_context in node.name:
                    context_nodes.append(node)

    context_ids = {n.id for n in context_nodes}
    findings: List[Dict[str, Any]] = []
    unresolved: List[str] = []

    for raw in names:
        matches = resolve_entity(
            kg,
            raw,
            [NodeType.RESTRICTED_ITEM.value, NodeType.HERB.value, NodeType.EXTERNAL_THERAPY.value],
        )
        if not matches:
            unresolved.append(raw)
            continue
        hits: List[Dict[str, Any]] = []
        for member in kg.cluster(matches[0].id):
            for edge in kg.out_edges(
                member.id,
                {EdgeType.CONTRAINDICATED_FOR.value, EdgeType.CAUTION_FOR.value},
            ):
                target = kg.nodes[edge.target]
                applies = (not context_ids) or (target.id in context_ids)
                hits.append(
                    {
                        "severity": (
                            "contraindicated"
                            if edge.type == EdgeType.CONTRAINDICATED_FOR.value
                            else "caution"
                        ),
                        "applies_to": node_brief(kg, target, with_sentence=False, with_attrs=("disease",)),
                        "matches_patient_context": applies,
                        "provenance": edge_evidence(edge, max_items=1),
                    }
                )
        findings.append(
            {
                "item": raw,
                "resolved": node_brief(kg, matches[0], with_sentence=False),
                "restrictions": hits,
                "n_restrictions": len(hits),
                "n_matching_context": sum(1 for h in hits if h["matches_patient_context"]),
            }
        )

    any_hit = any(f["n_restrictions"] for f in findings)
    coverage = Coverage.SUPPORTED if any_hit else (Coverage.EMPTY if findings else Coverage.NOT_COVERED)
    caveats = [SAFETY_SCOPE_NOTE]
    if not any_hit and findings:
        caveats.insert(
            0,
            "图谱中这些条目没有禁忌记录。这只表示来源文献未记载，"
            "不能推断为临床上无禁忌。",
        )
    if unresolved:
        caveats.append(f"未在图谱中找到：{unresolved}")

    return ToolResult(
        tool=_RESTRICT_SPEC.name,
        ok=True,
        coverage=coverage,
        data={
            "patient_context": patient_context or None,
            "resolved_contexts": [node_brief(kg, n, with_sentence=False) for n in context_nodes[:6]],
            "findings": findings,
            "unresolved": unresolved,
        },
        caveats=caveats,
    )


# --------------------------------------------------------------------------- #
# check_combination -- deliberately reports NOT_COVERED for incompatibility
# --------------------------------------------------------------------------- #

_COMBO_SPEC = ToolSpec(
    name="check_combination",
    description=(
        "配伍核查。重要：本图谱不含十八反/十九畏配伍禁忌表，"
        "因此无法判定配伍禁忌，禁忌结论一律返回 not_covered。"
        "工具会返回图谱确实拥有的旁证：这些药材是否曾在同一张图谱方剂中共同出现"
        "（共现是相容性的弱证据，不是安全性结论），以及各药材的有毒标注。"
        "配伍禁忌题目应依据你自身知识作答，并说明图谱未提供依据。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "拟联用的药材名列表（两个及以上）。",
            }
        },
        "required": ["items"],
    },
    domains=_PA_DOMAINS,
    deterministic=True,
    phase=ToolPhase.VERIFICATION,
)


@REGISTRY.register(_COMBO_SPEC)
def check_combination(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    names = [str(n) for n in (args.get("items") or [])]
    if len(names) < 2:
        return ToolResult(
            tool=_COMBO_SPEC.name,
            ok=False,
            coverage=Coverage.NOT_COVERED,
            error="至少需要两个药材名。",
        )
    resolved, unresolved = _resolve_drugs(ctx, names, [NodeType.HERB.value])

    formulas_by_herb: Dict[str, Set[str]] = {}
    for raw, node in resolved.items():
        containers: Set[str] = set()
        for member in kg.cluster(node.id):
            for edge in kg.in_edges(member.id, {EdgeType.CONTAINS_HERB.value}):
                containers.add(edge.source)
        formulas_by_herb[raw] = containers

    pairs: List[Dict[str, Any]] = []
    keys = sorted(formulas_by_herb)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            shared = formulas_by_herb[a] & formulas_by_herb[b]
            pairs.append(
                {
                    "pair": [a, b],
                    "incompatibility_verdict": "not_covered",
                    "co_occurs_in_graph_formulas": sorted(
                        kg.node(f).name for f in list(shared)[:8] if kg.node(f)
                    ),
                    "n_co_occurrences": len(shared),
                    "interpretation": (
                        "两药在图谱方剂中共同出现，属于相容性的弱旁证"
                        if shared
                        else "两药在图谱方剂中未共同出现；这既不能证明相容也不能证明相反"
                    ),
                }
            )

    toxic: List[Dict[str, Any]] = []
    for raw, node in resolved.items():
        entry = node.attrs.get("pharmacopoeia_entry")
        nature = str(entry.get("nature_taste_meridian") or "") if isinstance(entry, Mapping) else ""
        flag = next((m for m in TOXICITY_MARKERS if m in nature), None)
        if flag:
            toxic.append({"herb": raw, "toxicity_flag": flag})

    return ToolResult(
        tool=_COMBO_SPEC.name,
        ok=True,
        coverage=Coverage.NOT_COVERED,
        data={"pairs": pairs, "toxicity_flags": toxic, "unresolved": unresolved},
        caveats=[
            "图谱不含十八反、十九畏或中西药相互作用表，配伍禁忌无法由图谱判定。",
            "共现证据来自国家诊疗方案的推荐方药，不构成安全性背书。",
        ],
    )


# --------------------------------------------------------------------------- #
# check_decoction_requirement
# --------------------------------------------------------------------------- #

_DECOCT_SPEC = ToolSpec(
    name="check_decoction_requirement",
    description=(
        "特殊煎煮/用法核查（确定性）。输入药材名，返回图谱原文中标注过的特殊用法"
        "（先煎、后下、包煎、另煎、烊化、冲服等）及其原文依据句。"
        "标注来自诊疗方案推荐方药中该药材名后紧跟的括注，按位置归属，不会把同句中"
        "其他药材的括注错记到本药材上。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "待核查的药材名列表。",
            },
            "claimed_requirement": {
                "type": "string",
                "description": "可选，题目声称的用法（如 \"先煎\"），用于直接给出支持/未支持判定。",
            },
        },
        "required": ["items"],
    },
    domains=_PA_DOMAINS,
    deterministic=True,
    phase=ToolPhase.VERIFICATION,
)


@REGISTRY.register(_DECOCT_SPEC)
def check_decoction_requirement(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    names = [str(n) for n in (args.get("items") or [])]
    claimed = str(args.get("claimed_requirement") or "").strip()
    resolved, unresolved = _resolve_drugs(ctx, names, [NodeType.HERB.value])

    findings: List[Dict[str, Any]] = []
    for raw, node in resolved.items():
        markers = kg.preparation_markers(node.id)
        record: Dict[str, Any] = {
            "item": raw,
            "resolved": node_brief(kg, node, with_sentence=False),
            "attested_requirements": sorted(markers),
            "evidence": {k: v[:2] for k, v in markers.items()},
        }
        if claimed:
            if claimed in markers:
                record["claim_verdict"] = "supported"
            elif markers:
                record["claim_verdict"] = "contradicted_by_graph"
                record["detail"] = (
                    f"图谱只标注了 {sorted(markers)}，未见 {claimed!r}。"
                )
            else:
                record["claim_verdict"] = "not_covered"
                record["detail"] = "图谱没有该药材的特殊用法标注。"
        findings.append(record)

    attested = any(f["attested_requirements"] for f in findings)
    coverage = (
        Coverage.SUPPORTED
        if attested
        else (Coverage.EMPTY if findings else Coverage.NOT_COVERED)
    )
    caveats = [
        "标注来源于国家中医诊疗方案的推荐方药括注，覆盖面窄于药典："
        "图谱未标注不等于该药材无特殊煎法要求。",
    ]
    if unresolved:
        caveats.append(f"未在图谱中找到：{unresolved}")
    return ToolResult(
        tool=_DECOCT_SPEC.name,
        ok=True,
        coverage=coverage,
        data={
            "findings": findings,
            "recognised_markers": list(DECOCTION_MARKERS),
            "unresolved": unresolved,
        },
        caveats=caveats,
    )
