"""Clinical-reasoning tools (tools 1-4).

These four are the entire tool surface an SDT agent sees.  Between them they
implement the two-stage retrieval the study design calls for -- semantic anchor
retrieval, then typed graph expansion -- without ever exposing a formula, a
herb or a patent medicine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from tcm_kg.schema import Domain, EdgeType, NodeType
from tcm_kg.store import Node

from ._common import doc_brief, documents_block, edge_evidence, node_brief, resolve_entity
from .base import REGISTRY, Coverage, ToolContext, ToolResult, ToolSpec

_ALL_DOMAINS = (Domain.CLINICAL, Domain.SAFETY, Domain.PATHWAY, Domain.FULL)


# --------------------------------------------------------------------------- #
# Tool 1: search_tcm_entities
# --------------------------------------------------------------------------- #

_SEARCH_SPEC = ToolSpec(
    name="search_tcm_entities",
    description=(
        "语义检索知识图谱中的实体。输入一段自由文本（如患者主诉、症状、舌脉描述、"
        "药品名或规则关键词），返回最相关的实体及其来源证据句。"
        "这是进入图谱的唯一入口：先用本工具锚定实体，再用其他工具沿图谱展开。"
        "返回的每个实体都带有 id，后续工具应使用该 id 或实体名。"
        "注意：图谱中不存在“症状”和“病机”实体，症状只能通过实体的来源证据句间接匹配。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索文本。建议只保留临床特征（症状、舌象、脉象）或药品/规则关键词，去掉年龄性别等噪声。",
            },
            "entity_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选，限定实体类型，例如 [\"Syndrome\"]、[\"Disease\",\"DiseaseSubtype\"]。留空则检索当前域内全部类型。",
            },
            "anchors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选，已确定的实体 id 列表。提供后会叠加图结构相关性与文献共现证据重排序。",
            },
            "top_k": {
                "type": "integer",
                "description": "返回条数，默认 8，最大 20。",
            },
        },
        "required": ["query"],
    },
    domains=_ALL_DOMAINS,
)


@REGISTRY.register(_SEARCH_SPEC)
def search_tcm_entities(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    query = str(args.get("query", "")).strip()
    requested = args.get("entity_types") or None
    if requested is not None:
        requested = [str(t) for t in requested]
    visible = ctx.visible_types(requested)
    if requested and not visible:
        return ToolResult(
            tool=_SEARCH_SPEC.name,
            ok=False,
            coverage=Coverage.NOT_COVERED,
            error=(
                f"实体类型 {requested} 在当前访问域 {ctx.domain.value} 中不可见。"
                f"可用类型: {ctx.visible_types()}"
            ),
            caveats=[ctx.policy.rationale],
        )
    top_k = max(1, min(int(args.get("top_k") or ctx.retriever.params.top_k), 20))
    hits = ctx.retriever.search(
        query,
        domain=ctx.domain,
        node_types=visible,
        anchors=[str(a) for a in (args.get("anchors") or [])],
        top_k=top_k,
    )
    if not hits:
        return ToolResult(
            tool=_SEARCH_SPEC.name,
            ok=True,
            coverage=Coverage.EMPTY,
            data={"query": query, "results": []},
            caveats=["检索无命中；图谱可能未收录该主题，请勿据此断定该知识不存在。"],
        )
    return ToolResult(
        tool=_SEARCH_SPEC.name,
        ok=True,
        coverage=Coverage.SUPPORTED,
        data={"query": query, "results": [h.to_dict() for h in hits]},
    )


# --------------------------------------------------------------------------- #
# Tool 2: retrieve_clinical_context
# --------------------------------------------------------------------------- #

_CONTEXT_SPEC = ToolSpec(
    name="retrieve_clinical_context",
    description=(
        "以疾病、疾病亚型或临床路径阶段为锚点，返回其结构化诊疗上下文子图："
        "所属科室、疾病亚型、该疾病/亚型下的全部候选证候，以及临床路径阶段"
        "（含入径标准、出径标准、监测项目、疗效指标）。"
        "用于把病例约束到一个明确的疾病范围内，从而收敛证候候选空间。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "疾病名、疾病亚型名或实体 id。例如 \"心悸（心律失常-室性早搏）\" 或 \"Disease::心悸（心律失常-室性早搏）\"。",
            },
            "include_pathway": {
                "type": "boolean",
                "description": "是否返回临床路径阶段，默认 true。",
            },
            "max_syndromes": {
                "type": "integer",
                "description": "最多返回的证候数，默认 12。",
            },
        },
        "required": ["entity"],
    },
    domains=_ALL_DOMAINS,
)


@REGISTRY.register(_CONTEXT_SPEC)
def retrieve_clinical_context(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    name = str(args.get("entity", "")).strip()
    include_pathway = bool(args.get("include_pathway", True))
    max_syndromes = max(1, min(int(args.get("max_syndromes") or 12), 40))

    node = kg.node(name)
    candidates = [node] if node else resolve_entity(
        kg,
        name,
        [NodeType.DISEASE.value, NodeType.DISEASE_SUBTYPE.value, NodeType.PATHWAY_STAGE.value],
        retriever=ctx.retriever,
        domain=ctx.domain,
    )
    if not candidates:
        return ToolResult(
            tool=_CONTEXT_SPEC.name,
            ok=True,
            coverage=Coverage.EMPTY,
            data={"entity": name, "resolved": None},
            caveats=[f"图谱中未找到名为 {name!r} 的疾病或亚型；请先用 search_tcm_entities 锚定。"],
        )

    anchor = candidates[0]
    ctx.assert_visible(anchor.type)

    disease = anchor
    if anchor.type == NodeType.DISEASE_SUBTYPE.value:
        parents = kg.in_edges(anchor.id, {EdgeType.HAS_SUBTYPE.value})
        if parents:
            disease = kg.nodes[parents[0].source]
    elif anchor.type == NodeType.PATHWAY_STAGE.value:
        parents = kg.in_edges(anchor.id, {EdgeType.HAS_PATHWAY_STAGE.value})
        if parents:
            disease = kg.nodes[parents[0].source]

    payload: Dict[str, Any] = {
        "resolved": node_brief(kg, anchor),
        "disease": node_brief(kg, disease, with_attrs=("tcm_name", "western_name")),
        "alternatives": [node_brief(kg, n, with_sentence=False) for n in candidates[1:4]],
    }

    payload["departments"] = [
        node_brief(kg, target, with_sentence=False)
        for _edge, target in kg.neighbours(disease.id, {EdgeType.BELONGS_TO_DEPARTMENT.value})
    ]

    subtypes: List[Dict[str, Any]] = []
    for edge, target in kg.neighbours(disease.id, {EdgeType.HAS_SUBTYPE.value}):
        entry = node_brief(kg, target, with_sentence=False)
        entry["syndromes"] = [
            node_brief(kg, syn)
            for _e, syn in kg.neighbours(target.id, {EdgeType.SUBTYPE_HAS_SYNDROME.value})
        ][:max_syndromes]
        subtypes.append(entry)
    payload["subtypes"] = subtypes

    syndromes: List[Dict[str, Any]] = []
    for edge, target in kg.neighbours(disease.id, {EdgeType.HAS_SYNDROME.value}):
        entry = node_brief(kg, target)
        entry["provenance"] = edge_evidence(edge)
        syndromes.append(entry)
    payload["syndromes"] = syndromes[:max_syndromes]

    if include_pathway:
        stages = [
            node_brief(
                kg,
                target,
                with_attrs=(
                    "order",
                    "variant",
                    "entry_criteria",
                    "exit_criteria",
                    "monitoring_items",
                    "outcome_indicators",
                ),
            )
            for _e, target in kg.neighbours(disease.id, {EdgeType.HAS_PATHWAY_STAGE.value})
        ]
        stages.sort(key=lambda s: (s.get("variant") or "", s.get("order") or 0))
        payload["pathway_stages"] = stages[:12]

    payload["documents"] = documents_block(kg, [disease.id] + [s["id"] for s in syndromes[:3]])

    coverage = Coverage.SUPPORTED if syndromes or subtypes else Coverage.PARTIAL
    caveats: List[str] = []
    if not syndromes and not subtypes:
        caveats.append("该疾病在图谱中没有关联证候，无法用于证候收敛。")
    return ToolResult(
        tool=_CONTEXT_SPEC.name, ok=True, coverage=coverage, data=payload, caveats=caveats
    )


# --------------------------------------------------------------------------- #
# Tool 3: retrieve_syndrome_evidence
# --------------------------------------------------------------------------- #

_SYNDROME_SPEC = ToolSpec(
    name="retrieve_syndrome_evidence",
    description=(
        "查询一个证候的全部图谱证据：该证候在诊疗方案原文中的定义句"
        "（通常包含主症、次症、舌象、脉象），出现该证候的疾病与疾病亚型，"
        "相关临床路径阶段，以及来源文献。"
        "用于核对候选证候与病例临床特征是否吻合——这是把证候候选转为最终判断的关键一步。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "syndrome": {
                "type": "string",
                "description": "证候名或实体 id，例如 \"心虚胆怯证\"。",
            },
            "disease_filter": {
                "type": "string",
                "description": "可选，只返回该疾病范围内的关联信息。",
            },
        },
        "required": ["syndrome"],
    },
    domains=_ALL_DOMAINS,
)


@REGISTRY.register(_SYNDROME_SPEC)
def retrieve_syndrome_evidence(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    name = str(args.get("syndrome", "")).strip()
    disease_filter = str(args.get("disease_filter") or "").strip()

    node = kg.node(name)
    matches = [node] if node and node.type == NodeType.SYNDROME.value else resolve_entity(
        kg, name, [NodeType.SYNDROME.value], retriever=ctx.retriever, domain=ctx.domain
    )
    if not matches:
        return ToolResult(
            tool=_SYNDROME_SPEC.name,
            ok=True,
            coverage=Coverage.EMPTY,
            data={"syndrome": name, "resolved": None},
            caveats=[
                f"图谱未收录证候 {name!r}。图谱共收录 "
                f"{len(kg.of_type(NodeType.SYNDROME.value))} 个证候，未收录不等于该证候不成立。"
            ],
        )

    syndrome = matches[0]
    ctx.assert_visible(syndrome.type)
    payload: Dict[str, Any] = {
        "resolved": node_brief(kg, syndrome, max_sentence=400),
        "definition_sentence": syndrome.sentence(),
    }

    diseases: List[Dict[str, Any]] = []
    for edge in kg.in_edges(syndrome.id, {EdgeType.HAS_SYNDROME.value}):
        parent = kg.nodes[edge.source]
        if disease_filter and disease_filter not in parent.name:
            continue
        entry = node_brief(kg, parent, with_sentence=False, with_attrs=("tcm_name", "western_name"))
        entry["provenance"] = edge_evidence(edge)
        diseases.append(entry)
    payload["diseases"] = diseases[:10]

    subtypes: List[Dict[str, Any]] = []
    for edge in kg.in_edges(syndrome.id, {EdgeType.SUBTYPE_HAS_SYNDROME.value}):
        parent = kg.nodes[edge.source]
        if disease_filter and disease_filter not in str(parent.get("disease", "")):
            continue
        entry = node_brief(kg, parent, with_sentence=False, with_attrs=("disease",))
        entry["provenance"] = edge_evidence(edge)
        subtypes.append(entry)
    payload["disease_subtypes"] = subtypes[:10]

    stages: List[Dict[str, Any]] = []
    for entry in diseases[:3]:
        for _e, stage in kg.neighbours(entry["id"], {EdgeType.HAS_PATHWAY_STAGE.value}):
            stages.append(
                node_brief(kg, stage, with_attrs=("order", "variant", "monitoring_items"))
            )
    payload["pathway_stages"] = stages[:6]
    payload["documents"] = documents_block(kg, [syndrome.id])
    payload["sibling_syndromes"] = _siblings(ctx, syndrome, diseases)

    coverage = Coverage.SUPPORTED if syndrome.sentence() else Coverage.PARTIAL
    caveats: List[str] = []
    if not syndrome.sentence():
        caveats.append(
            "该证候节点没有保留原文定义句，只能依据其所属疾病与文献推断，"
            "证据强度低于带定义句的证候。"
        )
    return ToolResult(
        tool=_SYNDROME_SPEC.name, ok=True, coverage=coverage, data=payload, caveats=caveats
    )


def _siblings(ctx: ToolContext, syndrome: Node, diseases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Competing syndromes under the same diseases -- the discriminative set."""
    kg = ctx.kg
    out: List[Dict[str, Any]] = []
    seen = {syndrome.id}
    for entry in diseases[:2]:
        for _edge, sibling in kg.neighbours(entry["id"], {EdgeType.HAS_SYNDROME.value}):
            if sibling.id in seen:
                continue
            seen.add(sibling.id)
            out.append(node_brief(kg, sibling))
    return out[:10]


# --------------------------------------------------------------------------- #
# Tool 4: retrieve_source_evidence
# --------------------------------------------------------------------------- #

_SOURCE_SPEC = ToolSpec(
    name="retrieve_source_evidence",
    description=(
        "溯源工具。给定实体或文献编号（doc_id），返回其来源文献的完整元数据"
        "（标题、文献类型、科室、版本、发布机构、生效日期、被取代关系），"
        "以及该文献中同时出现的其他实体。"
        "用于判断证据等级与时效性：例如 2017 年版诊疗方案与试行版临床路径的效力不同。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "实体名或实体 id，返回其全部来源文献。"},
            "doc_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选，直接指定文献编号，如 [\"doc_0000\"]。",
            },
            "include_cooccurring": {
                "type": "boolean",
                "description": "是否返回同一文献中共现的其他实体，默认 true。",
            },
        },
        "required": [],
    },
    domains=_ALL_DOMAINS,
)


@REGISTRY.register(_SOURCE_SPEC)
def retrieve_source_evidence(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    entity = str(args.get("entity") or "").strip()
    doc_ids = [str(d) for d in (args.get("doc_ids") or [])]
    include_cooccurring = bool(args.get("include_cooccurring", True))

    if not entity and not doc_ids:
        return ToolResult(
            tool=_SOURCE_SPEC.name,
            ok=False,
            coverage=Coverage.NOT_COVERED,
            error="必须提供 entity 或 doc_ids 之一。",
        )

    resolved: Optional[Node] = None
    if entity:
        node = kg.node(entity)
        matches = [node] if node else resolve_entity(
            kg, entity, ctx.visible_types(), retriever=ctx.retriever, domain=ctx.domain
        )
        if matches:
            resolved = matches[0]
            ctx.assert_visible(resolved.type)
            doc_ids = list(dict.fromkeys(doc_ids + list(resolved.source_docs)))

    documents: List[Dict[str, Any]] = []
    for doc_id in doc_ids[:8]:
        doc = kg.document(doc_id)
        if doc is None:
            continue
        record = doc_brief(doc)
        if include_cooccurring:
            others = [
                node_brief(kg, n, with_sentence=False)
                for n in kg.nodes_in_document(doc_id)
                if n.type in ctx.policy.visible_types() and n.type != NodeType.DOCUMENT_SOURCE.value
            ]
            record["co_occurring_entities"] = others[:25]
            record["n_co_occurring"] = len(others)
        documents.append(record)

    payload = {
        "entity": node_brief(kg, resolved, with_sentence=False) if resolved else None,
        "documents": documents,
    }
    if not documents:
        return ToolResult(
            tool=_SOURCE_SPEC.name,
            ok=True,
            coverage=Coverage.EMPTY,
            data=payload,
            caveats=["未找到对应文献记录。"],
        )
    return ToolResult(
        tool=_SOURCE_SPEC.name,
        ok=True,
        coverage=Coverage.SUPPORTED,
        data=payload,
        caveats=[
            "图谱只保存文献元数据与抽取出的证据句，不保存文献全文；"
            "无法据此回答原文中未被抽取的细节。"
        ],
    )
