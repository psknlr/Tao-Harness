"""Tool 8: ``verify_tcm_decision`` -- a rule engine, not a second model.

The verifier never calls an LLM.  It takes a decision the model has already
made and tests it against the graph, so the verification signal is identical
for every model under test and cannot itself become a source of capability
difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from tcm_kg.normalize import canonical_syndrome, char_ngrams, syndrome_atoms
from tcm_kg.schema import Domain, EdgeType, NodeType

from ._common import edge_evidence, node_brief, resolve_entity
from .base import REGISTRY, Coverage, ToolContext, ToolPhase, ToolResult, ToolSpec


class Verdict(str, Enum):
    #: the graph positively attests the claimed link
    SUPPORTED = "supported"
    #: the graph attests part of it (entity exists, link does not)
    PARTIAL = "partially_supported"
    #: the graph attests a link that conflicts with the claim
    CONTRADICTED = "contradicted"
    #: the entity is absent from the graph -- no opinion either way
    NOT_IN_GRAPH = "not_in_graph"
    #: the graph does not encode this class of claim at all
    NOT_COVERED = "not_covered"


_VERIFY_SPEC = ToolSpec(
    name="verify_tcm_decision",
    description=(
        "确定性校验工具（不调用大模型，纯图谱规则判定）。"
        "输入你已经得出的结论：证候，以及可选的疾病/亚型/治法，"
        "同时可传入病例的临床特征列表。"
        "返回逐条校验结果：该证候是否在图谱中存在；是否隶属于所锚定的疾病或亚型；"
        "所述治法是否为该证候的图谱治法；证候原文定义句与病例特征的重合特征词。"
        "用于在给出最终答案前做一次一致性自检；校验不通过不等于结论错误，但需要重新审视。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "syndrome": {"type": "string", "description": "你判定的证候名。"},
            "disease": {"type": "string", "description": "可选，你锚定的疾病或疾病亚型名。"},
            "treatment_principle": {
                "type": "string",
                "description": "可选，你给出的治法。仅作反向一致性证据，不应用于反推证候。",
            },
            "clinical_features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选，从病例中提取的临床特征（症状/舌象/脉象）列表，用于计算与证候定义句的特征重合。",
            },
        },
        "required": ["syndrome"],
    },
    domains=(Domain.CLINICAL, Domain.SAFETY, Domain.PATHWAY, Domain.FULL),
    deterministic=True,
    verification=True,
    phase=ToolPhase.VERIFICATION,
)


@REGISTRY.register(_VERIFY_SPEC)
def verify_tcm_decision(ctx: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    kg = ctx.kg
    claimed = str(args.get("syndrome", "")).strip()
    disease_name = str(args.get("disease") or "").strip()
    principle = str(args.get("treatment_principle") or "").strip()
    features = [str(f) for f in (args.get("clinical_features") or [])]

    checks: List[Dict[str, Any]] = []
    node = kg.node(claimed)
    matches = (
        [node]
        if node and node.type == NodeType.SYNDROME.value
        else resolve_entity(kg, claimed, [NodeType.SYNDROME.value])
    )
    if not matches:
        # try atom-wise resolution for compound syndromes
        atoms = syndrome_atoms(claimed)
        for atom in atoms:
            matches.extend(resolve_entity(kg, atom, [NodeType.SYNDROME.value]))

    if not matches:
        checks.append(
            {
                "check": "syndrome_exists",
                "verdict": Verdict.NOT_IN_GRAPH.value,
                "detail": (
                    f"图谱中没有名为 {claimed!r} 的证候节点。图谱收录 "
                    f"{len(kg.of_type(NodeType.SYNDROME.value))} 个证候，"
                    "覆盖国家中医诊疗方案范围，未收录不代表判断错误。"
                ),
            }
        )
        return ToolResult(
            tool=_VERIFY_SPEC.name,
            ok=True,
            coverage=Coverage.PARTIAL,
            data={"syndrome": claimed, "overall": Verdict.NOT_IN_GRAPH.value, "checks": checks},
            caveats=["证候不在图谱中，本次校验无法提供支持或反对的证据。"],
        )

    syndrome = matches[0]
    checks.append(
        {
            "check": "syndrome_exists",
            "verdict": Verdict.SUPPORTED.value,
            "detail": f"匹配到图谱证候 {syndrome.name}。",
            "node": node_brief(kg, syndrome, max_sentence=400),
        }
    )

    disease_check = _check_disease_link(ctx, syndrome, disease_name)
    checks.append(disease_check)
    if principle:
        checks.append(_check_principle(ctx, syndrome, principle))
    if features:
        checks.append(
            _check_feature_overlap(ctx, syndrome, features, disease_check.get("_disease_id"))
        )

    verdicts = [c["verdict"] for c in checks]
    if Verdict.CONTRADICTED.value in verdicts:
        overall = Verdict.CONTRADICTED.value
    elif all(v == Verdict.SUPPORTED.value for v in verdicts):
        overall = Verdict.SUPPORTED.value
    else:
        overall = Verdict.PARTIAL.value

    return ToolResult(
        tool=_VERIFY_SPEC.name,
        ok=True,
        coverage=Coverage.SUPPORTED,
        data={
            "syndrome": claimed,
            "resolved": node_brief(kg, syndrome, with_sentence=False),
            "overall": overall,
            "checks": checks,
        },
        caveats=[
            "本工具只判断结论与图谱的一致性，不判断临床正确性。"
            "图谱未覆盖的证候或关系会返回 not_in_graph，不应据此否定结论。"
        ],
    )


def _check_disease_link(ctx: ToolContext, syndrome: Any, disease_name: str) -> Dict[str, Any]:
    kg = ctx.kg
    parents = [
        (edge, kg.nodes[edge.source])
        for edge in kg.in_edges(
            syndrome.id,
            {EdgeType.HAS_SYNDROME.value, EdgeType.SUBTYPE_HAS_SYNDROME.value},
        )
    ]
    if not disease_name:
        return {
            "check": "syndrome_disease_link",
            "verdict": Verdict.PARTIAL.value if parents else Verdict.NOT_IN_GRAPH.value,
            "detail": "未提供疾病锚点，仅列出该证候在图谱中所属的疾病。",
            "attested_diseases": [
                node_brief(kg, parent, with_sentence=False) for _e, parent in parents[:8]
            ],
        }

    matched = [
        (edge, parent)
        for edge, parent in parents
        if disease_name in parent.name or parent.name in disease_name
        or disease_name in str(parent.get("disease", ""))
    ]
    if matched:
        edge, parent = matched[0]
        return {
            "check": "syndrome_disease_link",
            "verdict": Verdict.SUPPORTED.value,
            "detail": f"图谱记载 {parent.name} 下存在证候 {syndrome.name}。",
            "provenance": edge_evidence(edge),
            "_disease_id": parent.id,
        }
    if parents:
        return {
            "check": "syndrome_disease_link",
            "verdict": Verdict.CONTRADICTED.value,
            "detail": (
                f"图谱中 {syndrome.name} 只隶属于 "
                f"{[p.name for _e, p in parents[:5]]}，不包含 {disease_name!r}。"
                "疾病锚定与证候可能不匹配，请复核。"
            ),
            "attested_diseases": [
                node_brief(kg, parent, with_sentence=False) for _e, parent in parents[:8]
            ],
        }
    return {
        "check": "syndrome_disease_link",
        "verdict": Verdict.NOT_IN_GRAPH.value,
        "detail": f"{syndrome.name} 在图谱中没有关联到任何疾病。",
    }


def _check_principle(ctx: ToolContext, syndrome: Any, principle: str) -> Dict[str, Any]:
    kg = ctx.kg
    ctx.assert_visible(NodeType.TREATMENT_PRINCIPLE.value, verification=True)
    attested = [
        (edge, target)
        for edge, target in kg.neighbours(syndrome.id, {EdgeType.TREATED_BY_PRINCIPLE.value})
    ]
    claimed_tokens = set(char_ngrams(principle, (2,)))
    best: Optional[Any] = None
    best_overlap = 0.0
    for edge, target in attested:
        tokens = set(char_ngrams(target.name, (2,)))
        if not tokens:
            continue
        overlap = len(tokens & claimed_tokens) / len(tokens | claimed_tokens)
        if overlap > best_overlap:
            best_overlap, best = overlap, (edge, target)
    if best and best_overlap >= 0.5:
        edge, target = best
        return {
            "check": "treatment_principle_consistency",
            "verdict": Verdict.SUPPORTED.value,
            "detail": f"与图谱治法 {target.name!r} 高度一致（重合度 {best_overlap:.2f}）。",
            "provenance": edge_evidence(edge),
        }
    if attested:
        return {
            "check": "treatment_principle_consistency",
            "verdict": Verdict.PARTIAL.value,
            "detail": (
                f"图谱记载该证候治法为 {[t.name for _e, t in attested[:4]]}，"
                f"与所述 {principle!r} 重合度较低（{best_overlap:.2f}）。"
            ),
        }
    return {
        "check": "treatment_principle_consistency",
        "verdict": Verdict.NOT_IN_GRAPH.value,
        "detail": "图谱未记载该证候的治法。",
    }


def _check_feature_overlap(
    ctx: ToolContext,
    syndrome: Any,
    features: Sequence[str],
    disease_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Lexical overlap between the case features and the syndrome's presentation.

    Deterministic and interpretable on purpose: it reports *which* feature
    strings appear in the protocol's sentence, so a reader of the trace can see
    exactly why a candidate was reinforced.

    The sentence is taken **in the anchored disease's context** where one was
    established. Comparing a knee case against the same syndrome's presentation
    in a cardiovascular protocol would report near-zero overlap and mark a
    correct answer unsupported -- the verifier would be penalising the model
    for the graph's indexing rather than for its reasoning.
    """
    presentation = ctx.kg.syndrome_presentation(syndrome.id, disease_id)
    sentence = presentation["sentence"]
    if not sentence:
        return {
            "check": "feature_overlap",
            "verdict": Verdict.NOT_COVERED.value,
            "detail": "该证候没有保留原文表现描述，无法计算特征重合。",
        }
    matched = [f for f in features if f and f in sentence]
    definition_tokens = set(char_ngrams(sentence, (2,)))
    feature_tokens = set(char_ngrams(" ".join(features), (2,)))
    jaccard = (
        len(definition_tokens & feature_tokens) / len(feature_tokens)
        if feature_tokens
        else 0.0
    )
    verdict = (
        Verdict.SUPPORTED.value
        if matched or jaccard >= 0.25
        else Verdict.PARTIAL.value
    )
    out = {
        "check": "feature_overlap",
        "verdict": verdict,
        "detail": (
            f"病例特征与证候表现描述的二元词重合率 {jaccard:.2f}；"
            f"逐字命中 {matched if matched else '无'}。"
        ),
        "presentation_sentence": sentence[:400],
        "presentation_scope": presentation["scope"],
        "matched_features": matched,
        "coverage_ratio": round(jaccard, 4),
    }
    if presentation["scope"] == "global_first_mention":
        # a low overlap here may mean the sentence is from another disease,
        # not that the answer is wrong
        out["caveat"] = (
            "表现描述来自该证候的全局首次出现文献，可能属于其他疾病语境；"
            "重合率偏低不足以否定该证候。"
        )
        if verdict == Verdict.PARTIAL.value:
            out["verdict"] = Verdict.NOT_COVERED.value
    return out
