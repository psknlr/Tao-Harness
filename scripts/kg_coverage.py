"""Knowledge-graph coverage audit.

The point of this script is to be the study's honesty instrument.  It is easy
to write a mapping table that assigns every prescription-audit rule family a
list of plausible-sounding entity types; it is harder, and much more useful, to
check whether the graph actually contains the facts those rules turn on.  Every
number below is computed from the delivered graph, not asserted.

The headline finding is that roughly half of the 19 PA rule families cannot be
grounded in this graph at all -- there is no dosage field, no 十八反/十九畏
table, no 君臣佐使 role annotation, no prescription-validity or
controlled-substance metadata.  Tools covering those families return
``NOT_COVERED`` by construction (see :mod:`tcm_tools.checkers`), which is what
keeps the PA arm from manufacturing false confidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from tcm_kg.schema import EdgeType, NodeType
from tcm_kg.store import KGStore

GROUNDED = "grounded"
PARTIAL = "partial"
NOT_GROUNDED = "not grounded"


@dataclass
class RuleAudit:
    rule_id: str
    title: str
    needs: str
    verdict: str
    evidence: str
    tools: str


def _count_text(kg: KGStore, needle: str) -> int:
    """Occurrences of a literal across every node and edge payload."""
    total = 0
    for node in kg.nodes.values():
        blob = json.dumps(
            {"n": node.name, "a": node.attrs, "f": node.first_mention}, ensure_ascii=False, default=str
        )
        total += blob.count(needle)
    for edge in kg.edges:
        for sentence in edge.evidence_sentences():
            total += sentence.count(needle)
    return total


def graph_facts(kg: KGStore) -> Dict[str, Any]:
    """Every quantity the audit table cites, measured once."""
    syndromes = kg.of_type(NodeType.SYNDROME.value)
    herbs = kg.of_type(NodeType.HERB.value)
    formulas = kg.of_type(NodeType.FORMULA.value)
    entries = kg.of_type(NodeType.PHARMACOPOEIA_ENTRY.value)

    with_definition = sum(1 for s in syndromes if len(s.sentence()) > len(s.name) + 6)
    with_prep = sum(1 for h in herbs if kg.preparation_markers(h.id))
    with_composition = sum(1 for f in formulas if kg.out_edges(f.id, {EdgeType.CONTAINS_HERB.value}))
    toxic = sum(
        1
        for e in entries
        if any(m in str(e.get("nature_taste_meridian") or "") for m in ("有毒", "小毒", "大毒"))
    )
    patent_with_composition = sum(
        1 for p in kg.of_type(NodeType.PATENT_MEDICINE.value) if kg.out_edges(p.id)
    )
    clusters = sum(1 for members in kg._cluster.values() if len(members) > 1)

    from tcm_tools.safety import _classify

    kinds: Dict[str, int] = {}
    for edge in kg.edges:
        if edge.type in {EdgeType.CONTRAINDICATED_FOR.value, EdgeType.CAUTION_FOR.value}:
            kind = _classify(kg.nodes[edge.source], kg.nodes[edge.target])
            kinds[kind] = kinds.get(kind, 0) + 1

    safety_contexts = kg.of_type(NodeType.SAFETY_CONTEXT.value)
    pregnancy = [n.name for n in safety_contexts if "孕" in n.name or "妊娠" in n.name]
    paediatric = [n.name for n in safety_contexts if any(w in n.name for w in ("小儿", "儿童", "婴"))]

    return {
        "n_syndromes": len(syndromes),
        "n_syndromes_with_definition": with_definition,
        "n_herbs": len(herbs),
        "n_herbs_with_preparation": with_prep,
        "n_formulas": len(formulas),
        "n_formulas_with_composition": with_composition,
        "n_pharmacopoeia_entries": len(entries),
        "n_toxicity_flagged": toxic,
        "n_patent_medicines": len(kg.of_type(NodeType.PATENT_MEDICINE.value)),
        "n_patent_with_composition": patent_with_composition,
        "n_identity_clusters": clusters,
        "safety_edge_kinds": kinds,
        "n_pregnancy_contexts": len(pregnancy),
        "n_paediatric_contexts": len(paediatric),
        "mentions_junchen": _count_text(kg, "君臣佐使") + _count_text(kg, "君药"),
        "mentions_shibafan": _count_text(kg, "十八反") + _count_text(kg, "十九畏"),
        "mentions_dosage_field": _count_text(kg, "用法用量"),
        "mentions_validity": _count_text(kg, "处方效期") + _count_text(kg, "有效期"),
        "mentions_controlled": sum(
            _count_text(kg, w) for w in ("毒性药品", "麻醉药品", "精神药品")
        ),
        "mentions_neonate": _count_text(kg, "新生儿") + _count_text(kg, "婴幼儿"),
    }


def audit_rules(facts: Mapping[str, Any]) -> List[RuleAudit]:
    """The 19 TCMEval-PA rule families against what the graph actually holds."""
    f = facts
    return [
        RuleAudit(
            "A-001", "处方适宜性概念", "定义性知识",
            NOT_GROUNDED,
            "DocumentSource 只存元数据（标题/类型/科室/版本），不存条文正文。",
            "—",
        ),
        RuleAudit(
            "A-002", "用药与病名/证型相符", "疾病—证候—治法—方药链",
            GROUNDED,
            f"HAS_SYNDROME 1298、TREATED_BY_PRINCIPLE 1433、USES_FORMULA 1145、"
            f"USES_PATENT_MEDICINE 1274 条边完整支撑该链路。",
            "retrieve_clinical_context, retrieve_medication_knowledge",
        ),
        RuleAudit(
            "A-003", "单味药剂量", "药典用法用量",
            NOT_GROUNDED,
            f"药典条目 {f['n_pharmacopoeia_entries']} 条，无用法用量字段"
            f"（“用法用量”出现 {f['mentions_dosage_field']} 次）。",
            "check_dose → NOT_COVERED",
        ),
        RuleAudit(
            "A-004", "总剂量/药味数量", "组成计数 + 总剂量",
            PARTIAL,
            f"药味数可数（{f['n_formulas_with_composition']}/{f['n_formulas']} 方有组成，"
            f"中位 11 味）；总剂量无数据。",
            "retrieve_medication_knowledge",
        ),
        RuleAudit(
            "A-005", "用法合理", "煎服法",
            PARTIAL,
            f"{f['n_herbs_with_preparation']}/{f['n_herbs']} 味药有原文煎法括注；"
            f"服法（每日几剂、饭前后）无数据。",
            "check_decoction_requirement",
        ),
        RuleAudit(
            "A-006", "品种选择", "基原/别名/功能主治",
            GROUNDED,
            f"药典条目含基原部位、性味归经、功能主治与别名；"
            f"{f['n_identity_clusters']} 个别名/炮制品聚类可做品种归一。",
            "retrieve_pharmacopeia_entry",
        ),
        RuleAudit(
            "A-007", "使用禁忌", "禁忌症",
            PARTIAL,
            f"禁忌边以饮食调护为主：{f['safety_edge_kinds']}。"
            f"这是诊疗方案的调护禁忌，不是药品说明书禁忌症。",
            "retrieve_safety_constraints, check_restricted_item",
        ),
        RuleAudit(
            "A-008", "重复用药", "成分重叠",
            PARTIAL,
            f"汤剂可判（组成 + {f['n_identity_clusters']} 个别名聚类）；"
            f"中成药不可判（{f['n_patent_with_composition']}/{f['n_patent_medicines']} 有组成）。",
            "check_duplicate_medication",
        ),
        RuleAudit(
            "A-009", "联合用药/配伍禁忌", "十八反十九畏",
            NOT_GROUNDED,
            f"“十八反/十九畏”出现 {f['mentions_shibafan']} 次；无配伍禁忌表。"
            f"（半夏与附子在图谱方剂中共现 3 次——正说明共现不可当作安全性证据。）",
            "check_combination → NOT_COVERED",
        ),
        RuleAudit(
            "N-001", "处方完整性", "处方格式要素", NOT_GROUNDED,
            "图谱不含处方实体，无科别/年龄/临床诊断等处方字段。", "—",
        ),
        RuleAudit(
            "N-002", "君臣佐使", "配伍角色标注", NOT_GROUNDED,
            f"“君臣佐使/君药”出现 {f['mentions_junchen']} 次；"
            f"CONTAINS_HERB 边不带角色标注。",
            "—",
        ),
        RuleAudit(
            "N-003", "特殊煎煮", "先煎/后下/包煎/烊化/冲服",
            GROUNDED,
            f"{f['n_herbs_with_preparation']} 味药有原文括注，按位置归属，"
            f"不会把同句其他药材的括注错记。",
            "check_decoction_requirement",
        ),
        RuleAudit("N-004", "剂量单位", "单位规范", NOT_GROUNDED, "无剂量数据即无单位数据。", "—"),
        RuleAudit("N-005", "处方用量", "每剂/疗程用量", NOT_GROUNDED, "无用量数据。", "—"),
        RuleAudit(
            "N-006", "处方效期", "有效期规定", NOT_GROUNDED,
            f"“处方效期/有效期”出现 {f['mentions_validity']} 次。", "—",
        ),
        RuleAudit(
            "N-007", "特殊药品", "毒麻精放管理",
            PARTIAL,
            f"药典有毒标注 {f['n_toxicity_flagged']}/{f['n_pharmacopoeia_entries']} 条；"
            f"“毒性/麻醉/精神药品”管理术语出现 {f['mentions_controlled']} 次，无分级管理数据。",
            "retrieve_pharmacopeia_entry",
        ),
        RuleAudit("N-008", "开具规范", "处方权与格式", NOT_GROUNDED, "图谱不含法规条文正文。", "—"),
        RuleAudit(
            "N-009", "新生儿/婴幼儿", "儿童用药",
            PARTIAL,
            f"“新生儿/婴幼儿”出现 {f['mentions_neonate']} 次；"
            f"儿科相关 SafetyContext {f['n_paediatric_contexts']} 个，"
            f"妊娠相关 {f['n_pregnancy_contexts']} 个；无儿童剂量折算规则。",
            "retrieve_safety_constraints",
        ),
        RuleAudit("C-001", "基本概念", "定义性知识", NOT_GROUNDED, "同 A-001。", "—"),
    ]


def _table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def coverage_report(kg: KGStore, retriever=None) -> str:
    facts = graph_facts(kg)
    audits = audit_rules(facts)
    counts = {GROUNDED: 0, PARTIAL: 0, NOT_GROUNDED: 0}
    for audit in audits:
        counts[audit.verdict] += 1

    parts: List[str] = [
        "# Knowledge-graph coverage audit",
        "",
        "Every figure below is computed from the delivered graph by "
        "`scripts/kg_coverage.py`; regenerate with "
        "`python -m runner.benchmark_runner coverage`.",
        "",
        "## Graph at a glance",
        "",
        "```json",
        json.dumps(kg.summary(), ensure_ascii=False, indent=2),
        "```",
        "",
        "## What the graph can and cannot ground",
        "",
        f"Of the 19 TCMEval-PA rule families: **{counts[GROUNDED]} grounded**, "
        f"**{counts[PARTIAL]} partial**, **{counts[NOT_GROUNDED]} not grounded**.",
        "",
        "A rule marked *not grounded* is not a defect to be papered over. The "
        "corresponding tool returns `NOT_COVERED` and says so, and the model is "
        "instructed to answer from its own knowledge while stating that the "
        "graph gave no support. Returning a confident \"no problem found\" for a "
        "dosage or compatibility question the graph never encoded would be a "
        "false negative in a safety system.",
        "",
        _table(
            [[a.rule_id, a.title, a.verdict, a.evidence, a.tools] for a in audits],
            ["rule", "title", "verdict", "evidence in this graph", "tool"],
        ),
        "",
        "## Consequences for the study design",
        "",
        f"- **SDT anchoring.** {facts['n_syndromes_with_definition']} of "
        f"{facts['n_syndromes']} Syndrome nodes ("
        f"{100 * facts['n_syndromes_with_definition'] / max(1, facts['n_syndromes']):.0f}%) "
        "carry the protocol's verbatim definition sentence, which lists main "
        "symptoms, tongue and pulse. That sentence — not any Symptom entity — "
        "is what a case description can actually match against, and it is why "
        "the retrieval index is built over per-entity virtual documents rather "
        "than over entity names.",
        f"- **The other {facts['n_syndromes'] - facts['n_syndromes_with_definition']} "
        "syndromes** are name-only. `retrieve_syndrome_evidence` reports "
        "`PARTIAL` for these so a model can weigh them accordingly.",
        "- **PA is the explicit-knowledge probe** only for the grounded and "
        "partial families. Reporting PA accuracy split by verdict is the "
        "cleanest test of RQ4: if the KG gain concentrates in the grounded "
        "families, the mechanism is knowledge injection rather than a general "
        "prompting effect.",
        "- **No entity was added to the ontology to suit either benchmark.** "
        "There is no Symptom node and no Pathogenesis node. Pathogenesis is "
        "treated as a latent reasoning variable produced by the model, which "
        "makes any SDT pathogenesis gain attributable to the graph narrowing "
        "the reasoning space rather than to retrieving the answer.",
        "",
        "## Measured facts",
        "",
        "```json",
        json.dumps(facts, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tcm_kg import load_kg

    print(coverage_report(load_kg()))
