"""Per-benchmark adapters.

A task knows four things: which access domain it runs in, how to turn a dataset
item into a user message, how to build the *deterministic* retrieval block used
by the static KG-RAG condition, and how to hand its own answer to the
verification rule engine.  Everything else -- the loop, the budget, the parsing
-- is shared, which is what keeps SDT and PA comparable as two probes of one
framework rather than two different systems.
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tcm_kg.normalize import canonical_syndrome
from tcm_kg.schema import Domain, EdgeType, NodeType
from tcm_kg.store import KGStore

from .parsing import coerce_list, coerce_str
from .prompts import load_prompt

#: Demographic and administrative fragments that add no retrieval signal and
#: measurably dilute character-n-gram BM25.  Stripped identically for every
#: model, in the static condition and in the agent's suggested query alike.
_NOISE_PATTERNS = (
    r"[男女]性?[,，、\s]",
    r"\d+\s*(?:岁|周岁|月龄|天|岁半)",
    r"(?:患者|病人|初诊|复诊|就诊|门诊|住院)[,，:：]?",
    r"\d{4}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?",
    r"(?:主诉|现病史|既往史|个人史|婚育史|家族史)[:：]",
)
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS))


def clinical_query(text: str, *, max_chars: int = 400) -> str:
    """Deterministic retrieval query derived from raw case text."""
    cleaned = _NOISE_RE.sub(" ", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


@dataclass
class ContextBudget:
    """Caps on injected KG context, identical for every model."""

    max_chars: int = 6000
    max_entities_per_block: int = 8

    def fingerprint(self) -> str:
        return f"{self.max_chars}/{self.max_entities_per_block}"


class Task(abc.ABC):
    """Benchmark-specific glue."""

    name: str
    domain: Domain

    def __init__(self, kg: KGStore, retriever, budget: Optional[ContextBudget] = None):
        self.kg = kg
        self.retriever = retriever
        self.context_budget = budget or ContextBudget()

    # ------------------------------------------------------------- prompting
    @abc.abstractmethod
    def system_prompt(self, condition: str) -> str: ...

    @abc.abstractmethod
    def user_message(self, item: Mapping[str, Any]) -> str: ...

    @abc.abstractmethod
    def answer_fields(self) -> Sequence[str]: ...

    # -------------------------------------------------------------- retrieval
    @abc.abstractmethod
    def static_context(self, item: Mapping[str, Any]) -> Dict[str, Any]: ...

    def render_context(self, context: Mapping[str, Any]) -> str:
        text = json.dumps(context, ensure_ascii=False, indent=None, sort_keys=True)
        limit = self.context_budget.max_chars
        if len(text) > limit:
            text = text[: limit - 24] + '..."<context truncated>"}'
        return "【知识图谱检索证据】\n" + text

    # ----------------------------------------------------------- verification
    @abc.abstractmethod
    def verify_arguments(
        self, result: Mapping[str, Any], item: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]: ...

    def normalise_result(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(result)


# --------------------------------------------------------------------------- #
# SDT
# --------------------------------------------------------------------------- #


class SDTTask(Task):
    """TCMEval-SDT: syndrome differentiation from a clinical record."""

    name = "sdt"
    domain = Domain.CLINICAL

    def system_prompt(self, condition: str) -> str:
        if condition == "M0":
            return load_prompt("sdt_m0_base")
        parts = [load_prompt("sdt_structured")]
        if condition in {"M2", "M3", "M4"}:
            parts.append(load_prompt("sdt_kg_note"))
        if condition in {"M3", "M4"}:
            parts.append(load_prompt("sdt_agent"))
        return "\n".join(parts)

    def user_message(self, item: Mapping[str, Any]) -> str:
        return f"【病例】\n{coerce_str(item.get('clinical_data'))}"

    def answer_fields(self) -> Sequence[str]:
        return ("clinical_information", "pathogenesis", "syndrome", "explanation")

    def static_context(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        query = clinical_query(item.get("clinical_data"))
        limit = self.context_budget.max_entities_per_block

        diseases = self.retriever.search(
            query,
            domain=self.domain,
            node_types=[NodeType.DISEASE.value, NodeType.DISEASE_SUBTYPE.value],
            top_k=3,
        )
        anchors = [h.node_id for h in diseases]
        syndromes = self.retriever.search(
            query,
            domain=self.domain,
            node_types=[NodeType.SYNDROME.value],
            anchors=anchors,
            top_k=limit,
        )

        subgraph: List[Dict[str, Any]] = []
        for hit in diseases:
            node = self.kg.node(hit.node_id)
            if node is None:
                continue
            children = [
                {"name": syn.name, "definition": syn.sentence()[:200]}
                for _e, syn in self.kg.neighbours(
                    node.id,
                    {EdgeType.HAS_SYNDROME.value, EdgeType.SUBTYPE_HAS_SYNDROME.value},
                )
            ][:limit]
            subgraph.append(
                {
                    "disease": node.name,
                    "score": round(hit.score, 3),
                    "candidate_syndromes": children,
                }
            )

        documents: List[Dict[str, Any]] = []
        seen_docs: set = set()
        for hit in list(diseases) + list(syndromes):
            for doc_id in hit.source_docs[:2]:
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                doc = self.kg.document(doc_id)
                if doc is not None:
                    documents.append(
                        {
                            "doc_id": doc_id,
                            "title": doc.attrs.get("title"),
                            "doc_type": doc.attrs.get("doc_type"),
                            "department": doc.attrs.get("department"),
                            "version": doc.attrs.get("version_label"),
                        }
                    )
        return {
            "retrieval_query": query,
            "disease_anchors": subgraph,
            "syndrome_candidates": [
                {
                    "name": h.name,
                    "score": round(h.score, 3),
                    "definition": h.matched_text[:220],
                }
                for h in syndromes
            ],
            "documents": documents[:6],
            "note": (
                "图谱不含症状与病机实体；证候的 definition 字段是诊疗方案原文定义句。"
            ),
        }

    def verify_arguments(
        self, result: Mapping[str, Any], item: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        syndrome = canonical_syndrome(coerce_str(result.get("syndrome")))
        if not syndrome:
            return None
        features = coerce_list(result.get("clinical_information"))
        return {
            "syndrome": syndrome,
            "clinical_features": features[:20],
        }


# --------------------------------------------------------------------------- #
# PA
# --------------------------------------------------------------------------- #

_OPTION_RE = re.compile(r"^\s*([A-Z])\s*[.、:：)）]\s*(.+)$")


class PATask(Task):
    """TCMEval-PA: multiple-choice prescription audit."""

    name = "pa"
    domain = Domain.SAFETY

    def system_prompt(self, condition: str) -> str:
        if condition == "M0":
            return load_prompt("pa_m0_base")
        parts = [load_prompt("pa_structured")]
        if condition in {"M2", "M3", "M4"}:
            parts.append(load_prompt("pa_kg_note"))
        if condition in {"M3", "M4"}:
            parts.append(load_prompt("pa_agent"))
        return "\n".join(parts)

    def user_message(self, item: Mapping[str, Any]) -> str:
        lines = [f"【题目】\n{coerce_str(item.get('question'))}"]
        options = item.get("options")
        if options:
            lines.append("【选项】")
            if isinstance(options, Mapping):
                for key in sorted(options):
                    lines.append(f"{key}. {coerce_str(options[key])}")
            else:
                for option in options:
                    lines.append(coerce_str(option))
        kind = item.get("question_type") or item.get("type")
        if kind:
            lines.append(f"【题型】{coerce_str(kind)}")
        return "\n".join(lines)

    def answer_fields(self) -> Sequence[str]:
        return ("rule_category", "option_analysis", "answer", "reasoning")

    def static_context(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        query = self._query(item)
        limit = self.context_budget.max_entities_per_block

        drugs = self.retriever.search(
            query,
            domain=self.domain,
            node_types=[
                NodeType.HERB.value,
                NodeType.FORMULA.value,
                NodeType.PATENT_MEDICINE.value,
            ],
            top_k=limit,
        )
        pharma = self.retriever.search(
            query,
            domain=self.domain,
            node_types=[NodeType.PHARMACOPOEIA_ENTRY.value],
            top_k=limit,
        )
        safety = self.retriever.search(
            query,
            domain=self.domain,
            node_types=[NodeType.SAFETY_CONTEXT.value, NodeType.RESTRICTED_ITEM.value],
            top_k=limit,
        )

        pharma_entries = []
        for hit in pharma:
            node = self.kg.node(hit.node_id)
            if node is None:
                continue
            pharma_entries.append(
                {
                    "name": node.name,
                    "nature_taste_meridian": node.attrs.get("nature_taste_meridian"),
                    "functions": str(node.attrs.get("pharmacopoeial_functions") or "")[:200],
                    "part_used": node.attrs.get("part_used"),
                    "aliases": list(node.attrs.get("common_aliases") or [])[:6],
                    "dosage_available": False,
                }
            )

        drug_entries = []
        for hit in drugs:
            node = self.kg.node(hit.node_id)
            if node is None:
                continue
            entry: Dict[str, Any] = {"name": node.name, "type": node.type}
            markers = (
                self.kg.preparation_markers(node.id)
                if node.type == NodeType.HERB.value
                else {}
            )
            if markers:
                entry["preparation_requirements"] = sorted(markers)
            if node.type == NodeType.FORMULA.value:
                entry["composition"] = [
                    herb.name
                    for _e, herb in self.kg.neighbours(node.id, {EdgeType.CONTAINS_HERB.value})
                ][:30]
            drug_entries.append(entry)

        safety_entries = []
        for hit in safety:
            node = self.kg.node(hit.node_id)
            if node is None:
                continue
            restrictions = []
            for edge in self.kg.in_edges(
                node.id, {EdgeType.CONTRAINDICATED_FOR.value, EdgeType.CAUTION_FOR.value}
            ) + self.kg.out_edges(
                node.id, {EdgeType.CONTRAINDICATED_FOR.value, EdgeType.CAUTION_FOR.value}
            ):
                other = (
                    self.kg.nodes[edge.source]
                    if edge.target == node.id
                    else self.kg.nodes[edge.target]
                )
                restrictions.append(
                    {
                        "severity": (
                            "contraindicated"
                            if edge.type == EdgeType.CONTRAINDICATED_FOR.value
                            else "caution"
                        ),
                        "counterpart": other.name,
                        "quote": (edge.evidence_sentences() or [""])[0][:160],
                    }
                )
            safety_entries.append({"name": node.name, "type": node.type, "restrictions": restrictions[:6]})

        return {
            "retrieval_query": query,
            "drugs": drug_entries,
            "pharmacopoeia": pharma_entries,
            "safety": safety_entries,
            "not_covered_by_graph": [
                "用法用量与剂量上限",
                "十八反十九畏配伍禁忌表",
                "妊娠禁忌药分级表",
                "中西药相互作用表",
                "中成药处方组成",
            ],
        }

    @staticmethod
    def _query(item: Mapping[str, Any]) -> str:
        parts = [coerce_str(item.get("question"))]
        options = item.get("options")
        if isinstance(options, Mapping):
            parts.extend(coerce_str(v) for v in options.values())
        elif isinstance(options, (list, tuple)):
            parts.extend(coerce_str(o) for o in options)
        return re.sub(r"\s+", " ", " ".join(parts))[:400]

    def verify_arguments(
        self, result: Mapping[str, Any], item: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        # PA answers are option letters, which the syndrome verifier cannot
        # check.  Verification for PA is done by re-running the deterministic
        # checkers named by the rule category (see runtime._verify_pa).
        return None

    def normalise_result(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(result)
        out["answer"] = normalise_options(result.get("answer"))
        return out


_LETTER_RE = re.compile(r"[A-Za-z]")


def normalise_options(value: Any) -> List[str]:
    """Coerce whatever a model emitted into a sorted list of option letters."""
    letters: List[str] = []
    for chunk in coerce_list(value):
        for match in _LETTER_RE.findall(chunk):
            upper = match.upper()
            if upper not in letters:
                letters.append(upper)
    return sorted(letters)


TASKS = {"sdt": SDTTask, "pa": PATask}


def build_task(name: str, kg: KGStore, retriever, budget: Optional[ContextBudget] = None) -> Task:
    key = name.lower()
    if key not in TASKS:
        raise ValueError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[key](kg, retriever, budget)
