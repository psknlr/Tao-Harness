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

from tcm_kg.normalize import DECOCTION_MARKERS, canonical_syndrome
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


#: Clause separators in Chinese clinical prose.
_CLAUSE_RE = re.compile(r"[，,。；;、\n]+")
#: Leading narrative that carries no findings.
_NARRATIVE_PREFIX_RE = re.compile(
    r"^(?:主诉及病史|主诉|现病史|诊查|既往史|初诊|复诊|患者)[:：]?"
)


def case_clauses(text: Any, *, max_clauses: int = 40, min_len: int = 2) -> List[str]:
    """Split raw case text into candidate findings, deterministically.

    Used by the verifier so that its evidence comes from the case rather than
    from the model's own summary. Crude on purpose: any cleverness here would
    become a second, unvalidated extraction model sitting inside the
    verification path.
    """
    out: List[str] = []
    for clause in _CLAUSE_RE.split(str(text or "")):
        clause = _NARRATIVE_PREFIX_RE.sub("", clause.strip()).strip()
        clause = _NOISE_RE.sub(" ", clause).strip()
        if len(clause) >= min_len and clause not in out:
            out.append(clause)
        if len(out) >= max_clauses:
            break
    return out


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

    def normalise_result(
        self, result: Mapping[str, Any], item: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        return dict(result)

    @staticmethod
    def clamp_letters(value: Any, options: Mapping[str, Any]) -> List[str]:
        """Keep only letters that name a real option for this item.

        A model that answers ``K`` on a ten-option question has not made a
        scoreable choice; silently passing it through would let a formatting
        slip read as a wrong answer, and letting it into a submission file
        would be scored as a wrong pick by the official evaluator, which
        dilutes credit for the picks that were right.
        """
        letters = normalise_options(value)
        if not options:
            return letters
        valid = {str(k).upper() for k in options}
        return [letter for letter in letters if letter in valid]


# --------------------------------------------------------------------------- #
# SDT
# --------------------------------------------------------------------------- #


class SDTTask(Task):
    """TCMEval-SDT: a four-task benchmark, scored 0.2/0.3/0.4/0.1.

    Tasks 2 and 3 are ten-option multiple choice with possibly several correct
    answers, which changes what the knowledge graph is for: the options arrive
    as *named* pathogeneses and syndromes, so the agent can look each candidate
    up directly instead of having to guess a name from the case text. That is a
    far better fit for this graph than free-form naming, and it is why
    ``static_context`` resolves the option list against the graph rather than
    only running the case text through retrieval.
    """

    name = "sdt"
    domain = Domain.CLINICAL

    def system_prompt(self, condition: str) -> str:
        if condition == "M0":
            return load_prompt("sdt_m0_base")
        parts = [load_prompt("sdt_structured")]
        # M2C sees the same static KG block as M2, so it needs the same note
        if condition in {"M2", "M2C", "M3", "M4", "M3C"}:
            parts.append(load_prompt("sdt_kg_note"))
        if condition in {"M3", "M4", "M3C"}:
            parts.append(load_prompt("sdt_agent"))
        return "\n".join(parts)

    def user_message(self, item: Mapping[str, Any]) -> str:
        lines = [f"【病例】\n{coerce_str(item.get('clinical_data'))}", ""]
        lines.append("【任务二 病机选项】")
        lines.extend(self._render_options(item.get("pathogenesis_options")))
        lines.append("")
        lines.append("【任务三 证候选项】")
        lines.extend(self._render_options(item.get("syndrome_options")))
        return "\n".join(lines)

    @staticmethod
    def _render_options(options: Any) -> List[str]:
        if not isinstance(options, Mapping) or not options:
            return ["（本题未提供选项）"]
        return [f"{letter}. {text}" for letter, text in sorted(options.items())]

    def answer_fields(self) -> Sequence[str]:
        return ("clinical_information", "pathogenesis_answer", "syndrome_answer", "explanation")

    # ------------------------------------------------------------- retrieval
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

        return {
            "retrieval_query": query,
            "disease_anchors": subgraph,
            "syndrome_candidates_from_case": [
                {"name": h.name, "score": round(h.score, 3), "definition": h.matched_text[:220]}
                for h in syndromes
            ],
            "syndrome_option_lookup": self._lookup_options(item.get("syndrome_options")),
            "note": (
                "图谱不含症状与病机实体；证候的 definition 是诊疗方案原文定义句。"
                "option_lookup 中 found=false 只表示该名称不在本图谱收录范围内，"
                "不是排除该选项的理由。"
                "任务二（病机）的选项刻意不做图谱检索：病机不是图谱实体，"
                "病机判断必须由临床信息与证候证据自行推理得出。"
            ),
        }

    def _lookup_options(self, options: Any) -> List[Dict[str, Any]]:
        """Resolve each named **syndrome** option against the graph.

        Reported for every option, found or not, so the model sees a uniform
        table rather than a list biased toward whatever the graph happens to
        cover -- absence of an option from this graph is not evidence against
        it, and a partial table would imply otherwise.

        Deliberately syndrome-only. An earlier version also ran the *task-2
        pathogenesis* options through this method with a flag that was supposed
        to disable syndrome matching but did not: the first lookup always hit
        Syndrome nodes regardless. That is option-conditioned retrieval
        leakage. Pathogenesis names and syndrome names share surface
        vocabulary, so "which options does the graph recognise" is a signal
        correlated with the answer, obtained without any reasoning. It also
        destroyed the study's cleanest claim -- that a task-2 gain must come
        from constrained reasoning, because the graph holds no pathogenesis
        entity. Task 2 is now never queried against the graph at all.
        """
        if not isinstance(options, Mapping):
            return []
        out: List[Dict[str, Any]] = []
        for letter, text in sorted(options.items()):
            name = str(text).strip()
            record: Dict[str, Any] = {"option": letter, "name": name, "found": False}
            matches = self.kg.find_by_name(name, [NodeType.SYNDROME.value])
            if not matches:
                matches = self.kg.find_by_name(canonical_syndrome(name), [NodeType.SYNDROME.value])
            if not matches:
                # the option pool is drawn from case annotations, so many names
                # are phrase-like ("阴亏之体"); fall back to a similarity-gated
                # lexical hit rather than an unconditional top-1
                for hit in self.retriever.search(
                    name, domain=self.domain, node_types=[NodeType.SYNDROME.value], top_k=3
                ):
                    node = self.kg.node(hit.node_id)
                    if node is not None and _surface_overlap(name, node.name) >= 0.5:
                        matches = [node]
                        break
            if matches:
                node = matches[0]
                record.update(
                    {
                        "found": True,
                        "graph_name": node.name,
                        "definition": node.sentence()[:220],
                        "diseases": [
                            self.kg.nodes[e.source].name
                            for e in self.kg.in_edges(
                                node.id,
                                {
                                    EdgeType.HAS_SYNDROME.value,
                                    EdgeType.SUBTYPE_HAS_SYNDROME.value,
                                },
                            )
                        ][:4],
                    }
                )
            out.append(record)
        return out

    # ---------------------------------------------------------- verification
    def verify_arguments(
        self, result: Mapping[str, Any], item: Mapping[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Verify the *text* behind every chosen syndrome letter.

        Returns one argument set per selected option. 27 of the 50 test cases
        have more than one correct syndrome, and models select more than one
        accordingly; verifying only the first -- as an earlier version did --
        left most of a multi-select answer unchecked and made the M4 arm's
        verification signal a function of answer order.
        """
        options = item.get("syndrome_options") or {}
        letters = self.clamp_letters(result.get("syndrome_answer"), options)
        # Features come from the **raw case**, not from the model's own
        # extraction. Verifying a syndrome against the findings the model
        # itself chose to report is circular: a model that decided on 肝郁气滞
        # tends to have listed 胸胁胀痛 and 脉弦, so the verifier confirms the
        # claim using evidence the claimant curated. Deriving them from the
        # case text keeps the verifier independent of the claim-maker, which is
        # the whole point of having one.
        features = case_clauses(item.get("clinical_data"))
        return [
            {
                "syndrome": canonical_syndrome(str(options[letter]).strip()),
                "clinical_features": features,
                "_option": letter,
            }
            for letter in letters
            if letter in options and str(options[letter]).strip()
        ] or None

    def normalise_result(
        self, result: Mapping[str, Any], item: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        out = dict(result)
        pathogenesis_options = (item or {}).get("pathogenesis_options") or {}
        syndrome_options = (item or {}).get("syndrome_options") or {}
        out["pathogenesis_answer"] = self.clamp_letters(
            result.get("pathogenesis_answer"), pathogenesis_options
        )
        out["syndrome_answer"] = self.clamp_letters(
            result.get("syndrome_answer"), syndrome_options
        )
        out["clinical_information"] = [
            part for part in coerce_list(result.get("clinical_information")) if part
        ]
        out["explanation"] = coerce_str(result.get("explanation"))
        return out


def _surface_overlap(left: str, right: str) -> float:
    """Character-bigram Jaccard between two names."""
    from tcm_kg.normalize import char_ngrams, normalize_text

    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return 0.0
    if a == b or a in b or b in a:
        return 1.0
    x, y = set(char_ngrams(a, (2,))), set(char_ngrams(b, (2,)))
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


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
        if condition in {"M2", "M2C", "M3", "M4", "M3C"}:
            parts.append(load_prompt("pa_kg_note"))
        if condition in {"M3", "M4", "M3C"}:
            parts.append(load_prompt("pa_agent"))
        return "\n".join(parts)

    def user_message(self, item: Mapping[str, Any]) -> str:
        lines = [f"【题目】\n{coerce_str(item.get('question'))}", "【选项】"]
        options = item.get("options")
        if isinstance(options, Mapping):
            for key in sorted(options):
                lines.append(f"{key}. {coerce_str(options[key])}")
        elif options:
            lines.extend(coerce_str(option) for option in options)
        n_gold = len(item.get("answer_letters") or [])
        # The released set is 297 single- plus 31 multiple-choice and marks the
        # distinction only through the answer key, which is gold. Telling the
        # model how many options are correct would leak that key, so the prompt
        # says only that either is possible.
        lines.append("【题型】单选或多选，请自行判断")
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

    #: Rule family -> the deterministic checkers that can adjudicate it.
    #: Only families the graph can actually ground appear here; the rest are
    #: left to the coverage audit, which is the honest answer for a rule with
    #: no data behind it.
    RULE_CHECKERS: Mapping[str, Sequence[str]] = {
        "N-003": ("check_decoction_requirement",),
        "A-005": ("check_decoction_requirement",),
        "A-008": ("check_duplicate_medication",),
        "A-007": ("check_restricted_item",),
        "A-009": ("check_combination",),
        "A-003": ("check_dose",),
        "A-004": ("check_dose",),
        "N-007": ("check_restricted_item",),
        "N-009": ("check_restricted_item",),
    }

    #: Surface forms a model may use for a rule family, mapped to its code.
    #: This is how the verifier learns which checker to run **from the model's
    #: own stated category**, never from the benchmark's annotation.
    CATEGORY_ALIASES: Mapping[str, str] = {
        "特殊煎煮": "N-003", "煎煮": "N-003", "煎法": "N-003", "先煎": "N-003",
        "后下": "N-003", "包煎": "N-003", "烊化": "N-003", "冲服": "N-003",
        "用法": "A-005", "服法": "A-005", "用法用量": "A-005",
        "重复用药": "A-008", "重复": "A-008",
        "使用禁忌": "A-007", "禁忌": "A-007", "禁忌症": "A-007",
        "配伍禁忌": "A-009", "配伍": "A-009", "十八反": "A-009", "十九畏": "A-009",
        "联合用药": "A-009", "相互作用": "A-009",
        "剂量": "A-003", "单味药剂量": "A-003", "超量": "A-003", "用量": "A-003",
        "总剂量": "A-004", "药味数": "A-004",
        "特殊药品": "N-007", "毒性药品": "N-007", "有毒": "N-007",
        "新生儿": "N-009", "婴幼儿": "N-009", "儿童用药": "N-009",
    }

    def verify_arguments(
        self, result: Mapping[str, Any], item: Mapping[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Deterministic checks to re-run against the answer the model gave.

        PA answers are option letters, so there is nothing for the syndrome
        verifier to check. What *can* be re-checked is the drug knowledge the
        answer rests on: the entities named in the question and in the options
        the model selected, put back through the rule engine for the family
        this item belongs to.

        This is what the M4 arm is supposed to mean. Without it -- as in an
        earlier version, whose comment pointed at a ``_verify_pa`` that was
        never written -- M4 was only a coverage audit plus one more revision
        turn, and any M3→M4 gain would have measured extra thinking rather
        than verification.

        **Routing comes from the model, not from the answer key.** The
        benchmark annotates each item with a ``rule_id``, and an earlier
        version read it here to pick the checker. That is oracle routing: a
        model in M4 was silently told which safety rule the question was about
        -- dose, or contraindication, or incompatibility -- which is a large
        part of the work, and no other arm received it. Any M3→M4 gain would
        then have partly measured a label the deployment setting cannot
        supply. The category now comes from the model's own
        ``rule_category`` output, which the structured prompt already asks
        for; ``rule_id`` survives only in scoring, where it belongs.
        """
        rule = self._declared_rule(result)
        checkers = self.RULE_CHECKERS.get(rule) if rule else None
        if not checkers:
            return None
        entities = self._entities_in_play(result, item)
        if not entities:
            return None

        calls: List[Dict[str, Any]] = []
        for checker in checkers:
            if checker == "check_dose":
                calls.append({"_tool": checker, "items": [{"name": e} for e in entities]})
            elif checker == "check_restricted_item":
                calls.append(
                    {
                        "_tool": checker,
                        "items": entities,
                        "patient_context": self._patient_context(item),
                    }
                )
            elif checker == "check_combination" and len(entities) >= 2:
                calls.append({"_tool": checker, "items": entities})
            elif checker == "check_decoction_requirement":
                call: Dict[str, Any] = {"_tool": checker, "items": entities}
                # Pass what the model actually claimed, so the checker returns a
                # verdict on the answer rather than a bare list of attested
                # markers. Without it the check runs but adjudicates nothing.
                claimed = self._claimed_preparation(result, item)
                if claimed:
                    call["claimed_requirement"] = claimed
                calls.append(call)
            elif checker == "check_duplicate_medication":
                calls.append({"_tool": checker, "items": entities})
        return calls or None

    @staticmethod
    def _claimed_preparation(
        result: Mapping[str, Any], item: Mapping[str, Any]
    ) -> Optional[str]:
        """The preparation method the selected option asserts, if any."""
        options = item.get("options") or {}
        letters = PATask.clamp_letters(result.get("answer"), options)
        for letter in letters:
            text = str(options.get(letter) or "")
            for marker in DECOCTION_MARKERS:
                if marker in text:
                    return marker
        return None

    def _declared_rule(self, result: Mapping[str, Any]) -> Optional[str]:
        """Map the model's own stated rule category onto a family code.

        Accepts an explicit code (``"A-003"``) or a Chinese surface form
        (``"剂量"``). Returns ``None`` when the model said nothing usable, in
        which case verification falls back to the coverage audit -- a model
        that cannot categorise its own question does not get routed for free.
        """
        declared = coerce_str(result.get("rule_category"))
        if not declared:
            return None
        upper = declared.upper()
        match = re.search(r"\b([ANC]-\d{3})\b", upper)
        if match and match.group(1) in self.RULE_CHECKERS:
            return match.group(1)
        # longest alias first, so 单味药剂量 beats 剂量
        for alias in sorted(self.CATEGORY_ALIASES, key=len, reverse=True):
            if alias in declared:
                return self.CATEGORY_ALIASES[alias]
        return None

    def _entities_in_play(
        self, result: Mapping[str, Any], item: Mapping[str, Any]
    ) -> List[str]:
        """Drug-like names the answer actually depends on.

        Drawn from the options the model selected plus the question stem, and
        resolved against the graph so free text does not reach the checkers.
        Capped, because a checker fed twenty speculative names produces noise
        rather than verification.
        """
        options = item.get("options") or {}
        letters = self.clamp_letters(result.get("answer"), options)
        texts = [str(options[l]) for l in letters if l in options]
        texts.append(str(item.get("question") or ""))

        found: List[str] = []
        for node_type in (NodeType.HERB.value, NodeType.FORMULA.value, NodeType.PATENT_MEDICINE.value):
            for node in self.kg.of_type(node_type):
                name = node.base_name or node.name
                if len(name) < 2:
                    continue
                if any(name in text for text in texts) and name not in found:
                    found.append(name)
                if len(found) >= 6:
                    return found
        return found

    @staticmethod
    def _patient_context(item: Mapping[str, Any]) -> str:
        """Cohort mentioned in the stem, for the restriction checker."""
        question = str(item.get("question") or "")
        for cohort in ("孕妇", "妊娠", "哺乳", "儿童", "小儿", "婴幼儿", "新生儿", "老年"):
            if cohort in question:
                return cohort
        return ""

    def normalise_result(
        self, result: Mapping[str, Any], item: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        out = dict(result)
        out["answer"] = self.clamp_letters(result.get("answer"), (item or {}).get("options") or {})
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


class ClinicalPathwayTask(Task):
    """TCM-CP: execute a staged clinical pathway.

    Runs in the pathway domain, which withholds nothing -- executing a pathway
    means deciding on treatment, so the treatment sub-graph has to be
    reachable. That is safe here and *not* safe for SDT: nothing in TCM-CP is
    answerable by inverting a syndrome->formula mapping, whereas in SDT that
    inversion would hand over the answer.
    """

    name = "cp"
    domain = Domain.PATHWAY

    def system_prompt(self, condition: str) -> str:
        parts = [load_prompt("cp_structured")]
        if condition in {"M3", "M4", "M3C"}:
            parts.append(load_prompt("cp_agent"))
        return "\n".join(parts)

    def user_message(self, item: Mapping[str, Any]) -> str:
        lines = [f"【患者情况】\n{coerce_str(item.get('vignette'))}", ""]
        if item.get("disease"):
            lines.append(f"【病种】{coerce_str(item.get('disease'))}")
        lines.append(f"【问题】{coerce_str(item.get('question'))}")
        lines.append("【选项】")
        options = item.get("options") or {}
        for key in sorted(options):
            lines.append(f"{key}. {coerce_str(options[key])}")
        return "\n".join(lines)

    def answer_fields(self) -> Sequence[str]:
        return ("decision_type", "reasoning", "answer")

    def static_context(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        """Fixed retrieval: the disease's stages, without naming the answer."""
        disease = coerce_str(item.get("disease"))
        limit = self.context_budget.max_entities_per_block
        matches = self.kg.find_by_name(disease, [NodeType.DISEASE.value])
        if not matches:
            return {"disease": disease, "stages": [], "note": "图谱中未找到该病种的临床路径。"}
        stages = [
            t for _e, t in self.kg.neighbours(matches[0].id, {EdgeType.HAS_PATHWAY_STAGE.value})
        ]
        stages.sort(key=lambda s: (str(s.get("variant") or ""), s.get("order") or 0))
        rendered = []
        for stage in stages[:limit]:
            rendered.append(
                {
                    "stage": stage.name,
                    "order": stage.get("order"),
                    "variant": stage.get("variant"),
                    "day_actions": [str(a) for a in (stage.get("day_actions") or [])][:10],
                    "nursing_items": [str(a) for a in (stage.get("nursing_items") or [])][:6],
                    "monitoring_items": [str(a) for a in (stage.get("monitoring_items") or [])][:6],
                    "entry_criteria": [str(a) for a in (stage.get("entry_criteria") or [])][:4],
                    "exit_criteria": [str(a) for a in (stage.get("exit_criteria") or [])][:4],
                    "next_stages": [
                        t.name
                        for _e, t in self.kg.neighbours(stage.id, {EdgeType.NEXT_STAGE.value})
                    ],
                }
            )
        context: Dict[str, Any] = {"disease": disease, "stages": rendered}
        syndrome = coerce_str(item.get("syndrome"))
        if syndrome:
            found = self.kg.find_by_name(syndrome, [NodeType.SYNDROME.value])
            if found:
                context["treatment"] = {
                    "syndrome": found[0].name,
                    "principles": [
                        t.name
                        for _e, t in self.kg.neighbours(
                            found[0].id, {EdgeType.TREATED_BY_PRINCIPLE.value}
                        )
                    ][:4],
                    "formulas": [
                        t.name
                        for _e, t in self.kg.neighbours(found[0].id, {EdgeType.USES_FORMULA.value})
                    ][:4],
                }
        return context

    def verify_arguments(
        self, result: Mapping[str, Any], item: Mapping[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Re-run the deterministic transition evaluator on transition items."""
        if item.get("subtask") != "CP6_transition_decision":
            return None
        stage_id = coerce_str(item.get("stage_id"))
        findings = [str(f) for f in (item.get("followup_findings") or [])]
        if not stage_id or not findings:
            return None
        return [
            {
                "_tool": "evaluate_pathway_transition",
                "stage_id": stage_id,
                "disease": coerce_str(item.get("disease")),
                "findings": findings,
            }
        ]

    def normalise_result(
        self, result: Mapping[str, Any], item: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        out = dict(result)
        out["answer"] = self.clamp_letters(result.get("answer"), (item or {}).get("options") or {})
        return out


TASKS = {"sdt": SDTTask, "pa": PATask, "cp": ClinicalPathwayTask}


def build_task(name: str, kg: KGStore, retriever, budget: Optional[ContextBudget] = None) -> Task:
    key = name.lower()
    if key not in TASKS:
        raise ValueError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[key](kg, retriever, budget)
