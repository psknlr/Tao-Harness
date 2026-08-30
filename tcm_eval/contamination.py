"""Does the knowledge graph already contain the benchmark's answers?

This is the audit the whole design rests on, and paired comparison cannot do
it for you.

Pre-training contamination is a shared confound: if a frontier model memorised
a classical medical record, every arm of that model memorised it equally, and
the paired ``M1 → M2`` difference cancels it. Graph contamination is not
shared. Only M2, M3 and M4 can read the graph. So if a benchmark case and its
answer are sitting in the graph's evidence text, the KG arms are retrieving the
answer key while the no-KG arms reason -- and that difference lands in exactly
the contrast the study reports as the knowledge-graph effect.

The audit works over the text a KG arm can actually reach: node definition
sentences (7,257 of them), edge evidence sentences (32,403) and
``DocumentSource`` metadata. Four levels, cheapest first:

``exact``
    A normalised gold answer string appears verbatim in graph text.
``ngram``
    Character 5-gram Jaccard between the case narrative and a graph sentence.
    Chinese clinical prose is rewritten between sources far more often than it
    is copied, so verbatim matching alone would miss a paraphrased record.
``containment``
    The share of the *case's* n-grams present in one graph sentence. Jaccard
    punishes a short sentence matched against a long case; containment does
    not, and a graph sentence that is a subset of the case is exactly the
    shape a leak takes.
``provenance``
    The case's cited source, where the benchmark records one, against
    ``DocumentSource`` titles and filenames.

Everything here is deterministic and dependency-free: character n-grams need no
segmenter, and the same input gives the same verdict on any machine. There is
no embedding level -- an embedding model would be a second uncontrolled
variable in an audit whose whole purpose is to be checkable.

The output is a stratum per case (``clean`` / ``possible`` / ``likely``), not a
yes-or-no verdict on the benchmark. What matters for the paper is the
sensitivity analysis it enables: report the KG contrasts on the clean stratum
alone. A gain that survives there cannot be read as retrieval of the answer.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

#: Character n-gram width. 5 is long enough that common clinical phrasing
#: ("舌淡红脉细") does not dominate, short enough to survive the reordering a
#: rewritten case record undergoes.
NGRAM = 5

#: A gold string shorter than this is a category label, not a leak: 血瘀证
#: appears in hundreds of guidelines and its presence says nothing.
MIN_EXACT_LEN = 6

#: Stratum thresholds on the strongest similarity found for a case.
LIKELY = 0.40
POSSIBLE = 0.20

STRATA = ("clean", "possible", "likely")


def normalise(text: str) -> str:
    """Width- and punctuation-insensitive form, for comparing two passages."""
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"[\s\W_]+", "", text)


def ngrams(text: str, n: int = NGRAM) -> Set[str]:
    text = normalise(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(needle: Set[str], haystack: Set[str]) -> float:
    """Share of ``needle`` present in ``haystack``."""
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


@dataclass
class GraphText:
    """Every passage a KG arm can read, indexed for overlap search.

    Built once and reused across cases. The inverted index over n-grams is what
    makes this tractable: 40k passages against 328 cases is 13M comparisons
    done naively, and a few thousand after candidate generation.
    """

    passages: List[str] = field(default_factory=list)
    origins: List[str] = field(default_factory=list)
    _grams: List[Set[str]] = field(default_factory=list)
    _index: Dict[str, Set[int]] = field(default_factory=dict)
    doc_titles: List[str] = field(default_factory=list)

    @classmethod
    def from_kg(cls, kg: Any) -> "GraphText":
        """Index everything an agent can read, not everything easy to reach.

        The first version indexed node ``first_mention`` sentences and edge
        evidence, and the docstring claimed that was "the text a KG arm can
        reach". It was not. Measured against the attributes the tools return
        and the retriever indexes, it missed **99.8% of PathwayStage text,
        100% of PharmacoPoeiaEntry, 100% of ExternalTherapy** and 96% of
        SafetyContext -- so a case drawn from a monitoring item, a
        pharmacopoeia function or an external-therapy protocol was scored
        `clean` while `retrieve_pathway_stage` would hand it straight to M3.

        The corpus is now built from two sources that between them define
        reachability:

        * ``KGStore.virtual_document`` for every node -- by construction the
          exact text the retriever indexes, so the audit cannot fall behind
          retrieval without the index falling behind too;
        * every string-valued node attribute as its own **atom**, because a
          virtual document is a long concatenation and a leak is usually one
          field. Matching the concatenation alone lets a single leaked
          monitoring item disappear into a 4,000-character document.

        Both are needed: atoms give precision, whole documents catch a case
        assembled from several fields of one entity.
        """
        self = cls()
        seen: Set[str] = set()

        def add(text: str, origin: str) -> None:
            text = str(text or "").strip()
            key = normalise(text)
            if len(key) < NGRAM or key in seen:
                return
            seen.add(key)
            self.passages.append(text)
            self.origins.append(origin)

        def atoms(value: Any, origin: str) -> None:
            """Every leaf string under a node attribute, however nested."""
            if isinstance(value, str):
                add(value, origin)
            elif isinstance(value, Mapping):
                for item in value.values():
                    atoms(item, origin)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    atoms(item, origin)

        for node in kg.nodes.values():
            add(kg.virtual_document(node.id, include_edge_evidence=False),
                f"vdoc:{node.id}")
            for key, value in (node.attrs or {}).items():
                atoms(value, f"node:{node.id}:{key}")
            sentence = node.sentence()
            if sentence:
                add(sentence, f"node:{node.id}")
            if str(node.type) == "DocumentSource":
                self.doc_titles.append(
                    f"{node.name} {node.attrs.get('title') or ''} "
                    f"{node.attrs.get('source_filename') or ''}"
                )

        edges = kg.edges.values() if hasattr(kg.edges, "values") else kg.edges
        for edge in edges:
            for sentence in edge.evidence_sentences():
                add(sentence, f"edge:{edge.id}")

        for i, passage in enumerate(self.passages):
            grams = ngrams(passage)
            self._grams.append(grams)
            for gram in grams:
                self._index.setdefault(gram, set()).add(i)
        return self

    def __len__(self) -> int:
        return len(self.passages)

    def best_overlap(
        self, text: str, *, max_candidates: int = 400
    ) -> Dict[str, Any]:
        """Closest graph passage, with overlap measured **both ways**.

        The first version scored only ``graph_in_case`` -- the share of a graph
        passage present in the case. That direction catches a short graph
        sentence quoted inside a long case and misses the opposite, which is
        just as much a leak: a short benchmark case that is a verbatim
        *excerpt* of a long graph passage. Measured on a 544-character passage,
        a 40-character excerpt of it scored 0.315 and landed in ``possible``
        instead of ``likely``; against a longer passage the same excerpt falls
        below threshold entirely and reads as ``clean``.

        Both directions are reported, and the stratum uses the larger. They
        answer different questions -- "did the graph quote this case?" and "is
        this case an excerpt of the graph?" -- and either being high is a leak.
        """
        grams = ngrams(text)
        empty = {
            "jaccard": 0.0,
            "graph_in_case": 0.0,
            "case_in_graph": 0.0,
            "overlap": 0.0,
            "passage": None,
            "origin": None,
        }
        if not grams:
            return empty
        hits: Dict[int, int] = {}
        for gram in grams:
            for i in self._index.get(gram, ()):  # candidate generation
                hits[i] = hits.get(i, 0) + 1
        if not hits:
            return empty
        ranked = sorted(hits.items(), key=lambda kv: -kv[1])[:max_candidates]
        best = dict(empty)
        for i, _n in ranked:
            j = jaccard(grams, self._grams[i])
            graph_in_case = containment(self._grams[i], grams)
            case_in_graph = containment(grams, self._grams[i])
            overlap = max(j, graph_in_case, case_in_graph)
            if overlap > best["overlap"]:
                best = {
                    "jaccard": j,
                    "graph_in_case": graph_in_case,
                    "case_in_graph": case_in_graph,
                    "overlap": overlap,
                    "passage": self.passages[i],
                    "origin": self.origins[i],
                }
        return best

    def contains_exact(self, text: str) -> Optional[str]:
        """A passage containing this string verbatim, ignoring punctuation."""
        key = normalise(text)
        if len(key) < MIN_EXACT_LEN:
            return None
        grams = ngrams(text)
        if not grams:
            return None
        candidates: Dict[int, int] = {}
        for gram in grams:
            for i in self._index.get(gram, ()):
                candidates[i] = candidates.get(i, 0) + 1
        for i, n in sorted(candidates.items(), key=lambda kv: -kv[1])[:200]:
            if key in normalise(self.passages[i]):
                return self.passages[i]
        return None

    def cites_source(self, text: str) -> Optional[str]:
        """A DocumentSource whose title matches a citation in the case."""
        key = normalise(text)
        if len(key) < MIN_EXACT_LEN:
            return None
        for title in self.doc_titles:
            norm = normalise(title)
            if norm and (norm in key or key in norm):
                return title
        return None


#: Which case fields are the *question* (a leak means the case is in the graph)
#: and which are the *answer* (a leak means the answer is).
CASE_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "sdt": ("clinical_data", "clinical_information"),
    "pa": ("question", "explanation"),
    "cp": ("vignette", "question"),
}
ANSWER_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "sdt": ("syndrome_text", "pathogenesis_text", "explanatory_summary",
            "syndrome_differentiation"),
    "pa": ("answer_text", "rule_summary"),
    "cp": ("answer_text",),
}


def _text_of(item: Mapping[str, Any], keys: Sequence[str]) -> str:
    parts: List[str] = []
    for key in keys:
        value = item.get(key)
        if isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def _answer_parts(item: Mapping[str, Any]) -> List[str]:
    """The gold answer as separate strings, one per selected option.

    Concatenating them before matching was a systematic false negative on
    every multi-select item. Two options each lifted verbatim from the graph
    -- from *different* passages, as they would be -- produced a joined string
    that appears in no single passage, so ``answer_exact_in_graph`` came back
    False and the case was filed ``clean`` while both of its correct answers
    sat in the graph. SDT tasks 2 and 3 are multi-select on 27 of 50 cases.
    """
    options = item.get("options") or {}
    letters = item.get("answer_letters") or item.get("syndrome_letters") or []
    parts = [str(options[l]).strip() for l in letters if l in options]
    return [p for p in parts if p]


def audit_case(
    item: Mapping[str, Any], graph: GraphText, kind: str
) -> Dict[str, Any]:
    """Every contamination signal for one case, plus its stratum."""
    case_text = _text_of(item, CASE_FIELDS.get(kind, ()))
    parts = _answer_parts(item)
    field_answer = _text_of(item, ANSWER_FIELDS.get(kind, ())).strip()
    if field_answer:
        parts = parts + [field_answer]

    overlap = graph.best_overlap(case_text) if case_text else graph.best_overlap("")
    cited = graph.cites_source(case_text) if case_text else None

    # Each gold part on its own. One correct option sitting verbatim in the
    # graph is a leak whether or not its siblings do.
    exact_parts = 0
    part_overlap = 0.0
    for part in parts:
        if graph.contains_exact(part):
            exact_parts += 1
        else:
            part_overlap = max(part_overlap, graph.best_overlap(part)["overlap"])

    score = max(overlap["overlap"], part_overlap)
    if exact_parts or cited or score >= LIKELY:
        stratum = "likely"
    elif score >= POSSIBLE:
        stratum = "possible"
    else:
        stratum = "clean"

    return {
        "case_id": str(item.get("id") or ""),
        "stratum": stratum,
        "ngram_jaccard": round(overlap["jaccard"], 4),
        # Both directions, kept apart: "did the graph quote this case?" and
        # "is this case an excerpt of the graph?" are different questions.
        "graph_in_case": round(overlap["graph_in_case"], 4),
        "case_in_graph": round(overlap["case_in_graph"], 4),
        "containment": round(
            max(overlap["graph_in_case"], overlap["case_in_graph"]), 4
        ),
        "overlap": round(score, 4),
        "n_gold_parts": len(parts),
        "n_gold_parts_exact_in_graph": exact_parts,
        "answer_exact_in_graph": bool(exact_parts),
        "max_gold_part_overlap": round(part_overlap, 4),
        "cited_source_in_graph": cited,
        "closest_passage": (overlap["passage"] or "")[:200],
        "closest_origin": overlap["origin"],
        "implicated_origin": overlap["origin"],
    }


def audit_dataset(
    items: Sequence[Mapping[str, Any]], graph: GraphText, kind: str
) -> Dict[str, Any]:
    rows = [audit_case(item, graph, kind) for item in items]
    counts = {s: sum(1 for r in rows if r["stratum"] == s) for s in STRATA}
    return {
        "dataset": kind,
        "n_cases": len(rows),
        "n_graph_passages": len(graph),
        "strata": counts,
        "share_clean": round(counts["clean"] / len(rows), 4) if rows else 0.0,
        "n_answer_exact": sum(1 for r in rows if r["answer_exact_in_graph"]),
        "n_gold_parts_exact": sum(r["n_gold_parts_exact_in_graph"] for r in rows),
        "thresholds": {"likely": LIKELY, "possible": POSSIBLE, "ngram": NGRAM},
        "cases": rows,
    }


#: Bumped whenever the corpus, the similarity measure or the thresholds change.
#: An audit produced under a different version is not comparable, and the
#: corpus rebuild that raised coverage from ~50% to 100% is exactly the kind of
#: change that must invalidate every report written before it.
AUDIT_VERSION = "2"


def audit_identity(
    *,
    kg_hash: str,
    dataset_hash: str,
    case_set_hash: str,
    gold_hash: str = "",
) -> Dict[str, Any]:
    """What an audit is *about*, so a stale one cannot be reused silently."""
    return {
        "audit_version": AUDIT_VERSION,
        "kg_content_sha256": kg_hash,
        "dataset_sha256": dataset_hash,
        "dataset_gold_sha256": gold_hash,
        "case_set_sha256": case_set_hash,
        "ngram": NGRAM,
        "thresholds": {"likely": LIKELY, "possible": POSSIBLE},
        "min_exact_len": MIN_EXACT_LEN,
    }


def identity_conflicts(
    frozen: Mapping[str, Any], current: Mapping[str, Any]
) -> List[str]:
    """Reasons a stored audit does not describe the run being scored.

    Without this, editing the graph and forgetting to re-run the audit left the
    old ``clean`` stratification in place -- and the contamination sensitivity
    analysis, whose entire purpose is to show the gain is not retrieval, would
    have been computed against a graph that no longer existed.
    """
    out: List[str] = []
    for key, value in current.items():
        before = frozen.get(key)
        if before in (None, "", {}) or value in (None, "", {}):
            continue
        if before != value:
            out.append(f"{key}: audit {str(before)[:16]} != current {str(value)[:16]}")
    return out


def load_report(path: str | Path) -> Dict[str, Any]:
    """A written audit: ``{"identity": ..., "cases": {case_id: row}}``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "identity": payload.get("identity") or {},
        "cases": {r["case_id"]: r for r in payload.get("cases", [])},
    }


def format_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Contamination audit — {report['dataset'].upper()}",
        "",
        f"{report['n_cases']} cases against {report['n_graph_passages']} graph passages "
        f"(character {report['thresholds']['ngram']}-grams).",
        "",
        "| stratum | n | share |",
        "|---|---|---|",
    ]
    total = report["n_cases"] or 1
    for stratum in STRATA:
        n = report["strata"][stratum]
        lines.append(f"| {stratum} | {n} | {n / total:.1%} |")
    lines += [
        "",
        f"Gold answer found verbatim in graph text: **{report['n_answer_exact']}**.",
        "",
        "A case is `likely` when its gold answer appears verbatim in the graph, "
        "when it cites a source the graph holds, or when its narrative overlaps a "
        f"graph passage at ≥ {report['thresholds']['likely']:.0%}; `possible` at "
        f"≥ {report['thresholds']['possible']:.0%}.",
        "",
        "**The number that matters is not this table but the sensitivity analysis "
        "it enables.** Report the KG contrasts on the `clean` stratum alone.",
        "",
        "State the result as what it is. A gain persisting on the lexically clean "
        "subset is **less consistent with detectable direct lexical or provenance "
        "overlap** — it is not proof that no contamination exists. This audit is "
        "lexical and deterministic by design; a case rewritten past character "
        "5-gram similarity, or paraphrased into different vocabulary, would pass "
        "it. `clean` means *this audit found nothing*, not *there is nothing*.",
    ]
    worst = sorted(report["cases"], key=lambda r: -r.get("overlap", 0.0))[:10]
    if worst:
        lines += [
            "",
            "## Closest cases",
            "",
            "| case | stratum | Jaccard | graph in case | case in graph | gold parts in graph | closest graph passage |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in worst:
            passage = (row["closest_passage"] or "").replace("|", "／")[:60]
            lines.append(
                f"| {row['case_id']} | {row['stratum']} | {row['ngram_jaccard']:.3f} "
                f"| {row.get('graph_in_case', 0):.3f} | {row.get('case_in_graph', 0):.3f} "
                f"| {row.get('n_gold_parts_exact_in_graph', 0)}/{row.get('n_gold_parts', 0)} "
                f"| {passage} |"
            )
    return "\n".join(lines)
