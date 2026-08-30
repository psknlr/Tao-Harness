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

        for node in kg.nodes.values():
            sentence = node.sentence()
            if sentence:
                add(sentence, f"node:{node.id}")
            if str(node.type) == "DocumentSource":
                self.doc_titles.append(f"{node.name} {node.get('title') or ''} "
                                       f"{node.get('source_filename') or ''}")

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
    ) -> Tuple[float, float, Optional[str], Optional[str]]:
        """``(jaccard, containment, passage, origin)`` for the closest passage."""
        grams = ngrams(text)
        if not grams:
            return 0.0, 0.0, None, None
        hits: Dict[int, int] = {}
        for gram in grams:
            for i in self._index.get(gram, ()):  # candidate generation
                hits[i] = hits.get(i, 0) + 1
        if not hits:
            return 0.0, 0.0, None, None
        ranked = sorted(hits.items(), key=lambda kv: -kv[1])[:max_candidates]
        best = (0.0, 0.0, None, None)
        for i, _n in ranked:
            j = jaccard(grams, self._grams[i])
            c = containment(self._grams[i], grams)  # graph text inside the case
            if max(j, c) > max(best[0], best[1]):
                best = (j, c, self.passages[i], self.origins[i])
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


def _answer_text(item: Mapping[str, Any]) -> str:
    """The gold answer as prose, whatever shape the benchmark stores it in."""
    options = item.get("options") or {}
    letters = item.get("answer_letters") or item.get("syndrome_letters") or []
    chosen = [str(options[l]) for l in letters if l in options]
    return " ".join(chosen)


def audit_case(
    item: Mapping[str, Any], graph: GraphText, kind: str
) -> Dict[str, Any]:
    """Every contamination signal for one case, plus its stratum."""
    case_text = _text_of(item, CASE_FIELDS.get(kind, ()))
    answer_text = " ".join(
        [_text_of(item, ANSWER_FIELDS.get(kind, ())), _answer_text(item)]
    ).strip()

    j, c, passage, origin = graph.best_overlap(case_text) if case_text else (0.0, 0.0, None, None)
    exact = graph.contains_exact(answer_text) if answer_text else None
    cited = graph.cites_source(case_text) if case_text else None

    # An exact answer match is decisive on its own; otherwise the strongest
    # overlap decides. Containment counts as much as Jaccard: a graph sentence
    # wholly inside the case is a leak even when the case is much longer.
    score = max(j, c)
    if exact:
        stratum = "likely"
    elif score >= LIKELY or cited:
        stratum = "likely"
    elif score >= POSSIBLE:
        stratum = "possible"
    else:
        stratum = "clean"

    return {
        "case_id": str(item.get("id") or ""),
        "stratum": stratum,
        "ngram_jaccard": round(j, 4),
        "containment": round(c, 4),
        "answer_exact_in_graph": bool(exact),
        "cited_source_in_graph": cited,
        "closest_passage": (passage or "")[:200],
        "closest_origin": origin,
        # Every source document behind the closest passage, so a
        # leave-source-out run knows what to withhold for this case.
        "implicated_origin": origin,
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
        "thresholds": {"likely": LIKELY, "possible": POSSIBLE, "ngram": NGRAM},
        "cases": rows,
    }


def load_report(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """``{case_id: row}`` from a written audit, for scoring-time stratification."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["case_id"]: r for r in payload.get("cases", [])}


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
        "it enables.** Report the KG contrasts on the `clean` stratum alone: a gain "
        "that survives there cannot be explained as retrieving the answer from the "
        "graph, which is the one reading a paired design cannot rule out by itself.",
    ]
    worst = sorted(
        report["cases"], key=lambda r: -max(r["ngram_jaccard"], r["containment"])
    )[:10]
    if worst:
        lines += ["", "## Closest cases", "", "| case | stratum | Jaccard | containment | closest graph passage |", "|---|---|---|---|---|"]
        for row in worst:
            passage = (row["closest_passage"] or "").replace("|", "／")[:70]
            lines.append(
                f"| {row['case_id']} | {row['stratum']} | {row['ngram_jaccard']:.3f} "
                f"| {row['containment']:.3f} | {passage} |"
            )
    return "\n".join(lines)
