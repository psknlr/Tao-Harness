"""Shared serialisers.

Every tool returns node summaries through these helpers so that the shape of
the context a model sees is identical no matter which tool produced it.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tcm_kg.normalize import char_ngrams, normalize_text
from tcm_kg.store import Edge, KGStore, Node

#: Attributes never echoed back to the model: bulky, or provenance plumbing that
#: the ``source_docs`` field already conveys.
_HIDDEN_ATTRS = frozenset({"pharmacopoeia_entry", "first_mention", "id", "doc_type_flag"})


def node_brief(
    kg: KGStore,
    node: Node,
    *,
    with_sentence: bool = True,
    with_attrs: Sequence[str] = (),
    max_sentence: int = 220,
) -> Dict[str, Any]:
    """Compact, uniform rendering of one entity."""
    out: Dict[str, Any] = {"id": node.id, "type": node.type, "name": node.name}
    if with_sentence:
        sentence = node.sentence()
        if sentence:
            out["source_sentence"] = sentence[:max_sentence]
    for key in with_attrs:
        value = node.attrs.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    if node.source_docs:
        out["source_docs"] = list(node.source_docs[:6])
    if node.attrs.get("evidence_caveat"):
        out["evidence_caveat"] = str(node.attrs["evidence_caveat"])[:200]
    return out


def edge_evidence(edge: Edge, *, max_items: int = 2, max_chars: int = 220) -> Dict[str, Any]:
    """Provenance block for one relation."""
    sentences = [s[:max_chars] for s in edge.evidence_sentences()[:max_items]]
    payload: Dict[str, Any] = {
        "relation": edge.type,
        "claim_type": edge.claim_type,
        "evidence_type": edge.evidence_type,
    }
    if sentences:
        payload["quotes"] = sentences
    if edge.source_docs:
        payload["source_docs"] = list(edge.source_docs[:6])
    return payload


def doc_brief(node: Node) -> Dict[str, Any]:
    """Rendering of a DocumentSource record."""
    keys = (
        "doc_id",
        "title",
        "doc_type",
        "department",
        "version_label",
        "institution_attribution",
        "publisher",
        "effective_date",
        "supersedes",
        "source_filename",
    )
    out: Dict[str, Any] = {"id": node.id}
    for key in keys:
        value = node.attrs.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    return out


def documents_block(kg: KGStore, node_ids: Iterable[str], *, limit: int = 6) -> List[Dict[str, Any]]:
    """Resolve ``source_docs`` of several nodes into DocumentSource records."""
    seen: List[str] = []
    for node_id in node_ids:
        node = kg.node(node_id)
        if node is None:
            continue
        for doc_id in node.source_docs:
            if doc_id not in seen:
                seen.append(doc_id)
    out: List[Dict[str, Any]] = []
    for doc_id in seen[:limit]:
        doc = kg.document(doc_id)
        if doc is not None:
            out.append(doc_brief(doc))
    return out


def resolve_entity(
    kg: KGStore,
    name: str,
    node_types: Optional[Sequence[str]] = None,
    *,
    retriever=None,
    domain=None,
) -> List[Node]:
    """Exact name / alias lookup, falling back to lexical search.

    Returning several candidates rather than one is deliberate: TCM naming is
    ambiguous (``桂枝`` the herb vs ``桂枝汤`` the formula), and letting the tool
    silently pick one would hide that ambiguity from both the model and the
    trace.
    """
    hits = kg.find_by_name(name, node_types)
    if hits:
        return hits
    # alias / cluster resolution
    for candidate in kg.find_by_name(name):
        for member in kg.cluster(candidate.id):
            if node_types is None or member.type in node_types:
                hits.append(member)
    if hits:
        return hits
    if retriever is not None and domain is not None:
        for hit in retriever.search(
            name, domain=domain, node_types=node_types, top_k=5
        ):
            node = kg.node(hit.node_id)
            if node is not None and name_similarity(name, node.name) >= MIN_NAME_SIMILARITY:
                hits.append(node)
    return hits


#: How close a retrieved name must be to the queried one before a tool will
#: treat them as the same entity.
MIN_NAME_SIMILARITY = 0.5


def name_similarity(query: str, candidate: str) -> float:
    """Character-bigram similarity between two entity *names*.

    BM25 scores are normalised against the best hit, so the top result for any
    query scores 1.0 -- including for a drug the graph has never heard of.
    Resolving on that alone would let a query about an absent drug come back
    with a different drug's contraindications, which in a prescription-audit
    setting is worse than returning nothing. Resolution therefore additionally
    requires the surface forms themselves to overlap.
    """
    left, right = normalize_text(query), normalize_text(candidate)
    if not left or not right:
        return 0.0
    if left == right or left in right or right in left:
        return 1.0
    a, b = set(char_ngrams(left, (2,))), set(char_ngrams(right, (2,)))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
