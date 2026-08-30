"""In-memory, fully indexed view of the TCM knowledge graph.

Design notes
------------
* No third-party graph library.  The graph is small (9,350 nodes / 27,972 edges)
  and a hand-rolled store keeps the harness dependency-free, which matters
  because every experimental condition must be reproducible from a clean
  checkout years later.
* Provenance is resolved through the ``source_docs`` attribute rather than
  through ``CITES_DOCUMENT`` edges: in the delivered graph ``CITES_DOCUMENT``
  only ever connects ``Disease -> DocumentSource`` (632 edges), while
  ``source_docs`` is present on 25,756 edges and on almost every node.
* ``SAME_AS`` / ``ALIAS_OF`` / ``PROCESSED_FROM`` / ``DERIVED_FROM`` between
  herbs are collapsed into canonical clusters so that a prescription naming
  ``全瓜蒌`` and a pharmacopoeia entry named ``瓜蒌`` resolve to one another.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .normalize import (
    canonical_department,
    canonical_syndrome,
    dedupe,
    markers_for_herb_in_sentence,
    normalize_text,
    split_herb_annotation,
)
from .schema import (
    INDEXED_TEXT_FIELDS,
    PROVENANCE_ATTR,
    SYMMETRIC_EDGES,
    Domain,
    DomainPolicy,
    EdgeType,
    NodeType,
    policy_for,
)

#: Herb-to-herb relations that mean "these two names denote the same drug for
#: retrieval purposes".  ``PROCESSED_FROM`` is included because 清半夏 and 半夏
#: share a pharmacopoeia lineage, but the flag is kept on the edge so a checker
#: can still distinguish processed from crude drug when that matters.
IDENTITY_EDGES: FrozenSet[str] = frozenset(
    {
        EdgeType.SAME_AS.value,
        EdgeType.ALIAS_OF.value,
        EdgeType.DERIVED_FROM.value,
        EdgeType.PROCESSED_FROM.value,
    }
)


@dataclass(frozen=True)
class Edge:
    """A typed, evidence-carrying relation."""

    id: str
    type: str
    source: str
    target: str
    source_docs: Tuple[str, ...] = ()
    evidence: Tuple[Mapping[str, Any], ...] = ()
    evidence_type: str = ""
    claim_type: str = ""

    def evidence_sentences(self) -> List[str]:
        out: List[str] = []
        for item in self.evidence:
            if isinstance(item, Mapping):
                sentence = item.get("sentence")
                if sentence:
                    out.append(str(sentence))
            elif isinstance(item, str):
                out.append(item)
        return out


@dataclass
class Node:
    """A graph entity plus the derived fields the tools rely on."""

    id: str
    type: str
    name: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    source_docs: Tuple[str, ...] = ()
    #: verbatim sentence where the entity first appears in its source document
    first_mention: Optional[Mapping[str, Any]] = None
    #: herb name with inline preparation annotation removed
    base_name: str = ""
    #: preparation markers lifted out of the name (先煎 / 后下 / ...)
    prep_markers: Tuple[str, ...] = ()

    def sentence(self) -> str:
        if isinstance(self.first_mention, Mapping):
            return str(self.first_mention.get("sentence") or "")
        return ""

    def get(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)


class KGStore:
    """Indexed access to nodes, edges, provenance and canonical clusters."""

    def __init__(self, nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._out: Dict[str, List[int]] = defaultdict(list)
        self._in: Dict[str, List[int]] = defaultdict(list)
        self._by_type: Dict[str, List[str]] = defaultdict(list)
        self._by_name: Dict[str, List[str]] = defaultdict(list)
        self._by_base_name: Dict[str, List[str]] = defaultdict(list)
        self._doc_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        self._doc_index: Dict[str, str] = {}
        self._canonical: Dict[str, str] = {}
        self._cluster: Dict[str, List[str]] = defaultdict(list)
        self._content_hash: Optional[str] = None

        self._load_nodes(nodes)
        self._load_edges(edges)
        self._build_identity_clusters()

    # ---------------------------------------------------------------- loading
    def _load_nodes(self, nodes: Sequence[Mapping[str, Any]]) -> None:
        for raw in nodes:
            node_id = str(raw.get("id"))
            ntype = str(raw.get("type"))
            name = str(raw.get("name") or raw.get("title") or raw.get("doc_id") or "")
            if ntype == NodeType.DEPARTMENT.value:
                name = canonical_department(name)
            attrs = {
                k: v
                for k, v in raw.items()
                if k not in {"id", "type", "name", PROVENANCE_ATTR, "first_mention"}
            }
            docs = tuple(str(d) for d in (raw.get(PROVENANCE_ATTR) or ()))
            base_name, markers = (
                split_herb_annotation(name)
                if ntype in {NodeType.HERB.value, NodeType.FORMULA.value}
                else (name, [])
            )
            node = Node(
                id=node_id,
                type=ntype,
                name=name,
                attrs=attrs,
                source_docs=docs,
                first_mention=raw.get("first_mention"),
                base_name=base_name,
                prep_markers=tuple(markers),
            )
            self.nodes[node_id] = node
            self._by_type[ntype].append(node_id)
            self._by_name[normalize_text(name)].append(node_id)
            if base_name != name:
                self._by_base_name[normalize_text(base_name)].append(node_id)
            if ntype == NodeType.SYNDROME.value:
                self._by_base_name[normalize_text(canonical_syndrome(name))].append(node_id)
            for doc in docs:
                self._doc_to_nodes[doc].add(node_id)
            if ntype == NodeType.DOCUMENT_SOURCE.value:
                doc_id = str(raw.get("doc_id") or "")
                if doc_id:
                    self._doc_index[doc_id] = node_id
                    self._doc_to_nodes[doc_id].add(node_id)
            # pharmacopoeia aliases are searchable names too
            for alias in self._aliases_of(raw):
                self._by_base_name[normalize_text(alias)].append(node_id)

    @staticmethod
    def _aliases_of(raw: Mapping[str, Any]) -> List[str]:
        out: List[str] = []
        entry = raw.get("pharmacopoeia_entry")
        if isinstance(entry, Mapping):
            out.extend(str(a) for a in (entry.get("common_aliases") or ()))
            canonical = entry.get("canonical_name")
            if canonical:
                out.append(str(canonical))
        out.extend(str(a) for a in (raw.get("common_aliases") or ()))
        return out

    def _load_edges(self, edges: Sequence[Mapping[str, Any]]) -> None:
        for raw in edges:
            src, dst = str(raw.get("from")), str(raw.get("to"))
            if src not in self.nodes or dst not in self.nodes:
                continue  # dangling edges are dropped and reported by validate_graph
            evidence = raw.get("evidence") or ()
            if isinstance(evidence, Mapping):
                evidence = (evidence,)
            edge = Edge(
                id=str(raw.get("_id") or f"{src}|{raw.get('type')}|{dst}"),
                type=str(raw.get("type")),
                source=src,
                target=dst,
                source_docs=tuple(str(d) for d in (raw.get(PROVENANCE_ATTR) or ())),
                evidence=tuple(e for e in evidence if isinstance(e, (Mapping, str))),
                evidence_type=str(raw.get("evidence_type") or ""),
                claim_type=str(raw.get("claim_type") or ""),
            )
            idx = len(self.edges)
            self.edges.append(edge)
            self._out[src].append(idx)
            self._in[dst].append(idx)
            for doc in edge.source_docs:
                self._doc_to_nodes[doc].update((src, dst))

    def _build_identity_clusters(self) -> None:
        """Union-find over identity edges, canonicalising to the shortest name.

        Shortest-name canonicalisation makes 瓜蒌 (not 全瓜蒌) the cluster head,
        which is also the pharmacopoeia head-word, so pharmacopoeia lookups hit.
        """
        parent: Dict[str, str] = {nid: nid for nid in self.nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for edge in self.edges:
            if edge.type in IDENTITY_EDGES:
                union(edge.source, edge.target)

        groups: Dict[str, List[str]] = defaultdict(list)
        for nid in self.nodes:
            groups[find(nid)].append(nid)

        for members in groups.values():
            head = min(members, key=lambda n: (len(self.nodes[n].base_name or self.nodes[n].name), n))
            for member in members:
                self._canonical[member] = head
            self._cluster[head] = sorted(members)

    # ---------------------------------------------------------------- lookups
    def __len__(self) -> int:
        return len(self.nodes)

    def node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def of_type(self, node_type: str | NodeType) -> List[Node]:
        key = node_type.value if isinstance(node_type, NodeType) else node_type
        return [self.nodes[n] for n in self._by_type.get(key, ())]

    def type_counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in sorted(self._by_type.items())}

    def edge_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for edge in self.edges:
            counts[edge.type] += 1
        return dict(sorted(counts.items()))

    def find_by_name(
        self, name: str, node_types: Optional[Iterable[str]] = None
    ) -> List[Node]:
        """Exact (normalised) name lookup, including aliases and base names."""
        key = normalize_text(name)
        ids = list(self._by_name.get(key, ())) + list(self._by_base_name.get(key, ()))
        wanted = set(node_types) if node_types else None
        out: List[Node] = []
        seen: Set[str] = set()
        for nid in ids:
            if nid in seen:
                continue
            seen.add(nid)
            node = self.nodes[nid]
            if wanted is None or node.type in wanted:
                out.append(node)
        return out

    def canonical_id(self, node_id: str) -> str:
        return self._canonical.get(node_id, node_id)

    def cluster(self, node_id: str) -> List[Node]:
        """All nodes denoting the same entity (alias / processed / derived)."""
        head = self.canonical_id(node_id)
        return [self.nodes[n] for n in self._cluster.get(head, [node_id])]

    # -------------------------------------------------------------- traversal
    def out_edges(
        self, node_id: str, edge_types: Optional[Iterable[str]] = None
    ) -> List[Edge]:
        wanted = set(edge_types) if edge_types else None
        return [
            self.edges[i]
            for i in self._out.get(node_id, ())
            if wanted is None or self.edges[i].type in wanted
        ]

    def in_edges(
        self, node_id: str, edge_types: Optional[Iterable[str]] = None
    ) -> List[Edge]:
        wanted = set(edge_types) if edge_types else None
        return [
            self.edges[i]
            for i in self._in.get(node_id, ())
            if wanted is None or self.edges[i].type in wanted
        ]

    def neighbours(
        self,
        node_id: str,
        edge_types: Optional[Iterable[str]] = None,
        *,
        direction: str = "out",
    ) -> List[Tuple[Edge, Node]]:
        out: List[Tuple[Edge, Node]] = []
        if direction in {"out", "both"}:
            for edge in self.out_edges(node_id, edge_types):
                out.append((edge, self.nodes[edge.target]))
        if direction in {"in", "both"}:
            for edge in self.in_edges(node_id, edge_types):
                out.append((edge, self.nodes[edge.source]))
        return out

    def expand(
        self,
        seeds: Iterable[str],
        *,
        hops: int = 2,
        allowed_types: Optional[FrozenSet[str]] = None,
        edge_types: Optional[Iterable[str]] = None,
        decay: float = 0.5,
        max_nodes: int = 400,
    ) -> Dict[str, float]:
        """Breadth-first typed expansion returning a distance-decayed score.

        This is the ``GraphRelevance`` term of the hybrid retrieval score.  It is
        deliberately a simple decayed BFS rather than personalised PageRank:
        with a graph this small the two rank almost identically, and BFS is
        trivially explainable in a paper and cheap to freeze.
        """
        scores: Dict[str, float] = {}
        frontier: List[str] = []
        for seed in seeds:
            if seed in self.nodes:
                scores[seed] = max(scores.get(seed, 0.0), 1.0)
                frontier.append(seed)
        for hop in range(1, hops + 1):
            weight = decay**hop
            nxt: List[str] = []
            for nid in frontier:
                for _edge, neighbour in self.neighbours(
                    nid, edge_types, direction="both"
                ):
                    if allowed_types is not None and neighbour.type not in allowed_types:
                        continue
                    if scores.get(neighbour.id, 0.0) >= weight:
                        continue
                    scores[neighbour.id] = weight
                    nxt.append(neighbour.id)
            frontier = nxt
            if len(scores) >= max_nodes:
                break
        return scores

    # ------------------------------------------------------------- provenance
    def document(self, doc_id: str) -> Optional[Node]:
        node_id = self._doc_index.get(doc_id)
        return self.nodes.get(node_id) if node_id else None

    def documents_for(self, node_id: str) -> List[Node]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        docs = [self.document(d) for d in node.source_docs]
        return [d for d in docs if d is not None]

    def nodes_in_document(self, doc_id: str) -> List[Node]:
        return [self.nodes[n] for n in sorted(self._doc_to_nodes.get(doc_id, ()))]

    def doc_overlap(self, node_id: str, docs: Iterable[str]) -> float:
        """Fraction of ``docs`` that also cite ``node_id`` -- the SourceEvidence term."""
        docs = list(docs)
        if not docs:
            return 0.0
        node = self.nodes.get(node_id)
        if node is None or not node.source_docs:
            return 0.0
        node_docs = set(node.source_docs)
        return sum(1 for d in docs if d in node_docs) / len(docs)

    # ------------------------------------------------------- text projections
    def virtual_document(self, node_id: str, *, include_edge_evidence: bool = True) -> str:
        """Concatenated searchable text for a node.

        This is what makes hybrid graph-RAG possible on a graph that stores no
        document bodies: the retrievable text is assembled from entity names,
        aliases, the verbatim ``first_mention`` sentence, type-specific
        structured fields, and -- crucially -- the evidence sentences of the
        edges incident to the node, which is where preparation requirements
        (先煎/后下/烊化) and dietary restrictions actually live.
        """
        node = self.nodes.get(node_id)
        if node is None:
            return ""
        parts: List[str] = [node.name]
        if node.base_name and node.base_name != node.name:
            parts.append(node.base_name)
        for field_name in INDEXED_TEXT_FIELDS.get(node.type, ()):
            value = node.attrs.get(field_name) if field_name != "name" else node.name
            if isinstance(value, (list, tuple)):
                parts.extend(str(v) for v in value)
            elif value:
                parts.append(str(value))
        parts.append(node.sentence())
        entry = node.attrs.get("pharmacopoeia_entry")
        if isinstance(entry, Mapping):
            for key in (
                "canonical_name",
                "part_used",
                "nature_taste_meridian",
                "pharmacopoeial_functions",
            ):
                if entry.get(key):
                    parts.append(str(entry[key]))
            parts.extend(str(a) for a in (entry.get("common_aliases") or ()))
        for alias in node.attrs.get("common_aliases") or ():
            parts.append(str(alias))
        if include_edge_evidence:
            for edge in self.out_edges(node_id) + self.in_edges(node_id):
                parts.extend(edge.evidence_sentences())
        return " ".join(dedupe(p.strip() for p in parts if p and str(p).strip()))

    def preparation_markers(self, herb_id: str) -> Dict[str, List[str]]:
        """Preparation requirements attested for a herb, with their evidence.

        Returns ``{marker: [evidence sentence, ...]}``.  Markers come from the
        herb's own annotated name and from the evidence of every incident
        ``CONTAINS_HERB`` / ``USES_HERB_DIRECT`` edge.
        """
        found: Dict[str, List[str]] = defaultdict(list)
        node = self.nodes.get(herb_id)
        if node is None:
            return {}
        members = self.cluster(herb_id)
        names = dedupe(
            [node.name, node.base_name]
            + [m.name for m in members]
            + [m.base_name for m in members]
        )
        for member in members:
            for marker in member.prep_markers:
                if member.name not in found[marker]:
                    found[marker].append(member.name)
            for edge in self.in_edges(
                member.id,
                {EdgeType.CONTAINS_HERB.value, EdgeType.USES_HERB_DIRECT.value},
            ):
                for sentence in edge.evidence_sentences():
                    for marker in markers_for_herb_in_sentence(sentence, names):
                        if sentence not in found[marker]:
                            found[marker].append(sentence)
        return {k: v for k, v in found.items()}

    # -------------------------------------------------------------- filtering
    def filter_by_domain(
        self, node_ids: Iterable[str], policy: DomainPolicy, *, verification: bool = False
    ) -> List[str]:
        return [
            nid
            for nid in node_ids
            if nid in self.nodes
            and policy.may_return(self.nodes[nid].type, verification=verification)
        ]

    def content_hash(self) -> str:
        """Semantic fingerprint of the graph's contents.

        Computed over every node's identity, type, name and attributes and
        every edge's endpoints, type and evidence -- not over the file bytes,
        so the JSON and GraphML exports of one graph hash identically while any
        change to a node attribute or a relation changes the hash.

        This is what the retrieval cache and the run manifest key on. Keying on
        node *count*, as an earlier version did, meant that editing a thousand
        relations or rewriting two thousand node attributes left the count at
        9,350 and silently reused a stale index -- a reproducibility failure
        that would surface as inexplicable score drift.
        """
        if self._content_hash is None:
            digest = hashlib.sha256()
            for node_id in sorted(self.nodes):
                node = self.nodes[node_id]
                digest.update(node_id.encode("utf-8"))
                digest.update(node.type.encode("utf-8"))
                digest.update(node.name.encode("utf-8"))
                digest.update(
                    json.dumps(node.attrs, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
                )
                digest.update(
                    json.dumps(node.first_mention, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
                )
            for edge in sorted(self.edges, key=lambda e: (e.source, e.type, e.target, e.id)):
                digest.update(f"{edge.source}|{edge.type}|{edge.target}".encode("utf-8"))
                digest.update(
                    json.dumps(edge.evidence, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
                )
            self._content_hash = digest.hexdigest()
        return self._content_hash

    def summary(self) -> Dict[str, Any]:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "node_types": self.type_counts(),
            "edge_types": self.edge_type_counts(),
            "n_documents": len(self._doc_index),
            "n_identity_clusters": sum(1 for v in self._cluster.values() if len(v) > 1),
            "content_hash": self.content_hash()[:16],
        }
