"""Hybrid graph-RAG retrieval over the TCM knowledge graph.

The graph stores no document bodies, so "semantic retrieval over
DocumentSource" cannot mean retrieving text passages.  What it can mean -- and
what this module implements -- is retrieval over a *virtual document* per
entity, assembled from names, aliases, the verbatim ``first_mention`` sentence
and the evidence sentences of incident edges (:meth:`KGStore.virtual_document`).
For 312 of 648 Syndrome nodes that virtual document contains the protocol's own
symptom/tongue/pulse sentence, which is exactly the evidence a case description
should match against.

Final ranking follows the score decomposition

    Score(v) = alpha * SemanticSimilarity(q, v)
             + beta  * GraphRelevance(anchors, v)
             + gamma * SourceEvidence(anchor_docs, v)

with all three terms normalised to ``[0, 1]``:

* **SemanticSimilarity** -- BM25 over character n-grams (optionally blended
  with a dense encoder, see :class:`EmbeddingProvider`).
* **GraphRelevance** -- distance-decayed BFS from the anchor set
  (:meth:`KGStore.expand`).
* **SourceEvidence** -- fraction of the anchors' source documents that also
  attest the candidate, i.e. document-level co-occurrence.

The lexical backend is the default because it is deterministic, offline and
identical for every model under test; a dense backend is opt-in and, once
chosen, is frozen for the whole experiment by the framework hash.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from .normalize import char_ngrams
from .schema import Domain, DomainPolicy, policy_for
from .store import KGStore

INDEX_VERSION = "1.0.0"


class EmbeddingProvider(Protocol):
    """Optional dense backend.

    Deliberately a protocol rather than a dependency: the headline experiments
    run on the lexical backend so that retrieval is byte-reproducible without a
    model download, while a lab that wants dense retrieval can register any
    encoder and have it frozen alongside everything else.
    """

    name: str

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:  # pragma: no cover
        ...


@dataclass(frozen=True)
class RetrievalParams:
    """Frozen retrieval hyper-parameters.

    These are part of the framework contract: identical for every model, and
    hashed into the run's ``framework_hash``.
    """

    alpha_semantic: float = 0.60
    beta_graph: float = 0.25
    gamma_source: float = 0.15
    top_k: int = 8
    graph_hops: int = 2
    graph_decay: float = 0.5
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    ngram_sizes: Tuple[int, ...] = (1, 2, 3)
    min_score: float = 0.0
    #: weight of the dense score inside SemanticSimilarity when an encoder is set
    dense_weight: float = 0.0

    def fingerprint(self) -> str:
        payload = "|".join(
            f"{k}={getattr(self, k)}" for k in sorted(self.__dataclass_fields__)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Hit:
    """A ranked retrieval result with its score decomposition."""

    node_id: str
    node_type: str
    name: str
    score: float
    semantic: float = 0.0
    graph: float = 0.0
    source: float = 0.0
    matched_text: str = ""
    source_docs: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "name": self.name,
            "score": round(self.score, 4),
            "score_parts": {
                "semantic": round(self.semantic, 4),
                "graph": round(self.graph, 4),
                "source": round(self.source, 4),
            },
            "evidence": self.matched_text,
            "source_docs": list(self.source_docs),
        }


class BM25Index:
    """Okapi BM25 over character n-grams. Pure Python, deterministic."""

    def __init__(self, params: RetrievalParams):
        self.params = params
        self.doc_ids: List[str] = []
        self.doc_len: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.avg_len: float = 0.0
        self._idf: Dict[str, float] = {}

    def build(self, documents: Mapping[str, str]) -> "BM25Index":
        for doc_id, text in documents.items():
            tokens = char_ngrams(text, self.params.ngram_sizes)
            idx = len(self.doc_ids)
            self.doc_ids.append(doc_id)
            self.doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self.postings[term].append((idx, freq))
        n_docs = max(1, len(self.doc_ids))
        self.avg_len = sum(self.doc_len) / n_docs
        for term, plist in self.postings.items():
            df = len(plist)
            self.__dict__.setdefault("_idf", {})
            self._idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        return self

    def search(self, query: str, limit: int = 200) -> List[Tuple[str, float]]:
        tokens = char_ngrams(query, self.params.ngram_sizes)
        if not tokens or not self.doc_ids:
            return []
        k1, b = self.params.bm25_k1, self.params.bm25_b
        scores: Dict[int, float] = defaultdict(float)
        for term, q_freq in Counter(tokens).items():
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self._idf.get(term, 0.0)
            for idx, freq in plist:
                denom = freq + k1 * (
                    1 - b + b * (self.doc_len[idx] / (self.avg_len or 1.0))
                )
                scores[idx] += idf * (freq * (k1 + 1) / (denom or 1.0)) * math.log(
                    1 + q_freq
                )
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))
        return [(self.doc_ids[i], s) for i, s in ranked[:limit]]


class KGRetriever:
    """Domain-scoped hybrid retriever."""

    def __init__(
        self,
        kg: KGStore,
        params: Optional[RetrievalParams] = None,
        *,
        encoder: Optional[EmbeddingProvider] = None,
    ):
        self.kg = kg
        self.params = params or RetrievalParams()
        self.encoder = encoder
        self._indexes: Dict[str, BM25Index] = {}
        self._documents: Dict[str, str] = {}

    # ------------------------------------------------------------------ build
    def _virtual_documents(self, allowed_types: FrozenSet[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for node_id, node in self.kg.nodes.items():
            if node.type not in allowed_types:
                continue
            if node_id not in self._documents:
                self._documents[node_id] = self.kg.virtual_document(node_id)
            out[node_id] = self._documents[node_id]
        return out

    def index_for(self, domain: Domain | str, *, verification: bool = False) -> BM25Index:
        policy = policy_for(domain)
        key = f"{policy.domain.value}:{'v' if verification else 'p'}"
        if key not in self._indexes:
            allowed = policy.visible_types(verification=verification)
            self._indexes[key] = BM25Index(self.params).build(
                self._virtual_documents(allowed)
            )
        return self._indexes[key]

    def warm(self, domains: Iterable[Domain | str] = (Domain.CLINICAL, Domain.SAFETY)) -> None:
        for domain in domains:
            self.index_for(domain)
            self.index_for(domain, verification=True)

    # ----------------------------------------------------------------- search
    def search(
        self,
        query: str,
        *,
        domain: Domain | str = Domain.CLINICAL,
        node_types: Optional[Iterable[str]] = None,
        anchors: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        verification: bool = False,
        candidate_pool: int = 300,
    ) -> List[Hit]:
        """Rank entities for ``query`` inside ``domain``.

        ``anchors`` are already-identified node ids (e.g. the disease the case
        was anchored to).  They drive the graph and source-evidence terms; with
        no anchors the score degenerates to pure BM25, which is the intended
        behaviour for the first, un-anchored retrieval pass.
        """
        policy = policy_for(domain)
        params = self.params
        k = top_k or params.top_k
        allowed = policy.visible_types(verification=verification)
        wanted = (
            {t for t in node_types if t in allowed} if node_types is not None else allowed
        )
        if not wanted:
            return []

        lexical = self.index_for(domain, verification=verification).search(
            query, limit=candidate_pool
        )
        max_lex = max((s for _, s in lexical), default=0.0) or 1.0
        semantic: Dict[str, float] = {nid: s / max_lex for nid, s in lexical}

        if self.encoder is not None and params.dense_weight > 0:
            semantic = self._blend_dense(query, semantic)

        anchors = list(anchors or [])
        graph_scores: Dict[str, float] = {}
        if anchors:
            graph_scores = self.kg.expand(
                anchors,
                hops=params.graph_hops,
                allowed_types=frozenset(allowed),
                decay=params.graph_decay,
            )
            # candidates reachable from the anchors deserve consideration even
            # when their surface form shares no n-gram with the query
            for node_id in graph_scores:
                semantic.setdefault(node_id, 0.0)

        anchor_docs: List[str] = []
        for anchor in anchors:
            node = self.kg.node(anchor)
            if node is not None:
                anchor_docs.extend(node.source_docs)

        hits: List[Hit] = []
        for node_id, sem in semantic.items():
            node = self.kg.node(node_id)
            if node is None or node.type not in wanted:
                continue
            graph = graph_scores.get(node_id, 0.0)
            source = self.kg.doc_overlap(node_id, anchor_docs) if anchor_docs else 0.0
            score = (
                params.alpha_semantic * sem
                + params.beta_graph * graph
                + params.gamma_source * source
            )
            if score < params.min_score:
                continue
            hits.append(
                Hit(
                    node_id=node_id,
                    node_type=node.type,
                    name=node.name,
                    score=score,
                    semantic=sem,
                    graph=graph,
                    source=source,
                    matched_text=node.sentence() or node.name,
                    source_docs=node.source_docs,
                )
            )
        hits.sort(key=lambda h: (-h.score, h.node_id))
        return hits[:k]

    def _blend_dense(self, query: str, lexical: Dict[str, float]) -> Dict[str, float]:
        assert self.encoder is not None
        ids = list(lexical)
        vectors = self.encoder.encode([self._documents.get(i, "") for i in ids])
        q_vec = self.encoder.encode([query])[0]
        weight = self.params.dense_weight
        blended: Dict[str, float] = {}
        for node_id, vec in zip(ids, vectors):
            blended[node_id] = (1 - weight) * lexical[node_id] + weight * _cosine(q_vec, vec)
        return blended

    # ------------------------------------------------------------ persistence
    def cache_key(self, kg_fingerprint: str) -> str:
        parts = [INDEX_VERSION, kg_fingerprint, self.params.fingerprint()]
        if self.encoder is not None:
            parts.append(getattr(self.encoder, "name", "encoder"))
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"documents": self._documents, "indexes": self._indexes}, handle)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception:  # a stale or truncated cache must never break a run
            return False
        self._documents = payload.get("documents", {})
        self._indexes = payload.get("indexes", {})
        return True


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return max(0.0, num / (na * nb))
