"""Ontology of the TCM clinical knowledge graph.

The ontology is *descriptive*: every node type, edge type and edge signature
declared here was derived from the delivered graph
(9,350 nodes / 27,972 edges) rather than invented.  Nothing is added to make a
benchmark easier to answer -- in particular there is deliberately **no**
``Symptom`` and no ``Pathogenesis`` node type.  Symptoms, tongue, pulse and
pathogenesis are *transient clinical features* produced by the LLM at run time
(see :mod:`tcm_agent.parser`); they never enter the graph.

The module also defines the two **access domains**.  Domains are the mechanism
that keeps the two benchmarks honest: an SDT agent physically cannot reach
Formula/Herb knowledge, so it cannot back-infer a syndrome from the treatment
that the reference answer implies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, Mapping, Sequence, Set, Tuple


class NodeType(str, Enum):
    """The 14 entity types actually present in the graph."""

    DEPARTMENT = "Department"
    DISEASE = "Disease"
    DISEASE_SUBTYPE = "DiseaseSubtype"
    SYNDROME = "Syndrome"
    TREATMENT_PRINCIPLE = "TreatmentPrinciple"
    FORMULA = "Formula"
    HERB = "Herb"
    PATENT_MEDICINE = "PatentMedicine"
    EXTERNAL_THERAPY = "ExternalTherapy"
    PATHWAY_STAGE = "PathwayStage"
    SAFETY_CONTEXT = "SafetyContext"
    RESTRICTED_ITEM = "RestrictedItem"
    PHARMACOPOEIA_ENTRY = "PharmacoPoeiaEntry"
    DOCUMENT_SOURCE = "DocumentSource"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class EdgeType(str, Enum):
    """The 20 relation types actually present in the graph."""

    BELONGS_TO_DEPARTMENT = "BELONGS_TO_DEPARTMENT"
    HAS_SUBTYPE = "HAS_SUBTYPE"
    HAS_SYNDROME = "HAS_SYNDROME"
    SUBTYPE_HAS_SYNDROME = "SUBTYPE_HAS_SYNDROME"
    TREATED_BY_PRINCIPLE = "TREATED_BY_PRINCIPLE"
    USES_FORMULA = "USES_FORMULA"
    USES_PATENT_MEDICINE = "USES_PATENT_MEDICINE"
    USES_EXTERNAL_THERAPY = "USES_EXTERNAL_THERAPY"
    USES_HERB_DIRECT = "USES_HERB_DIRECT"
    CONTAINS_HERB = "CONTAINS_HERB"
    HAS_PATHWAY_STAGE = "HAS_PATHWAY_STAGE"
    NEXT_STAGE = "NEXT_STAGE"
    REGISTERED_IN_PHARMACOPOEIA = "REGISTERED_IN_PHARMACOPOEIA"
    CITES_DOCUMENT = "CITES_DOCUMENT"
    CONTRAINDICATED_FOR = "CONTRAINDICATED_FOR"
    CAUTION_FOR = "CAUTION_FOR"
    ALIAS_OF = "ALIAS_OF"
    SAME_AS = "SAME_AS"
    DERIVED_FROM = "DERIVED_FROM"
    PROCESSED_FROM = "PROCESSED_FROM"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: ``(source type, edge type, target type)`` signatures observed in the graph.
#: Used by :func:`validate_graph` to detect corruption after any re-export.
EDGE_SIGNATURES: FrozenSet[Tuple[str, str, str]] = frozenset(
    {
        ("Formula", "CONTAINS_HERB", "Herb"),
        ("Syndrome", "USES_EXTERNAL_THERAPY", "ExternalTherapy"),
        ("Disease", "HAS_PATHWAY_STAGE", "PathwayStage"),
        ("Syndrome", "TREATED_BY_PRINCIPLE", "TreatmentPrinciple"),
        ("Disease", "HAS_SYNDROME", "Syndrome"),
        ("Syndrome", "USES_PATENT_MEDICINE", "PatentMedicine"),
        ("Syndrome", "USES_FORMULA", "Formula"),
        ("PathwayStage", "NEXT_STAGE", "PathwayStage"),
        ("Herb", "REGISTERED_IN_PHARMACOPOEIA", "PharmacoPoeiaEntry"),
        ("Disease", "CITES_DOCUMENT", "DocumentSource"),
        ("Disease", "HAS_SUBTYPE", "DiseaseSubtype"),
        ("Disease", "BELONGS_TO_DEPARTMENT", "Department"),
        ("Herb", "ALIAS_OF", "Herb"),
        ("DiseaseSubtype", "SUBTYPE_HAS_SYNDROME", "Syndrome"),
        ("RestrictedItem", "CONTRAINDICATED_FOR", "SafetyContext"),
        ("RestrictedItem", "CONTRAINDICATED_FOR", "Syndrome"),
        ("RestrictedItem", "CAUTION_FOR", "SafetyContext"),
        ("RestrictedItem", "CAUTION_FOR", "Syndrome"),
        ("Herb", "CONTRAINDICATED_FOR", "SafetyContext"),
        ("Syndrome", "USES_HERB_DIRECT", "Herb"),
        ("Herb", "SAME_AS", "Herb"),
        ("Formula", "SAME_AS", "Formula"),
        ("Herb", "DERIVED_FROM", "Herb"),
        ("Herb", "PROCESSED_FROM", "Herb"),
    }
)

#: Edge types whose direction carries no clinical meaning; traversal may cross
#: them either way when resolving an entity to its canonical form.
SYMMETRIC_EDGES: FrozenSet[str] = frozenset(
    {EdgeType.SAME_AS.value, EdgeType.ALIAS_OF.value}
)

#: Provenance is carried by the ``source_docs`` attribute (a list of ``doc_id``
#: strings) on nodes and edges, *not* by ``CITES_DOCUMENT`` edges, which only
#: connect Disease -> DocumentSource.  Every retrieval result therefore resolves
#: provenance through this attribute.
PROVENANCE_ATTR = "source_docs"


class Domain(str, Enum):
    """Tool-visible slice of the graph.

    ``CLINICAL`` is the syndrome-differentiation domain; ``SAFETY`` is the
    prescription-audit domain.  ``FULL`` exists only for ablations and for the
    coverage audit -- it must never be used for a headline experiment.
    """

    CLINICAL = "clinical_reasoning"
    SAFETY = "prescription_safety"
    FULL = "full_graph"


@dataclass(frozen=True)
class DomainPolicy:
    """Node types a domain may return, and the reason the rest are withheld."""

    domain: Domain
    allowed_nodes: FrozenSet[str]
    #: Types reachable for *verification only* -- a tool may use them to test a
    #: candidate the model already produced, but never to enumerate candidates.
    verification_only_nodes: FrozenSet[str] = frozenset()
    rationale: str = ""

    def may_return(self, node_type: str, *, verification: bool = False) -> bool:
        if node_type in self.allowed_nodes:
            return True
        return verification and node_type in self.verification_only_nodes

    def visible_types(self, *, verification: bool = False) -> FrozenSet[str]:
        if verification:
            return self.allowed_nodes | self.verification_only_nodes
        return self.allowed_nodes


_CLINICAL_NODES = frozenset(
    {
        NodeType.DEPARTMENT.value,
        NodeType.DISEASE.value,
        NodeType.DISEASE_SUBTYPE.value,
        NodeType.SYNDROME.value,
        NodeType.PATHWAY_STAGE.value,
        NodeType.DOCUMENT_SOURCE.value,
    }
)

_SAFETY_NODES = frozenset(
    {
        NodeType.DISEASE.value,
        NodeType.SYNDROME.value,
        NodeType.TREATMENT_PRINCIPLE.value,
        NodeType.FORMULA.value,
        NodeType.HERB.value,
        NodeType.PATENT_MEDICINE.value,
        NodeType.EXTERNAL_THERAPY.value,
        NodeType.SAFETY_CONTEXT.value,
        NodeType.RESTRICTED_ITEM.value,
        NodeType.PHARMACOPOEIA_ENTRY.value,
        NodeType.DOCUMENT_SOURCE.value,
    }
)

DOMAIN_POLICIES: Mapping[Domain, DomainPolicy] = {
    Domain.CLINICAL: DomainPolicy(
        domain=Domain.CLINICAL,
        allowed_nodes=_CLINICAL_NODES,
        verification_only_nodes=frozenset({NodeType.TREATMENT_PRINCIPLE.value}),
        rationale=(
            "SDT asks the model to name a syndrome. Formula / Herb / "
            "PatentMedicine / ExternalTherapy would let it invert the "
            "syndrome->treatment mapping and recover the answer from the "
            "prescription instead of from the clinical picture, which is not "
            "the ability SDT means to measure. TreatmentPrinciple is exposed "
            "for consistency checking only, never for candidate generation."
        ),
    ),
    Domain.SAFETY: DomainPolicy(
        domain=Domain.SAFETY,
        allowed_nodes=_SAFETY_NODES,
        rationale=(
            "PA asks about drug knowledge, safety constraints and pharmacopoeia "
            "regulation, so the medication and safety sub-graphs are the object "
            "of study and are fully exposed. PathwayStage / Department are "
            "withheld: they carry inpatient workflow detail irrelevant to "
            "prescription audit and only dilute retrieval."
        ),
    ),
    Domain.FULL: DomainPolicy(
        domain=Domain.FULL,
        allowed_nodes=frozenset(t.value for t in NodeType),
        rationale="Ablation / audit only. Not a headline condition.",
    ),
}


class DomainViolation(RuntimeError):
    """Raised when a tool would emit a node type its domain forbids."""


def policy_for(domain: Domain | str) -> DomainPolicy:
    if isinstance(domain, str):
        domain = Domain(domain)
    return DOMAIN_POLICIES[domain]


# --------------------------------------------------------------------------- #
# Node-type field maps used by the retrieval index and the tool serialisers.
# --------------------------------------------------------------------------- #

#: Free-text attributes that are worth indexing for semantic anchoring, per type.
#: ``first_mention.sentence`` is handled separately because it is nested and is
#: the single richest signal in the graph: for 312 of 648 Syndrome nodes it is
#: the verbatim symptom/tongue/pulse sentence from the source protocol.
INDEXED_TEXT_FIELDS: Mapping[str, Sequence[str]] = {
    NodeType.DISEASE.value: ("name", "full_name", "tcm_name", "western_name"),
    NodeType.DISEASE_SUBTYPE.value: ("name", "disease"),
    NodeType.SYNDROME.value: ("name",),
    NodeType.TREATMENT_PRINCIPLE.value: ("name",),
    NodeType.DEPARTMENT.value: ("name",),
    NodeType.FORMULA.value: ("name",),
    NodeType.HERB.value: ("name",),
    NodeType.PATENT_MEDICINE.value: ("name",),
    NodeType.EXTERNAL_THERAPY.value: ("name",),
    NodeType.SAFETY_CONTEXT.value: ("name", "disease"),
    NodeType.RESTRICTED_ITEM.value: ("name",),
    NodeType.PHARMACOPOEIA_ENTRY.value: (
        "name",
        "part_used",
        "nature_taste_meridian",
        "pharmacopoeial_functions",
    ),
    NodeType.DOCUMENT_SOURCE.value: (
        "title",
        "doc_type",
        "department",
        "version_label",
        "institution_attribution",
    ),
    NodeType.PATHWAY_STAGE.value: (
        "name",
        "disease",
        "variant",
        "entry_criteria",
        "exit_criteria",
        "monitoring_items",
        "outcome_indicators",
        "day_actions",
        "nursing_items",
    ),
}


def validate_graph(
    node_types: Mapping[str, str], edges: Iterable[Mapping[str, object]]
) -> Dict[str, Set[str]]:
    """Check a loaded graph against the declared ontology.

    Returns a mapping of problem kind -> offending descriptions.  An empty
    mapping means the graph matches the ontology exactly.
    """
    problems: Dict[str, Set[str]] = {}
    known_nodes = {t.value for t in NodeType}
    known_edges = {t.value for t in EdgeType}

    for node_id, node_type in node_types.items():
        if node_type not in known_nodes:
            problems.setdefault("unknown_node_type", set()).add(
                f"{node_id} :: {node_type}"
            )

    for edge in edges:
        etype = str(edge.get("type"))
        src = str(edge.get("from"))
        dst = str(edge.get("to"))
        if etype not in known_edges:
            problems.setdefault("unknown_edge_type", set()).add(etype)
            continue
        if src not in node_types:
            problems.setdefault("dangling_source", set()).add(f"{etype}:{src}")
            continue
        if dst not in node_types:
            problems.setdefault("dangling_target", set()).add(f"{etype}:{dst}")
            continue
        signature = (node_types[src], etype, node_types[dst])
        if signature not in EDGE_SIGNATURES:
            problems.setdefault("unknown_signature", set()).add(str(signature))

    return problems
