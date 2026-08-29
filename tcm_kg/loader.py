"""Loading the knowledge graph from its delivered artefacts.

Two formats are supported and must produce identical stores:

* ``tcm_knowledge_graph.json`` (optionally gzipped) -- node-link JSON with
  ``nodes`` and ``edges`` arrays.  This is the canonical artefact committed to
  the repository.
* ``tcm_knowledge_graph.graphml`` -- the same graph with every attribute
  flattened to a string.  JSON-valued attributes are re-parsed on load so both
  paths yield the same Python objects.
"""

from __future__ import annotations

import gzip
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schema import validate_graph
from .store import KGStore

_GRAPHML_NS = "{http://graphml.graphdrawing.org/xmlns}"

#: Attributes that hold JSON in the GraphML export and must be re-parsed.
_JSON_ATTRS = frozenset(
    {
        "source_docs",
        "first_mention",
        "pharmacopoeia_entry",
        "evidence",
        "common_aliases",
        "entry_criteria",
        "exit_criteria",
        "monitoring_items",
        "outcome_indicators",
        "day_actions",
        "nursing_items",
    }
)

DEFAULT_KG_PATH = Path(__file__).resolve().parent.parent / "kg" / "tcm_knowledge_graph.json.gz"


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_node_link(path: Path) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    with _open_maybe_gzip(path) as handle:
        payload = json.load(handle)
    nodes = payload.get("nodes") or []
    edges = payload.get("edges")
    if edges is None:  # networkx writes "links" when using node_link_data defaults
        edges = payload.get("links") or []
    normalised: List[Mapping[str, Any]] = []
    for edge in edges:
        item = dict(edge)
        item.setdefault("from", item.pop("source", None))
        item.setdefault("to", item.pop("target", None))
        normalised.append(item)
    return nodes, normalised


def _coerce(key: str, value: Optional[str]) -> Any:
    if value is None:
        return None
    if key in _JSON_ATTRS:
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def load_graphml(path: Path) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    tree = ET.parse(str(path))
    root = tree.getroot()
    keys: Dict[str, Tuple[str, str]] = {}
    for key in root.findall(f"{_GRAPHML_NS}key"):
        keys[key.attrib["id"]] = (key.attrib.get("attr.name", ""), key.attrib.get("attr.type", "string"))

    def read_data(element: ET.Element) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for data in element.findall(f"{_GRAPHML_NS}data"):
            name, dtype = keys.get(data.attrib.get("key", ""), ("", "string"))
            if not name:
                continue
            value: Any = _coerce(name, data.text)
            if dtype in {"long", "int"} and isinstance(value, str) and value.strip():
                try:
                    value = int(value)
                except ValueError:
                    pass
            out[name] = value
        return out

    graph = root.find(f"{_GRAPHML_NS}graph")
    nodes: List[Mapping[str, Any]] = []
    edges: List[Mapping[str, Any]] = []
    if graph is None:
        return nodes, edges
    for element in graph.findall(f"{_GRAPHML_NS}node"):
        payload = read_data(element)
        payload["id"] = element.attrib["id"]
        nodes.append(payload)
    for element in graph.findall(f"{_GRAPHML_NS}edge"):
        payload = read_data(element)
        payload["from"] = element.attrib["source"]
        payload["to"] = element.attrib["target"]
        payload.setdefault("_id", element.attrib.get("id", ""))
        edges.append(payload)
    return nodes, edges


def load_kg(path: str | os.PathLike[str] | None = None, *, validate: bool = False) -> KGStore:
    """Load the graph into a :class:`KGStore`.

    ``validate=True`` cross-checks the loaded graph against the declared
    ontology and raises on any mismatch; the CLI's ``inspect`` command uses it
    to prove that a re-exported graph still matches the published schema.
    """
    target = Path(path) if path else DEFAULT_KG_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"knowledge graph not found at {target}; pass --kg or place the "
            f"artefact at {DEFAULT_KG_PATH}"
        )
    if target.name.endswith(".graphml"):
        nodes, edges = load_graphml(target)
    else:
        nodes, edges = load_node_link(target)

    if validate:
        problems = validate_graph({str(n.get("id")): str(n.get("type")) for n in nodes}, edges)
        if problems:
            detail = "; ".join(f"{k}={sorted(v)[:3]}" for k, v in problems.items())
            raise ValueError(f"graph does not match declared ontology: {detail}")
    return KGStore(nodes, edges)
