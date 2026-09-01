"""The knowledge-graph figure must stay in sync with the schema and the graph.

A figure is the part of a paper a reader trusts without checking, so the two
ways it can quietly go wrong both get a test here: drawing a schema that no
longer matches ``EDGE_SIGNATURES``, and shipping an SVG generated from an older
graph than the one in ``kg/``.  Both are silent failures otherwise -- the
figure still renders, it is just no longer true.
"""

import importlib.util
import unittest
from pathlib import Path

from tcm_kg.schema import EDGE_SIGNATURES, NodeType

REPO = Path(__file__).resolve().parent.parent
SVG = REPO / "docs" / "figures" / "kg_schema.svg"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "figure_kg", REPO / "scripts" / "figure_kg.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FIG = _load_generator()


class TestFigureMatchesSchema(unittest.TestCase):
    def test_every_relation_is_drawn(self):
        FIG.check_coverage()          # raises SystemExit naming what is missing

    def test_the_guard_actually_fires(self):
        """A coverage check that cannot fail would pass forever after a change."""
        original = FIG.EDGE_SIGNATURES
        try:
            FIG.EDGE_SIGNATURES = list(original) + [
                ("Herb", "INVENTED_RELATION", "Department")
            ]
            with self.assertRaises(SystemExit):
                FIG.check_coverage()
        finally:
            FIG.EDGE_SIGNATURES = original

    def test_every_entity_type_has_a_place(self):
        placed = set(FIG.PLACEMENT)
        declared = {str(t) for t in NodeType}
        self.assertEqual(declared - placed, set(), "entity types missing from the figure")
        self.assertEqual(placed - declared, set(), "figure draws types the schema dropped")

    def test_relation_colour_is_defined_for_every_relation(self):
        for _src, rel, _dst in EDGE_SIGNATURES:
            self.assertIn(FIG.relation_group(rel), FIG.GROUP_COLOR)


class TestCommittedFigureIsCurrent(unittest.TestCase):
    """The checked-in SVG must be what today's graph and schema produce."""

    def test_svg_exists(self):
        self.assertTrue(SVG.exists(), f"{SVG} is missing; run scripts/figure_kg.py")

    def test_svg_is_not_stale(self):
        counts, edges, n_nodes, n_edges = FIG.collect()
        regenerated = FIG.draw(counts, edges, n_nodes, n_edges)
        self.assertEqual(
            regenerated,
            SVG.read_text(encoding="utf-8"),
            "docs/figures/kg_schema.svg is out of date with the graph or the "
            "generator; re-run scripts/figure_kg.py and commit the result.",
        )

    def test_headline_counts_appear_in_the_svg(self):
        _counts, _edges, n_nodes, n_edges = FIG.collect()
        svg = SVG.read_text(encoding="utf-8")
        self.assertIn(f"{n_nodes:,} entities", svg)
        self.assertIn(f"{n_edges:,} relations", svg)


if __name__ == "__main__":
    unittest.main()
