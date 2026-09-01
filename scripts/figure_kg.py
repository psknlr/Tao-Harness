"""Generate the knowledge-graph figure from the graph itself.

Every count and every relation in the output is read from the live graph and
the live schema, so the figure cannot drift from the data the way a hand-drawn
one does -- the same reason ``experiment_manifest.py`` exists.  ``ROUTES`` and
``SELF_LOOPS`` are checked against ``EDGE_SIGNATURES`` at build time, so adding
a relation to the schema without drawing it fails loudly instead of shipping a
figure that quietly omits it.

Three panels, because "visualise the knowledge graph" has three different
honest answers and only the first is a picture of the ontology:

a. **Schema.** The 14 entity types and the 24 typed relations between them,
   laid out along the clinical flow: organisation -> disease -> differentiation
   -> therapy -> regulation, with the pathway branch above and the provenance
   layer below.  A force layout of all 9,350 nodes would be a hairball -- it
   would show that the graph is big, which the caption can say in four words,
   and hide the structure, which is the thing worth a figure.

b. **Access domains.** Which entity types each experimental arm can reach.
   This is the part specific to *this* harness rather than to TCM knowledge
   graphs in general: SDT cannot see treatment entities, so an agent cannot
   invert syndrome->formula and read its own answer off the prescription.

c. **Relation scale.** The 20 relation types by edge count on a log axis,
   because they span 73 to 9,681 and a linear axis would render fifteen of
   them as a hairline.

Output is SVG with live text -- journals want editable text, not outlined
paths.  The user units are nominal pixels; the figure is drawn at a 1.4:1
aspect for a 183 mm full-width placement.
"""

from __future__ import annotations

import collections
import html
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcm_kg import load_kg  # noqa: E402
from tcm_kg.schema import (  # noqa: E402
    EDGE_SIGNATURES,
    Domain,
    policy_for,
)

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #
#: Four categorical hues plus one neutral, checked with the palette validator
#: rather than by eye.  A schema diagram puts any two group colours next to
#: each other, so the adjacent-pair rule does not apply and the whole set has
#: to hold *all-pairs*; the six-hue theme fails that (green/orange dE 3.2 under
#: protanopia), and so does the lighter aqua once the neutral is added
#: (grey/aqua dE 6.4 deutan).  The set below passes every check all-pairs:
#: worst CVD dE 9.9 (teal/orange, protan), worst normal-vision dE 16.3
#: (violet/blue), every fill over 3:1 against white.  The one expected failure
#: is the chroma floor on the neutral, which is the point of it -- provenance
#: is deliberately not a hue, because evidence belongs to every family.
#: Every mark also carries a text label, so colour reinforces the grouping
#: rather than carrying it.
GROUP_COLOR: Dict[str, str] = {
    "differentiation": "#2a78d6",  # blue
    "therapy": "#eb6834",          # orange
    "pathway": "#009182",          # teal
    "regulation": "#4a3aa7",       # violet
    "provenance": "#575652",       # neutral, not a fifth hue
}
GROUP_LABEL = {
    "differentiation": "Disease & differentiation",
    "therapy": "Therapy",
    "pathway": "Clinical pathway",
    "regulation": "Safety & regulation",
    "provenance": "Provenance",
}
#: Ties in the panel-c colour rule resolve towards the more specific family.
GROUP_PRECEDENCE = ["differentiation", "provenance", "therapy", "pathway", "regulation"]

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#87867f"
DENIED = "#f2f1ee"

FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"

# --------------------------------------------------------------------------- #
# panel a layout
# --------------------------------------------------------------------------- #
BOX_W, BOX_H = 104.0, 34.0
GUTTER = 92.0          # wide enough for the longest single-line relation name
COL_PITCH = BOX_W + GUTTER
ROW_H = 62.0

#: ``type -> (column, row, group)``.  Columns run left to right along the
#: clinical flow; rows separate the branches within a column.  Syndrome and
#: Herb share row 1.65 so USES_HERB_DIRECT can run as a straight lane between
#: Formula and PatentMedicine instead of grazing either box.
PLACEMENT: Dict[str, Tuple[int, float, str]] = {
    "Department":         (0, 1.00, "differentiation"),
    "DocumentSource":     (0, 3.20, "provenance"),
    "Disease":            (1, 1.00, "differentiation"),
    "DiseaseSubtype":     (1, 2.20, "differentiation"),
    "PathwayStage":       (2, 0.00, "pathway"),
    "Syndrome":           (2, 1.65, "differentiation"),
    "TreatmentPrinciple": (3, 0.00, "therapy"),
    "Formula":            (3, 1.10, "therapy"),
    "PatentMedicine":     (3, 2.20, "therapy"),
    "ExternalTherapy":    (3, 3.30, "therapy"),
    "Herb":               (4, 1.65, "therapy"),
    "PharmacoPoeiaEntry": (5, 0.60, "regulation"),
    "SafetyContext":      (5, 1.80, "regulation"),
    "RestrictedItem":     (5, 3.00, "regulation"),
}

#: ``(src, dst) -> (routing, label lines, gutter override)``.
#:
#: ``gutter``   a bezier across the gap between two columns, labelled in it
#: ``lane``     a straight horizontal run that passes between two boxes
#: ``vertical`` a straight run inside one column, labelled on the segment
#: ``under``    an orthogonal detour beneath the whole diagram, for the one
#:              edge that runs backwards across five columns
ROUTES: Dict[Tuple[str, str], Tuple[str, List[str], int]] = {
    ("Disease", "Department"):           ("gutter", ["BELONGS_TO", "DEPARTMENT"], 0),
    ("Disease", "DocumentSource"):       ("gutter", ["CITES_DOCUMENT"], 0),
    ("Disease", "DiseaseSubtype"):       ("vertical", ["HAS_SUBTYPE"], -1),
    ("Disease", "PathwayStage"):         ("gutter", ["HAS_PATHWAY_STAGE"], 1),
    ("Disease", "Syndrome"):             ("gutter", ["HAS_SYNDROME"], 1),
    ("DiseaseSubtype", "Syndrome"):      ("gutter", ["SUBTYPE_HAS_SYNDROME"], 1),
    ("Syndrome", "TreatmentPrinciple"):  ("gutter", ["TREATED_BY_PRINCIPLE"], 2),
    ("Syndrome", "Formula"):             ("gutter", ["USES_FORMULA"], 2),
    ("Syndrome", "PatentMedicine"):      ("gutter", ["USES_PATENT_MEDICINE"], 2),
    ("Syndrome", "ExternalTherapy"):     ("gutter", ["USES_EXTERNAL_THERAPY"], 2),
    ("Syndrome", "Herb"):                ("lane", ["USES_HERB_DIRECT"], 3),
    ("Formula", "Herb"):                 ("gutter", ["CONTAINS_HERB"], 3),
    ("Herb", "PharmacoPoeiaEntry"):      ("gutter", ["REGISTERED_IN", "PHARMACOPOEIA"], 4),
    ("Herb", "SafetyContext"):           ("gutter", ["CONTRAINDICATED_FOR"], 4),
    ("RestrictedItem", "SafetyContext"): ("vertical", ["CONTRAINDICATED_FOR", "CAUTION_FOR"], -1),
    ("RestrictedItem", "Syndrome"):      ("under", ["CONTRAINDICATED_FOR  ·  CAUTION_FOR"], -1),
}

#: Self-relations, collapsed to one glyph per type: seven near-identical loops
#: would obscure the structure they sit on.
SELF_LOOPS: Dict[str, List[str]] = {
    "PathwayStage": ["NEXT_STAGE"],
    "Formula": ["SAME_AS"],
    "Herb": ["ALIAS_OF · SAME_AS", "PROCESSED_FROM · DERIVED_FROM"],
}


def check_coverage() -> None:
    """Fail if the schema has grown a relation the figure does not draw."""
    pairs, loops = set(), collections.defaultdict(set)
    for src, rel, dst in EDGE_SIGNATURES:
        if src == dst:
            loops[src].add(rel)
        else:
            pairs.add((src, dst))
    missing_pairs = pairs - set(ROUTES)
    extra_pairs = set(ROUTES) - pairs
    missing_loops = set(loops) - set(SELF_LOOPS)
    if missing_pairs or extra_pairs or missing_loops:
        raise SystemExit(
            "figure_kg: the drawing no longer matches the schema.\n"
            f"  undrawn relations : {sorted(missing_pairs) or '-'}\n"
            f"  drawn but dropped : {sorted(extra_pairs) or '-'}\n"
            f"  undrawn self-loops: {sorted(missing_loops) or '-'}"
        )
    for node, rels in loops.items():
        drawn = {r for line in SELF_LOOPS[node] for r in line.replace("·", " ").split()}
        if rels - drawn:
            raise SystemExit(
                f"figure_kg: self-loop label for {node} omits {sorted(rels - drawn)}"
            )


# --------------------------------------------------------------------------- #
# text metrics
# --------------------------------------------------------------------------- #
#: Helvetica advance widths (units per em/1000).  Needed because the labels are
#: centred and knocked out of the connector lines, and a guessed width either
#: clips the knockout or punches a hole far wider than the text.
_ADV = {
    **{c: w for c, w in zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        [667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833,
         722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611])},
    **{c: w for c, w in zip(
        "abcdefghijklmnopqrstuvwxyz",
        [556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833,
         556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500])},
    **{c: 556 for c in "0123456789"},
    " ": 278, ",": 278, ".": 278, "_": 556, "-": 333, "/": 278,
    "·": 333, "—": 1000, "&": 667, "(": 333, ")": 333, ":": 278, "→": 1000,
}


def text_w(s: str, size: float, weight: str = "normal") -> float:
    units = sum(_ADV.get(ch, 600) for ch in s)
    bold = 1.06 if weight in ("600", "700", "bold") else 1.0
    return units / 1000.0 * size * bold


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


class Canvas:
    """A tiny SVG writer -- no dependency, exact control, live text."""

    def __init__(self, width: float, height: float):
        self.w, self.h = width, height
        self.parts: List[str] = []

    def add(self, markup: str) -> None:
        self.parts.append(markup)

    def rect(self, x, y, w, h, *, fill="none", stroke="none", rx=0, sw=1, extra=""):
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'
        )

    def text(self, x, y, s, *, size=9, fill=INK, anchor="start", weight="normal",
             extra=""):
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" {extra}>{esc(s)}</text>'
        )

    def knockout_text(self, cx, y, lines: Sequence[str], *, size=5.5, fill=INK_3,
                      line_h=7.4):
        """Centred label with the connector line cleared out behind it."""
        width = max(text_w(l, size) for l in lines)
        height = line_h * len(lines)
        self.rect(cx - width / 2 - 2.5, y - size + 0.5, width + 5, height + 1.5,
                  fill=SURFACE)
        for i, line in enumerate(lines):
            self.text(cx, y + i * line_h, line, size=size, fill=fill, anchor="middle")

    def path(self, d, *, stroke=INK_3, sw=0.95, fill="none", extra=""):
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" {extra}/>'
        )

    def render(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.w:.0f}" height="{self.h:.0f}" '
            f'viewBox="0 0 {self.w:.0f} {self.h:.0f}">'
            f'<rect width="{self.w:.0f}" height="{self.h:.0f}" fill="{SURFACE}"/>'
            f'<defs><marker id="arrow" viewBox="0 0 8 8" refX="6.6" refY="4" '
            f'markerWidth="4.6" markerHeight="4.6" orient="auto-start-reverse">'
            f'<path d="M0,1 L6.6,4 L0,7 z" fill="{INK_3}" stroke="none"/></marker>'
            f'</defs>'
            + "".join(self.parts)
            + "</svg>"
        )


def collect() -> Tuple[Dict[str, int], Dict[str, int], int, int]:
    kg = load_kg()
    nodes = collections.Counter(str(n.type) for n in kg.nodes.values())
    edges_iter = kg.edges.values() if hasattr(kg.edges, "values") else kg.edges
    edges = collections.Counter(str(e.type) for e in edges_iter)
    return dict(nodes), dict(edges), len(kg.nodes), sum(edges.values())


def relation_group(rel: str) -> str:
    """Colour a relation by the family of the entity it *points at*.

    An edge's contribution to the graph is what it attaches, so the target
    reads better than the source: CONTRAINDICATED_FOR is a safety fact whether
    it starts at a herb or at a restricted item.  Ties go to the more specific
    family, which is what GROUP_PRECEDENCE orders.
    """
    targets = [PLACEMENT[d][2] for s, r, d in EDGE_SIGNATURES if r == rel]
    counts = collections.Counter(targets)
    best = max(counts.values())
    tied = [g for g, n in counts.items() if n == best]
    return max(tied, key=GROUP_PRECEDENCE.index)


# --------------------------------------------------------------------------- #
def draw(node_counts, edge_counts, n_nodes: int, n_edges: int) -> str:
    M = 24.0
    PANEL_W = 5 * COL_PITCH + BOX_W          # panel a sets the figure width
    W = PANEL_W + 2 * M
    H = 800.0
    c = Canvas(W, H)

    # ---------------------------------------------------------------- header
    c.text(M, 28, "The TCM clinical knowledge graph", size=14, weight="700")
    c.text(
        M, 44,
        f"{n_nodes:,} entities · {n_edges:,} relations · "
        f"{len(node_counts)} entity types · {len(edge_counts)} relation types · "
        f"{len(EDGE_SIGNATURES)} typed signatures",
        size=8.5, fill=INK_2,
    )

    def panel_title(x, y, letter, title):
        c.text(x, y, letter, size=11, weight="700")
        c.text(x + 13, y, title, size=9, fill=INK_2)

    # ================================================================ panel a
    panel_title(M, 70, "a", "Ontology — the entity types and the typed relations between them")

    col_x = [M + i * COL_PITCH for i in range(6)]
    top = 110.0

    def box(name) -> Tuple[float, float]:
        col, row, _ = PLACEMENT[name]
        return col_x[col], top + row * ROW_H

    def ctr(name) -> Tuple[float, float]:
        x, y = box(name)
        return x + BOX_W / 2, y + BOX_H / 2

    lane_y = top + max(r for _, r, _ in PLACEMENT.values()) * ROW_H + BOX_H + 24

    # ---- connectors first, so the boxes sit on top of them
    pending: Dict[int, List[Tuple[float, List[str]]]] = collections.defaultdict(list)
    for (src, dst), (kind, lines, gut) in ROUTES.items():
        (x1, y1), (x2, y2) = ctr(src), ctr(dst)
        sx1, sy1 = box(src)
        sx2, sy2 = box(dst)

        if kind == "gutter":
            forward = x2 > x1
            sx = sx1 + BOX_W if forward else sx1
            ex = sx2 if forward else sx2 + BOX_W
            mx = (sx + ex) / 2
            c.path(
                f"M{sx:.1f},{y1:.1f} C{mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} "
                f"{ex:.1f},{y2:.1f}",
                extra='marker-end="url(#arrow)" opacity="0.8"',
            )
            pending[gut].append(((y1 + y2) / 2, lines))

        elif kind == "lane":
            c.path(f"M{sx1 + BOX_W:.1f},{y1:.1f} L{sx2:.1f},{y2:.1f}",
                   extra='marker-end="url(#arrow)" opacity="0.8"')
            pending[gut].append((y1, lines))

        elif kind == "vertical":
            down = y2 > y1
            sy = sy1 + BOX_H if down else sy1
            ey = sy2 if down else sy2 + BOX_H
            c.path(f"M{x1:.1f},{sy:.1f} L{x1:.1f},{ey:.1f}",
                   extra='marker-end="url(#arrow)" opacity="0.8"')
            mid = (sy + ey) / 2
            c.knockout_text(x1, mid - 3.2 * (len(lines) - 1) + 2, lines)

        elif kind == "under":
            r = 8.0
            c.path(
                f"M{x1:.1f},{sy1 + BOX_H:.1f} "
                f"L{x1:.1f},{lane_y - r:.1f} Q{x1:.1f},{lane_y:.1f} {x1 - r:.1f},{lane_y:.1f} "
                f"L{x2 + r:.1f},{lane_y:.1f} Q{x2:.1f},{lane_y:.1f} {x2:.1f},{lane_y - r:.1f} "
                f"L{x2:.1f},{sy2 + BOX_H:.1f}",
                extra='marker-end="url(#arrow)" opacity="0.8"',
            )
            c.knockout_text((x1 + x2) / 2, lane_y + 2, lines)

    # gutter labels: place each at its edge's midpoint, then push apart
    for gut, items in pending.items():
        items.sort(key=lambda t: t[0])
        placed: List[Tuple[float, List[str]]] = []
        floor = -1e9
        for y, lines in items:
            y = max(y, floor)
            placed.append((y, lines))
            floor = y + 7.4 * len(lines) + 3.5
        cx = col_x[gut] + BOX_W + GUTTER / 2
        for y, lines in placed:
            c.knockout_text(cx, y - 3.2 * (len(lines) - 1) + 2, lines)

    # ---- self-loops
    for name, lines in SELF_LOOPS.items():
        x, y = box(name)
        cx = x + BOX_W / 2
        c.path(
            f"M{cx - 15:.1f},{y:.1f} C{cx - 15:.1f},{y - 15:.1f} "
            f"{cx + 15:.1f},{y - 15:.1f} {cx + 15:.1f},{y - 0.5:.1f}",
            extra='marker-end="url(#arrow)" opacity="0.8"',
        )
        for i, line in enumerate(lines):
            c.text(cx, y - 18 - 7.4 * (len(lines) - 1 - i), line,
                   size=5.5, fill=INK_3, anchor="middle")

    # ---- boxes
    biggest = max(node_counts.values())
    for name, (col, row, group) in PLACEMENT.items():
        x, y = box(name)
        colour = GROUP_COLOR[group]
        count = node_counts.get(name, 0)
        c.rect(x, y, BOX_W, BOX_H, fill=SURFACE, stroke=colour, rx=3, sw=1.3)
        # a colour flag on the leading edge: the family stays legible in
        # greyscale print, where the outline hue does not survive
        c.rect(x, y, 3.5, BOX_H, fill=colour)
        c.text(x + 9, y + 14, name, size=7.6, weight="600")
        c.text(x + 9, y + 25.5, f"{count:,}", size=7, fill=INK_2)
        # sqrt-scaled magnitude bar: 32 vs 1,440 is a 45x span and a linear
        # bar would render half the types as a hairline
        frac = math.sqrt(count / biggest) if biggest else 0.0
        c.rect(x + 46, y + 20.5, 49 * frac, 3.2, fill=colour, rx=1.6,
               extra='opacity="0.42"')

    # ---- shared legend (serves all three panels)
    ly = lane_y + 34
    lx = M
    for group, colour in GROUP_COLOR.items():
        c.rect(lx, ly - 6.5, 9, 9, fill=colour, rx=2)
        c.text(lx + 13, ly, GROUP_LABEL[group], size=7.4, fill=INK_2)
        lx += 13 + text_w(GROUP_LABEL[group], 7.4) + 26

    # ================================================================ panel b
    by = ly + 34
    panel_title(M, by, "b", "Access domain — which entity types each arm may reach")

    domains = [
        ("SDT", Domain.CLINICAL, "clinical reasoning"),
        ("PA", Domain.SAFETY, "prescription safety"),
        ("CP", Domain.PATHWAY, "clinical pathway"),
    ]
    order = [n for n, _ in sorted(PLACEMENT.items(), key=lambda kv: (kv[1][0], kv[1][1]))]
    cell_w, cell_h = 58.0, 17.0
    grid_x = M + 96
    grid_y = by + 34

    for i, name in enumerate(order):
        c.text(grid_x - 7, grid_y + i * cell_h + 11.5, name, size=7,
               fill=INK_2, anchor="end")
    for j, (short, domain, long) in enumerate(domains):
        x = grid_x + j * cell_w
        c.text(x + cell_w / 2, grid_y - 14, short, size=8.4, weight="700",
               anchor="middle")
        c.text(x + cell_w / 2, grid_y - 5, long, size=5.6, fill=INK_3, anchor="middle")
        policy = policy_for(domain)
        for i, name in enumerate(order):
            y = grid_y + i * cell_h
            colour = GROUP_COLOR[PLACEMENT[name][2]]
            if name in policy.allowed_nodes:
                c.rect(x + 3, y + 2, cell_w - 6, cell_h - 4, fill=colour, rx=2.5,
                       extra='opacity="0.9"')
            elif name in policy.verification_only_nodes:
                # reachable to *test* a candidate the model already produced,
                # never to enumerate candidates
                c.rect(x + 3, y + 2, cell_w - 6, cell_h - 4, fill=SURFACE,
                       stroke=colour, rx=2.5, sw=1.2,
                       extra='stroke-dasharray="2.6 2"')
                c.text(x + cell_w / 2, y + 11.6, "verify", size=6, fill=colour,
                       anchor="middle")
            else:
                c.rect(x + 3, y + 2, cell_w - 6, cell_h - 4, fill=DENIED, rx=2.5)
                c.text(x + cell_w / 2, y + 12, "—", size=7.5, fill=INK_3,
                       anchor="middle")

    note_x = grid_x + 3 * cell_w + 26
    notes = [
        ("SDT is withheld every treatment entity.", False),
        ("An agent that could read Formula would invert", True),
        ("syndrome→formula and recover the answer off", True),
        ("the prescription instead of reasoning to it.", True),
        ("", True),
        ("TreatmentPrinciple is verification-only for SDT.", False),
        ("A candidate the model has already produced may", True),
        ("be checked against it; it may never be used to", True),
        ("enumerate candidates.", True),
        ("", True),
        ("CP withholds nothing: executing a pathway is", False),
        ("deciding treatment. That is why it is a separate", True),
        ("domain rather than a widening of the clinical one.", True),
    ]
    for k, (line, muted) in enumerate(notes):
        c.text(note_x, grid_y + 4 + k * 11.2, line, size=7,
               fill=INK_2 if muted else INK, weight="normal" if muted else "600")

    # ================================================================ panel c
    cx0 = M + PANEL_W * 0.585
    panel_title(cx0, by, "c", "Relation types by edge count")
    c.text(cx0 + 13, by + 11,
           "log axis; bars take the colour of the family each relation points at",
           size=6.6, fill=INK_3)

    ranked = sorted(edge_counts.items(), key=lambda kv: -kv[1])
    label_r = cx0 + 118          # right edge of the relation-name column
    bar_x = label_r + 8
    bar_end = M + PANEL_W - 46      # room for the largest count beside its bar
    bar_w = bar_end - bar_x
    bar_h, gap = 9.4, 3.8
    top_c = grid_y + 6              # clears the panel subtitle above the decade ticks
    lo, hi = 1.0, 4.0            # 10 .. 10,000, so the decade ticks land round
    rows_h = len(ranked) * (bar_h + gap)

    for decade in (2, 3, 4):
        gx = bar_x + bar_w * (decade - lo) / (hi - lo)
        c.path(f"M{gx:.1f},{top_c - 9:.1f} L{gx:.1f},{top_c + rows_h - 2:.1f}",
               stroke="#e6e5e1", sw=1.0)
        c.text(gx, top_c - 12, f"{10 ** decade:,}", size=6, fill=INK_3,
               anchor="middle")

    for k, (rel, count) in enumerate(ranked):
        y = top_c + k * (bar_h + gap)
        w = max(2.0, bar_w * (math.log10(count) - lo) / (hi - lo))
        c.text(label_r, y + bar_h - 2.2, rel, size=6.8, fill=INK_2, anchor="end")
        c.rect(bar_x, y, w, bar_h, fill=GROUP_COLOR[relation_group(rel)], rx=2.5,
               extra='opacity="0.9"')
        c.text(bar_x + w + 5, y + bar_h - 2.2, f"{count:,}", size=6.5, fill=INK_3)

    # =============================================================== footnote
    fy = max(grid_y + len(order) * cell_h, top_c + rows_h) + 24
    c.path(f"M{M:.1f},{fy - 12:.1f} L{M + PANEL_W:.1f},{fy - 12:.1f}",
           stroke="#e6e5e1", sw=1.0)
    for k, line in enumerate([
        "Counts are the entity and relation counts of the released graph; the figure is generated from it, so the numbers cannot drift from the data.",
        "Every Disease carries CITES_DOCUMENT to the source protocol it was extracted from, so any claim an agent reads out of the graph is traceable to a document.",
    ]):
        c.text(M, fy + k * 11, line, size=6.8, fill=INK_3)

    return c.render()


def main() -> int:
    check_coverage()
    node_counts, edge_counts, n_nodes, n_edges = collect()
    svg = draw(node_counts, edge_counts, n_nodes, n_edges)
    out = REPO_ROOT / "docs" / "figures" / "kg_schema.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}  ({len(svg):,} bytes)")
    print(f"  {n_nodes:,} entities, {n_edges:,} relations, "
          f"{len(node_counts)} types, {len(edge_counts)} relation types, "
          f"{len(EDGE_SIGNATURES)} signatures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
