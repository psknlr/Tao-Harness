# Figures

Figures are **generated from the artefacts they describe**, never drawn by
hand, for the same reason `scripts/experiment_manifest.py` exists: a number in
a figure is the part of a paper a reader trusts without checking, so it must
not be able to drift from the data.

## `kg_schema.svg` — the knowledge graph

```bash
python3 scripts/figure_kg.py          # reads kg/ and tcm_kg/schema.py
```

`tests/test_figure.py` fails if the committed SVG is not what today's graph and
schema produce, and `scripts/figure_kg.py` refuses to run at all if the schema
has grown a relation the drawing does not cover. So the figure cannot silently
go stale, and it cannot silently omit part of the ontology.

### Caption

> **The TCM clinical knowledge graph.**
> (**a**) Ontology. The 14 entity types (with entity counts, bar length
> proportional to √count) and the 24 typed relation signatures between them,
> arranged along the clinical flow from department and disease through
> differentiation to therapy and regulation, with the clinical-pathway branch
> above and the provenance layer below. Self-relations are collapsed to one
> glyph per type.
> (**b**) Access domain. Which entity types an agent may reach in each
> benchmark. Filled = readable; dashed = readable only to verify a candidate
> the model has already produced; grey = withheld. The clinical-reasoning
> domain withholds every treatment entity, so an agent cannot invert
> syndrome→formula and read its answer off the prescription instead of
> reasoning to it.
> (**c**) The 20 relation types by edge count on a log axis, coloured by the
> entity family each relation points at.
> Counts are those of the released graph: 9,350 entities and 27,972 relations.

### Why not a force-directed layout of all 9,350 nodes

It would show that the graph is big — which the caption says in four words —
and hide the structure, which is the thing worth a figure. Panel **a** is the
ontology, and panel **c** carries the scale that a hairball would gesture at.

### Rasterising

The SVG is the source of truth and carries live text, which is what journals
ask for. For a slide or a preprint preview, raster it at whatever scale you
need — e.g. with the Chromium already on most machines:

```bash
printf '<!doctype html><style>html,body{margin:0}</style>' > /tmp/fig.html
cat docs/figures/kg_schema.svg >> /tmp/fig.html
chromium --headless --disable-gpu --hide-scrollbars \
  --force-device-scale-factor=3 --window-size=1132,800 \
  --screenshot=kg_schema.png file:///tmp/fig.html      # 3396x2400, ~470 dpi at 183 mm
```

Rasters are deliberately **not** committed: nothing tests them, so they are the
one artefact here that could drift without anything noticing.

### Colour

The four categorical hues plus the provenance neutral were checked with a
palette validator rather than by eye, all-pairs (a schema diagram puts any two
families adjacent, so the adjacent-pairs rule does not apply):

| check | result |
|---|---|
| CVD separation | worst pair ΔE 9.9 — teal↔orange, protanopia |
| Normal vision | worst pair ΔE 16.3 — violet↔blue |
| Contrast vs surface | every fill ≥ 3:1 |
| Chroma floor | passes for all four hues; the provenance neutral is deliberately achromatic |

Colour never carries information alone: every box, cell and bar is also
labelled in text, and each entity box carries a colour flag on its leading edge
so the family survives greyscale print.
