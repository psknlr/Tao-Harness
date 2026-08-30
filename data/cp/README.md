# TCM-CP (clinical pathway)

**Built, not downloaded.** There is no published TCM clinical-pathway
benchmark, so this one is derived from the pathway layer of the knowledge graph:

```bash
python scripts/build_tcm_cp.py --out data/cp/TCM-CP.json
```

~5,400 items over ~300 diseases, in six subtask families:

| subtask | form | what it tests |
|---|---|---|
| CP1 pathway eligibility | eligible / not / insufficient | does the patient belong in this pathway |
| CP2 stage identification | single-select over the disease's own stages | can the agent locate the patient in the pathway |
| CP3 stage actions | multi-select | does it know what is due at this stage |
| CP4 treatment planning | single-select ×4 | syndrome → 治法 / 方剂 / 中成药 / 外治法 |
| CP5 monitoring | multi-select | what this pathway requires watching |
| CP6 transition decision | continue / advance / exit / insufficient | does it respect exit criteria |

## CP2 is filtered, and small on purpose

The first build was **underdetermined**: of 1,210 stage-identification items,
*zero* could be resolved from what the vignette exposed, and the median item
fit five stages of its own pathway. Pathway stages within one disease share
their monitoring text almost verbatim, so "which stage is this?" had no
derivable answer and scoring a model on it measured nothing.

Items are now emitted only when every distractor differs observably from the
gold stage, stage labels are disambiguated by variant where names repeat, and
vignettes carry a treatment-history line locating the patient in time. That
leaves 313 items from 1,212 stages. A smaller benchmark that measures
something beats a large one that does not; `TCM-CP.json.build_report.json`
records what was dropped and why.

## Read this before using the numbers

Gold answers come from the **same graph the KG arms consult**. TCM-CP is
therefore circular with respect to the KG conditions by construction: a KG arm
beating a no-KG arm here shows the agent can *retrieve and apply* pathway
knowledge faithfully, not that the graph makes a model clinically better.

It is an **instrument-capability benchmark**. Report it separately; never pool
its contrasts with SDT or PA, and never cite it as evidence for the
effectiveness claims. The harness labels it as such in the generated report and
the config says so at the top.

Two choices reduce, but cannot remove, the circularity:

- **Distractors come from other stages of the same disease**, so retrieving the
  disease is not enough — the agent has to land on the right stage.
- **Vignettes withhold the answer-bearing fields**: they are rendered from
  monitoring items and the syndrome picture, never from the stage name, its day
  actions, or its criteria.

## The safety endpoints

`CP6_transition_decision` carries the metrics worth watching:

- **`unsafe_transition`** — the model moved the patient onward (advance or
  discharge) when the criteria were unmet *or* unassessed. Any unsafe option
  in the answer counts, including a mixed answer containing both a safe and an
  unsafe choice; reading only the first sorted letter, as an earlier version
  did, scored `["A","C"]` as safe.
- **`missed_uncertainty`** — the gold answer was `insufficient_evidence` and
  the model did not say so. A pathway agent forced to always choose will choose
  when the record supports nothing, and in a discharge decision that is the
  dangerous direction. Roughly a third of CP6 items now have
  `insufficient_evidence` as the correct answer, generated from stages where
  some exit criteria are met and the rest simply have not been reassessed.

Contrasts on TCM-CP use a **disease-clustered bootstrap**: many items come from
one pathway, so treating them as independent observations understates the
confidence interval several-fold.
