# TCM-CP (clinical pathway)

**Built, not downloaded.** There is no published TCM clinical-pathway
benchmark, so this one is derived from the pathway layer of the knowledge graph:

```bash
python scripts/build_tcm_cp.py --out data/cp/TCM-CP.json
```

~3,700 items over ~300 diseases, in four subtasks:

| subtask | form | what it tests |
|---|---|---|
| CP2 stage identification | single-select over the disease's own stages | can the agent locate the patient in the pathway |
| CP3 stage actions | multi-select | does it know what is due at this stage |
| CP4 treatment principle | single-select | syndrome → 治法 |
| CP6 transition decision | continue / advance / exit / insufficient | does it respect exit criteria |

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

`CP6_transition_decision` carries the metric worth watching: `unsafe_transition`
counts items whose gold answer is "continue treatment" but where the model
recommended advancing or discharging — a recommendation to move a patient on
when the recorded criteria are not met.
