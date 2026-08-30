# Design notes

Reference for the reasoning chains, the tool contract and the measurement
decisions. The README covers *what* the harness does; this covers *why* each
piece is shaped the way it is.

---

## 1. SDT reasoning chain

TCMEval-SDT is four tasks under one weighted composite (0.2/0.3/0.4/0.1), with
tasks 2 and 3 as ten-option multiple choice. The graph has no `Symptom` and no
`Pathogenesis` entity, so the chain cannot run
`Symptom → Pathogenesis → Syndrome`. It runs:

```
Case text
   │
   ▼  ① clinical-information extraction        → task 1 (0.2)
Findings, in the case's own wording  ── transient, never written to the graph
   │
   ▼  ② deterministic cleaning + semantic anchor retrieval
Disease / DiseaseSubtype anchors
   │
   ▼  ③ option lookup: resolve each of the 10 named options against the graph
Per-option evidence (definition sentence, attested diseases, or "not found")
   │
   ▼  ④ LLM pathogenesis reasoning             → task 2 (0.3)
Pathogenesis options
   │
   ▼  ⑤ LLM syndrome selection                 → task 3 (0.4)
Syndrome options
   │
   ▼  ⑥ deterministic graph-consistency check on the chosen option's *text*
   ▼  ⑦ explanatory summary                    → task 4 (0.1, ROUGE-L)
Structured answer → official @-separated submission
```

Step ③ is what the multiple-choice format buys. The options arrive as named
pathogeneses and syndromes, so each candidate can be looked up directly rather
than guessed from the case text — a far better fit for this graph than
free-form naming. Every option is reported, found or not: a table biased toward
what the graph covers would imply that absence is evidence against an option,
which it is not.

Step ⑥ never calls a model, and it resolves the chosen *letter* to its option
*text* first — `verify_tcm_decision` cannot check a letter.

### Answer hygiene

Letters outside the item's own option set are dropped before scoring. A model
answering `K` on a ten-option question has not made a scoreable choice, and
under the official rule a stray letter counts as a wrong pick that dilutes
credit for the picks that were right.

### Ranking

```
Score(v) = α · SemanticSimilarity(q, v)      α = 0.60   BM25 over character n-grams
         + β · GraphRelevance(anchors, v)    β = 0.25   distance-decayed BFS
         + γ · SourceEvidence(docs, v)       γ = 0.15   document-level co-occurrence
```

All three terms are normalised to `[0,1]` and the weights are hashed into the
framework contract. Character n-grams rather than a segmenter: deterministic,
no model download, and better on compounded clinical vocabulary (`舌质暗红`,
`苔黄腻`) than a general-purpose segmenter never trained on TCM text. A dense
encoder can be registered through `EmbeddingProvider` and is then frozen
alongside everything else.

## 2. PA reasoning chain

```
Question + options
   │
   ▼  ① rule-family classification (LLM)
   ▼  ② knowledge-requirement planning (LLM chooses tools)
   │
   ├── drug knowledge      → retrieve_medication_knowledge
   ├── pharmacopoeia       → retrieve_pharmacopeia_entry
   ├── safety constraints  → retrieve_safety_constraints
   └── decidable checks    → check_decoction_requirement · check_duplicate_medication
                             check_restricted_item · check_dose · check_combination
   │
   ▼  ③ deterministic verdicts from the rule engine
   ▼  ④ LLM integration and option-by-option judgement
   ▼  ⑤ coverage audit (M4): did the answer claim graph support it never had?
Answer + evidence
```

Step ⑤ is deterministic. It flags a final answer that invokes the graph
(`根据图谱`, `图谱记载`) on a topic the graph does not encode — dose,
incompatibility, pregnancy grading, drug interaction — without acknowledging
the gap. That is the specific failure this graph invites, and
`coverage_honesty` in `tcm_eval/metrics.py` turns it into a reported number
rather than an anecdote.

---

## 3. Tool contract

| # | tool | domain | kind |
|---|---|---|---|
| 1 | `search_tcm_entities` | both | retrieval |
| 2 | `retrieve_clinical_context` | both | retrieval |
| 3 | `retrieve_syndrome_evidence` | both | retrieval |
| 4 | `retrieve_source_evidence` | both | provenance |
| 5 | `retrieve_medication_knowledge` | safety | retrieval |
| 6 | `retrieve_safety_constraints` | safety | retrieval |
| 7 | `retrieve_pharmacopeia_entry` | safety | retrieval |
| 8 | `verify_tcm_decision` | both | **rule engine** |
| 9 | `check_dose` | safety | **rule engine** → always `not_covered` |
| 10 | `check_duplicate_medication` | safety | **rule engine** |
| 11 | `check_restricted_item` | safety | **rule engine** |
| 12 | `check_combination` | safety | **rule engine** → always `not_covered` |
| 13 | `check_decoction_requirement` | safety | **rule engine** |

Tools 9 and 12 are the honest ones. They still return what the graph *does*
hold — toxicity flags, preparation markers, formula co-occurrence — labelled as
adjacent evidence, never as a verdict.

### Positional attribution

A protocol line lists many herbs and annotates only some:

> `生地、生石膏、地榆炭、生大黄(后下)等`

`后下` belongs to 大黄. Attributing it to every herb in the sentence would make
`check_decoction_requirement` wrong in exactly the cases N-003 asks about, so
`markers_for_herb_in_sentence` requires the parenthetical to open immediately
after that herb's own name. There is a test.

---

## 4. Entity resolution is conservative

BM25 scores are normalised against the best hit, so the top result for *any*
query scores 1.0 — including for a drug the graph has never seen. Resolving on
rank alone let a query for 阿司匹林 return another herb's contraindications.
Resolution therefore also requires surface-form overlap
(`name_similarity ≥ 0.5`). Returning nothing is safe; returning the wrong drug
is not.

---

## 5. Measurement decisions

**SDT uses the benchmark's own rules, not ours.** `tcm_eval/official_sdt.py`
reproduces `evaluate.py` and is verified against the vendored original on
perfect, random and empty submissions. Two quirks are preserved because they
change the reported number:

- A wrong option **dilutes rather than forfeits**:
  `correct / (|gold| + n_wrong)`. Selecting all ten options scores `|gold|/10`,
  not zero, so hedging is weakly rewarded. `n_syndrome_selected` is reported
  beside the score so hedging is visible rather than invisible.
- The evaluator never strips line terminators, so task 4 always compares
  `prediction + "\n"` against `reference + "\n"`. The shared newline gives an
  **empty explanation ~0.009 instead of 0**. Tiny, but a harness claiming
  agreement has to model it; pass `emulate_official_io=False` for clean
  arithmetic.

**PA uses strict set equality**, since no official scorer ships with it, with
exam-style partial credit reported alongside — with only 31 multiple-choice
items strict accuracy is noisy, and the pair reveals under-selection versus
guessing. The two benchmarks deliberately do *not* share a multi-select rule;
forcing one would misreport the other.

**PA is reported split by rule groundability.** Half the released items are in
families the graph cannot ground; pooling them with the grounded families
dilutes any real effect and invites the wrong conclusion.

**Paired tests throughout.** The same cases pass through every arm, so exact
McNemar (binary metrics) and paired bootstrap (continuous) condition on the
item. Holm–Bonferroni across the whole family: 5 models × 5 contrasts × 2
benchmarks would otherwise manufacture significance.

**Judge failure ≠ zero.** A judge error records *unscored*. A silent zero is
indistinguishable from a genuinely bad answer, and would penalise whichever
model most often trips the judge's parser.

---

## 6. Clinical pathway execution

The graph carries 1,440 `PathwayStage` nodes joined by 1,118 `NEXT_STAGE`
edges, with `day_actions` on 89% of stages and `nursing_items` on 64%. Until
V2 the agent could reach a stage's criteria but not its actions, and
`NEXT_STAGE` was never traversed — the pathway layer was static context rather
than a state machine.

```
Patient state + follow-up findings
   │
   ▼  retrieve_pathway_stage   (entry/exit criteria, day_actions,
   │                            nursing_items, monitoring, outcomes,
   │                            previous and next stages)
Current stage
   │
   ▼  retrieve_treatment_plan  (syndrome → 治法 → formula / patent / external)
Stage-appropriate treatment
   │
   ▼  evaluate_pathway_transition   ← deterministic, no model
   │     exit criteria     × findings
   │     successor entry criteria × findings
   │     NEXT_STAGE traversal
   ▼
continue | advance | exit | insufficient_evidence
```

The transition evaluator is deliberately conservative. Criteria are matched at
the character-bigram level against the supplied findings; partial overlap
reports `partial`, never `met`, and a finding that negates a criterion marks it
`contradicted` rather than satisfied. With no findings supplied it returns
`insufficient_evidence` — it will not recommend discharge from silence. Where
a stage records no exit criteria (only 12% do) it says so instead of inferring
that the patient may leave.

`unsafe_transition` is the endpoint worth watching in TCM-CP: gold says
continue treatment, the model said advance or discharge. Recommending a patient
onward when the recorded criteria are unmet is not symmetric with the opposite
error, and pooling both into an accuracy would hide it.

### Why a fourth domain

`clinical_pathway` withholds nothing, because executing a pathway *is* deciding
on treatment. That is sound here and would not be sound for SDT: nothing in
TCM-CP is answerable by inverting a syndrome→formula mapping, whereas in SDT
that inversion hands over the answer. Widening the clinical domain to enable
pathway work would have destroyed SDT's isolation; a separate domain does not.

## 7. Controlling for test-time compute

M3 spends more model calls than M2, and M4 more than M3. A raw `M2→M3` gain
therefore measures agency *and* extra thinking together. Two controls separate
them:

- **M2C** — M3's turn budget, no graph access. Each turn the model is prompted
  onward with no new information, so no evidence enters while the compute
  matches. `M2C→M3` is the agentic KG effect.
- **M3C** — M4's extra revision turn, with no verification evidence in it.
  `M3C→M4` is the effect of the verification *content* rather than of being
  asked to look again.

A compute-matched control only licenses its conclusion if the match held in
practice, so the report includes a parity table of realised calls and tokens
per pair, flagging any pair outside a 0.8–1.25 call ratio as `MISMATCH`.

## 8. Known limitations

1. **Coverage.** 10 of 19 PA rule families are ungroundable here — **166 of
   328 released items (51%)**, including the largest family, A-003 dosage (87
   items). PA gains must be read per rule family. The ungroundable families
   double as a control: a KG gain there is not knowledge injection.
2. **Half the syndromes are name-only.** 336 of 648 lack a definition sentence,
   so retrieval anchors them weakly. `retrieve_syndrome_evidence` reports
   `PARTIAL` for these. Relatedly, the graph recognises only ~32-37% of SDT's
   answer options at all — though it recognises gold options and distractors at
   the same rate, which is what keeps the option-lookup tool from leaking
   answers.
3. **Uniform text tool protocol.** Removes provider function-calling
   differences as a confound, at the cost of not measuring native
   tool-calling quality. Deliberate; state it in the methods.
4. **Lexical retrieval by default.** Reproducible offline and identical for
   every model, but weaker than a good Chinese medical embedding model. The
   dense path exists and is frozen when used; if you enable it, report it as a
   framework change — the hash will differ.
5. **No test-set contamination check.** SDT cases drawn from public sources may
   overlap frontier pre-training data. The M0→M3 *delta* is more trustworthy
   than any absolute number here.
