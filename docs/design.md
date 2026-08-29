# Design notes

Reference for the reasoning chains, the tool contract and the measurement
decisions. The README covers *what* the harness does; this covers *why* each
piece is shaped the way it is.

---

## 1. SDT reasoning chain

The graph has no `Symptom` and no `Pathogenesis` entity, so the chain cannot
run `Symptom → Pathogenesis → Syndrome`. It runs:

```
Case text
   │
   ▼  ① deterministic cleaning (demographics stripped, identically for all models)
Clinical features  ── transient runtime variables, never written to the graph
   │
   ▼  ② semantic anchor retrieval over per-entity virtual documents
Disease / DiseaseSubtype / PathwayStage anchors
   │
   ▼  ③ typed graph expansion (1–2 hops) + re-ranking
Syndrome candidates + definition sentences + DocumentSource
   │
   ▼  ④ LLM pathogenesis reasoning        ← the graph never supplies this
Pathogenesis (latent variable)
   │
   ▼  ⑤ LLM syndrome selection
Candidate syndrome
   │
   ▼  ⑥ deterministic graph-consistency check (verify_tcm_decision)
Final structured answer
```

Step ⑥ never calls a model. It checks that the syndrome exists in the graph,
that it belongs to the anchored disease, optionally that the stated treatment
principle matches, and reports which case features appear verbatim in the
syndrome's definition sentence. `not_in_graph` is explicitly *not* evidence
against the answer — the graph covers national protocols, not all of TCM.

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

---

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

**Syndrome partial credit.** Chinese syndrome names compose additively
(`痰阻血瘀，湿郁化热证` is two conjuncts). The headline metric stays strict
exact match; atom-F1 is reported beside it because an answer recovering one of
two conjuncts is genuinely closer than one recovering neither.

**Multi-select credit.** Strict set equality is primary, as the benchmark
intends. Exam-style partial credit (any wrong option forfeits; otherwise
proportional) is reported alongside because with only 31 multiple-choice items
strict accuracy is very noisy, and the pair reveals under-selection versus
guessing.

**Unscoreable steps are omitted, not zeroed.** A split that does not annotate
`pathogenesis` yields no `pathogenesis_f1`, rather than a perfect score for
matching emptiness or a zero for every model.

**Paired tests throughout.** The same cases pass through every arm, so exact
McNemar (binary metrics) and paired bootstrap (continuous) condition on the
item. Holm–Bonferroni across the whole family: 5 models × 5 contrasts × 2
benchmarks would otherwise manufacture significance.

**Judge failure ≠ zero.** A judge error records *unscored*. A silent zero is
indistinguishable from a genuinely bad answer, and would penalise whichever
model most often trips the judge's parser.

---

## 6. Known limitations

1. **Coverage.** 10 of 19 PA rule families are ungroundable here (see
   `kg_coverage.md`). PA gains should be read per rule family; pooling grounded
   and ungroundable families dilutes the effect and invites the wrong
   conclusion.
2. **Half the syndromes are name-only.** 336 of 648 lack a definition sentence,
   so retrieval anchors them weakly. `retrieve_syndrome_evidence` reports
   `PARTIAL` for these.
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
