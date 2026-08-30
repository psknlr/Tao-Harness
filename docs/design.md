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

- **M2C** — M3's turn budget, **M2's static KG block**, no tools. Both arms
  hold the same graph evidence, so `M2C→M3` isolates *adaptive retrieval*, not
  graph access. (An earlier version gave M2C no graph at all, which folded the
  KG effect back into a contrast whose whole purpose was to exclude it.)

  The budget match is now **per case**. M2C used to choose for itself when to
  answer, and it never reached the parity check — that ran only inside
  `run_branch_group`, which M2C was not part of. One case could give M2C two
  calls and M3 six while the averages agreed to within a few percent, and a
  mean is not a match for a paired test. M2C is generated in the same
  invocation as its M3 twin and pinned to the calls that twin actually spent:
  it cannot answer early (it is asked to reconsider) or run long (the final
  turn demands an answer).

  **Turn-matched, not compute-matched.** Calls are equal per case; tokens are
  not. An agent's tool results enter its later prompts, so M3 carries more
  input tokens than M2C at the same turn count -- measured at up to +23.5% --
  and M4's verification report is longer than M3C's sham one. Forcing tokens
  equal would delete the evidence the intervention *consists of*, so the report
  states the call ratio and the token ratio side by side and the claim is the
  one that is true.
- **M3C** — M4's extra revision turn, with no verification evidence in it.
  `M3C→M4` is the effect of the verification *content* rather than of being
  asked to look again.

A turn-matched control only licenses its conclusion if the match held **per
case**, not on average. M4 used to return early when no deterministic checker
adjudicated an item, spending one model call fewer than M3C on exactly those
cases — the mean stayed close while the contrast over that subset was a
second-turn effect. M4 now always takes its revision turn, on an explicit
`not_applicable` report. Branch groups check parity case by case, flag both
arms on a break, and the report drops flagged pairs from the contrast and
counts them in the parity table.

Because M4's arm now contains two treatments under one label — real
verification evidence, and a bare prompt to look again — the report splits the
`M3C→M4` gain by **verification stratum**: `deterministic` (a checker
adjudicated the answer), `audit_only` (the prose was audited for over-claiming)
and `not_applicable`. A gain concentrated in the last stratum is a second-turn
effect and has to be reported as one.

## 8. What each contrast licenses

The point of the control arms is that only some contrasts support a causal
claim. Stated plainly, so a table cannot be read as more than it is:

| contrast | claim it supports | what it still contains |
|---|---|---|
| `M1 − M0` | prompt structure helps | – |
| `M2 − M1` | static KG evidence helps | – |
| `M3 − M2` | *nothing on its own* | agency **and** extra compute |
| `M3 − M2C` | adaptive retrieval helps | – (KG evidence and compute held constant) |
| `M4 − M3` | *nothing on its own* | verification **and** an extra turn |
| `M4 − M3C` | verification content helps | – (trajectory and compute held constant) |
| `M0 − M3` | the whole scaffold helps | everything at once; not a KG result |

Three properties make the last two rows trustworthy, and each was absent in an
earlier version:

1. **Nothing routes on the answer key.** PA's verifier selects its checker from
   the model's own `rule_category`, not the benchmark's `rule_id`. Reading the
   annotation would have told M4 which safety rule was in play — a large part
   of the task, given to one arm only.
2. **M3, M3C and M4 share one agent phase.** Running them independently meant
   `M3C → M4` carried trajectory noise on top of the verification difference,
   and inconsistent provider seed support means a seed cannot remove it.
3. **The verifier is not reachable by the agent** — enforced, not merely
   withheld. Otherwise M3 is an optionally-self-verifying arm and `M3 → M4`
   contrasts optional with mandatory verification, a much weaker claim easily
   misread as the stronger one. This means the *whole* verification surface,
   not just `verify_tcm_decision`: every tool carries a `phase` (`agent`,
   `verification`, `both`), and the five deterministic PA checkers are
   verification-phase.

   Keeping them out of the prompt is not enough. A model can produce a name it
   was never shown — from its own priors, or from an earlier turn of a long
   context — and `ToolRegistry.call` checked name, domain, budget and required
   arguments but never phase, so the call went through. `ToolContext` now
   carries the phase and a cross-phase call is refused at the boundary. A
   regression test has an M3 agent emit `check_dose` and asserts it is
   rejected.
4. **Tool calls record who made them.** A `ToolStep` carries its phase, so
   `tool_selection_accuracy`, `coverage_honesty` and `pathogenesis_probe_rate`
   read only the calls the *model* chose. The M4 verifier calls the correct
   checker by construction; counting those would have credited M4 with tool-use
   skill for a choice it never made, and the report separates
   `agent calls` from `verifier calls` for the same reason.

## 9. Reproducibility: what a manifest freezes

A framework hash proves two arms shared a scaffold. It does not say which
graph, dataset, code revision or model snapshots produced a number. The
per-run manifest does:

| recorded | catches |
|---|---|
| `kg_content_sha256` | any edited node, edge, evidence or `source_docs` — the last because the SourceEvidence retrieval term reads it |
| `dataset_sha256`, `dataset_results_sha256` | a swapped or re-exported split |
| `case_ids` + `case_set_sha256` | a changed `limit` silently re-scoring more cases than the run covered |
| model `fingerprint_sha256` | a repointed model key, or an `extra_body` change such as switching on a reasoning mode |
| `tools_impl_sha256`, `scorers_impl_sha256`, `retrieval_impl_sha256`, `runtime_impl_sha256` | a rewritten checker body that leaves its `ToolSpec` untouched |
| `run_signatures` | traces that pooled across models or code revisions (below) |
| `git_commit`, `python`, `created_at` | everything else |

`score` and `report` read the frozen manifest rather than recomputing from the
config as it reads today — recomputation at report time produced a *different*
framework hash from the one the traces were generated under, so the report
attested to a run that never happened.

**The shared trajectory is an invariant, not an intention.** M3, M3C and M4
come from one agent phase — but a resume used to regenerate only the *missing*
arm, so a surviving M3C paired with a fresh-prefix M4 and `M3C→M4`, whose whole
claim is that the verification report is the only difference, silently became a
comparison of two independent runs. Nothing checked it: the pairing looked at
`parity_error` and not at provenance. Any incomplete group is now regenerated
whole, every arm carries a `branch_group`, and the co-generated contrasts
require both arms to share it.

**Run signature.** The framework hash has to stay *equal* across models: that
is what makes "every arm saw the same scaffold" a checkable claim. The same
blindness means traces from two model snapshots, or from before and after a
tool rewrite, carry one hash and pool silently. `run_signature` covers what the
framework hash omits — the model spec, the four implementation hashes and the
frozen case set — and is stamped on **every trace**, so the check survives a
missing or stale manifest. Condition is deliberately excluded: conditions are
the independent variable and must stay poolable.

It is used in three places:

- **Resume** matches on the signature. Resuming on the framework hash kept
  exactly the traces a resume exists to regenerate.
- **`write_manifest` never overwrites** a manifest describing a different
  experiment. It was the only record of what the traces beside it were produced
  under, and a resumed run with an edited config replaced that record while
  leaving the traces. `--new-run` archives it instead.
- **`score` fails closed.** Mixed framework hashes, mixed run signatures,
  traces that do not match the manifest's signature, a changed KG or dataset,
  or changed tool/retrieval/runtime code all stop the run. A changed *scorer*
  stays a note — re-scoring recorded traces with a fixed scorer is the point of
  separating generation from scoring. `--allow-drift` overrides, and the drift
  is written into `scored_manifest` and surfaced as a banner at the top of the
  report, so a forced number cannot later be mistaken for a clean one.

**Design signature.** One level up from the apparatus. The run signature omits
the condition on purpose, so the arms of one experiment can be pooled — which
leaves `conditions`, `samples`, `limit` and the sampling rule in no fingerprint
at all. Narrow a seven-arm config to `[M0, M1]` and the M2–M4 traces still
matched: resume kept them, and the manifest described a two-arm experiment
beside a seven-arm trace file. Drop `samples` from 3 to 1 and the extra samples
survived as ordinary items, so scoring stopped running consensus over them and
one case carried three times its neighbours' weight. Neither is visible to the
apparatus signature, because neither changes the apparatus. `design_signature`
covers both levels, is stamped on every trace, and gates resume alongside an
explicit check that a trace's condition and sample index are ones this run
declares.

**Dataset identity.** `case_id` is the primary key of three separate
mechanisms — the gold lookup, the `(case_id, condition, sample)` resume key and
`index_items` — and all three are dicts. A TCM-CP build shipped 825 rows over
331 identifiers because CP4 ids named the syndrome and not the disease, so a
model answering 弱视 was scored against 糖尿病性胃轻瘫's key, where the same
treatment sits at a different letter, and resuming a run silently collapsed the
duplicates. The builder now validates the whole set before writing (unique and
non-blank ids, gold inside the options, at least two distractors, no two
options carrying the same answer, no answer echoed in the question) and refuses
on any violation. `load_dataset` imposes unique keys for every benchmark and
records how many rows it had to rename — which immediately found TCM-SD's
released dev split shipping 178 repeated `user_id` values, a third-party file
that is not ours to rebuild and is disambiguated deterministically instead.

The generation cache keys on the model fingerprint, so turning on a vendor
reasoning mode no longer serves responses generated without it.

## 10. Contamination: the one confound pairing cannot remove

The whole design rests on a claim a paired comparison cannot establish.

Pre-training contamination is a **shared** confound. If a frontier model
memorised a classical medical record, every arm of that model memorised it
equally, so the paired `M1 → M2` difference cancels it. That is a real
strength of the design and it is why the deltas are more trustworthy than any
absolute score.

Graph contamination is **not shared**. Only M2, M3 and M4 can read the graph.
So a case whose answer sits in the graph's evidence text hands exactly those
arms the answer key while the others reason — and the difference lands in the
contrast the study reports as the knowledge-graph effect. No amount of pairing
removes it, because the leak *is* the intervention.

`benchmark_runner contaminate` audits the text a KG arm can actually reach —
7,257 node definition sentences, 32,403 edge evidence sentences and
`DocumentSource` metadata, 16,052 distinct passages — at four levels:

| level | what it catches |
|---|---|
| exact | the gold answer appears verbatim in graph text |
| n-gram Jaccard | a rewritten record: character 5-grams, no segmenter needed |
| containment | a graph sentence *inside* a longer case, which Jaccard understates |
| provenance | the case cites a source the graph holds |

Everything is deterministic and dependency-free. There is deliberately **no
embedding level**: an embedding model would be a second uncontrolled variable
inside an audit whose entire purpose is to be checkable by someone else.

Measured on the released benchmarks:

| benchmark | cases | clean | possible | likely |
|---|---|---|---|---|
| SDT | 50 | **70.0%** | 12.0% | 18.0% |
| PA | 328 | **89.0%** | 5.8% | 5.2% |
| CP | 500 | 0.0% | 0.0% | 100% |

An earlier version of this audit reported SDT at 100.0% and PA at 99.4%. Those
numbers were wrong, and the difference is the whole reason this section exists.
Three defects, each confirmed by construction before being fixed:

1. **The corpus was not what an agent can read.** It indexed node
   `first_mention` sentences and edge evidence and called that reachable text.
   Measured against what the tools return and the retriever indexes, it missed
   **99.8% of PathwayStage text, 100% of PharmacoPoeiaEntry, 100% of
   ExternalTherapy** and 96% of SafetyContext. A case drawn from a monitoring
   item or a pharmacopoeia function scored `clean` while
   `retrieve_pathway_stage` would hand it straight to M3. The corpus is now
   built from `KGStore.virtual_document` — by construction the text the
   retriever indexes, so the audit cannot fall behind retrieval without the
   index falling behind too — plus every string attribute as its own atom,
   because a leak is usually one field and a 4,000-character concatenation
   hides it. 16,052 passages became 33,223; per-type uncovered text is 0%.

2. **Containment ran one way only.** It measured the share of a graph passage
   inside the case and missed the reverse: a case that is a verbatim *excerpt*
   of a longer graph passage. A 40-character excerpt of a 544-character passage
   scored 0.315 and landed in `possible`; against a longer passage it reads
   `clean`. Both directions are measured now and the stratum takes the larger.

3. **Multi-select gold was concatenated before matching.** Two options each
   lifted verbatim from the graph — from *different* passages, as they would
   be — produced a joined string present in no single passage, so the case was
   filed `clean` with both of its answers sitting in the graph. SDT tasks 2 and
   3 are multi-select on 27 of 50 cases.

**The detector is validated, and that is what makes a null result mean
anything.** TCM-CP is contaminated by construction and the audit says so at
100%. Planted positives are caught — a verbatim copy, an answer lifted from
the graph, a case that is an excerpt of a passage, a multi-select item with one
leaked option — along with six realistic rewrite modes: punctuation swapped,
clauses reordered, embedded in a longer case, 30% of clauses dropped, half
kept. Unrelated text stays far below threshold.

The known limits, stated rather than papered over. A character-shuffled passage
is missed; no lexical method survives that, and no rewritten record looks like
it. More importantly, **`clean` means this audit found nothing, not that there
is nothing.** A case paraphrased into different vocabulary would pass. The
audit is hash-locked to the graph, dataset, resolved gold, case set and
thresholds it was computed against, and `score` refuses a report that does not
describe the run being scored — a stale audit is worse than none, because none
is at least visible in the report.

`score` attaches each case's stratum and the report recomputes `M1→M2` and
`M2C→M3` on the clean subset beside the full one. State the result as what it
is: a gain persisting on the lexically clean subset is **less consistent with
detectable direct lexical or provenance overlap**. It is not proof that no
contamination exists, and the difference matters in a medical-AI paper. Where
no audit has been run the report says so, because silence there looks like a
clean result.

Leave-source-out KG — withholding a case's own source document while that case
runs — is now worth building. The threshold stated when it was deferred was
"any `likely` case in an effectiveness benchmark"; the corrected audit finds
**9 in SDT and 17 in PA**, so that threshold is met. It is deferred to a
separate change rather than bundled here because it adds per-case graph
mutation, and the honest interim position is the one the report already takes:
run the KG contrasts on the clean subset and describe the result as a
sensitivity analysis, not a proof.

## 11. Reading TCM-CP: the macro-average

TCM-CP's nine subtasks run from 306 items (CP6, transition decisions) to 988
(CP3 and CP5). A pooled accuracy is therefore 53% stage lookup and monitoring
and 5.5% transition decision — so a model that reads stages well and moves
patients on badly reads as a good pathway executor.

The **prespecified CP endpoint is the macro-average**: the mean of the six
capability means (CP4 is one capability probed four ways, averaged first), each
weighted equally. Its contrasts use a **hierarchical** macro bootstrap —
resample diseases within each subtask, average the subtasks of a family, then
weight the six families equally — so the interval describes the same quantity
as the point estimate. The pooled cluster-bootstrap contrast is kept as a
secondary table.

The hierarchy is not a detail. An earlier version passed the nine subtasks
straight in as strata, which weights CP4 at 4/9 where the table above it
weights CP4 at 1/6. On a set where only CP4 improves from 0 to 1, the table
said Δ = 0.167 and the test said Δ = 0.444: the confidence interval and the
p-value described a quantity that appeared nowhere in the report, under a
comment asserting they matched. The test that should have caught it asserted
the *estimator's name*, which was correct while the number was wrong.

**CP4 primary is disease-specific only.** `Syndrome → Treatment` is stored as a
global binary relation, but the clinical fact is ternary: *this disease's*
guideline, for this syndrome, recommends this treatment. Read globally,
补中益气汤 is "the formula for 脾胃虚弱证" whether the pathway is 弱视,
吉兰巴雷综合征 or 糖尿病性胃轻瘫 — and the first global edge became the gold for
all three. Measured before the fix, 54.45% of CP4 gold treatments shared no
source document with the current `Disease → Syndrome` edge.

That is not proof of clinical error — **异病同治** is a real principle and the
graph's document boundaries are not clinical boundaries — but the system could
not show that *this pathway* recommends it, which is a weaker claim than a
pathway agent should make. `kg.treatments_of(syndrome, disease)` splits
`disease_specific` from `cross_disease_general` on shared provenance.

Preferring a grounded gold was not enough. When a disease's guideline recorded
no treatment of that type at all, the builder still fell back to a
cross-disease edge — 14.5% of CP4, 21% of the formal sample, and 48% of formal
CP4-patent items — so "which treatment does this pathway recommend?" was keyed
from another disease's evidence. That question cannot be answered that way.

CP4 primary is now disease-specific only, and the cross-disease items become
**CP4G**, asking what they can actually attest: whether a treatment has
syndrome-level support *across* diseases. That is knowledge transfer under
异病同治, not pathway execution, so CP4G is excluded from the six-family
macro-average and reported in its own table. Formal CP4 primary is 200/200
disease-grounded, from 158/200.

The two-hop subtype fix contributed here too: 224 syndromes reachable only
through a `DiseaseSubtype` had looked context-free, so their treatments were
filed as cross-disease — the disease's own guideline reclassified as somebody
else's.

The sample is drawn stratified by subtask (`dataset.stratify: subtask`) for the
same reason: a head-slice of 400 gives CP6 twenty-eight items, too thin to
carry a paired test on precisely the subtask with a patient-safety reading.

## 12. Known limitations

1. **Coverage.** 10 of 19 PA rule families are ungroundable here — **166 of
   328 released items (51%)**, including the largest family, A-003 dosage (87
   items). PA gains must be read per rule family. The ungroundable families
   double as a control: a KG gain there is not knowledge injection.
2. **Half the syndromes are name-only.** 336 of 648 lack a definition sentence,
   so retrieval anchors them weakly. `retrieve_syndrome_evidence` reports
   `PARTIAL` for these. Relatedly, the graph recognises only ~32–37% of SDT's
   answer options at all — though it recognises gold options and distractors at
   the same rate, which is what keeps the option-lookup tool from leaking
   answers.
3. **Uniform text tool protocol.** Removes provider function-calling
   differences as a confound, at the cost of not measuring native tool-calling
   quality. Deliberate; state it in the methods.
4. **Lexical retrieval by default.** Reproducible offline and identical for
   every model, but weaker than a good Chinese medical embedding model. The
   dense path exists and is frozen when used; enabling it is a framework
   change and the hash will differ.
5. **No test-set contamination check.** SDT cases are classical published
   medical records and may overlap frontier pre-training data. The
   turn-matched *deltas* are more trustworthy than any absolute number.
6. **TCM-CP is circular with respect to the KG arms.** Its gold answers come
   from the graph, so it measures pathway-execution faithfulness, not clinical
   effectiveness. Never pool it with SDT or PA. Its contrasts use a
   **disease-clustered** bootstrap: many items derive from one pathway, and
   treating them as independent understates the interval several-fold.
7. **TCM-CP drops what it cannot ask fairly.** The build emits 5,611 items
   from 299 diseases and discards 1,154 candidate stages: 911 with no exit
   criteria, 195 first stages rebalanced down to a 25% share (a first stage is
   identifiable from "no prior treatment" alone, which is a position cue and
   not stage reasoning), 46 whose distractors were not distinguishable from the
   gold stage, and 2 with too few distractor actions. That is the ceiling this
   graph supports; a larger CP2 would be a larger set of unanswerable
   questions.
   Every subtask is checked for answer leakage — no emitted field may contain
   the gold string, in either direction — and all nine are at 0%. Two leaks
   were found this way and fixed: CP5 at 100% (the monitoring plan was printed
   in the vignette it asked about) and CP3 at 59.8%, which no review had
   flagged and which only a general per-field guard caught.
8. **No longitudinal patient state.** TCM-CP items are independent decisions,
   not a trajectory. There is no persistent `PatientState`, no episode, no
   T0→T1→T2 sequence carrying treatment response, monitoring results, safety
   events or variance. The honest description of what exists is
   *graph-grounded pathway decision support*, not *autonomous longitudinal
   clinical pathway management*. This is the largest remaining gap between
   this harness and a complete clinical-pathway agent, and it is a design
   task rather than a bug.
9. **Five models is too few for the compensation claim.** Spearman ρ over
   n = 5 has very little power. Report a base-ability/gain relationship as
   *exploratory evidence of a compensatory pattern*, not as a finding, unless
   the panel grows to 10–15.
10. **This harness is not Codex or Claude Code.** It is a self-contained agent
    runtime, which is what makes the five-model comparison clean. If the
    research question is "does adding a KG help *inside* a commercial coding
    agent", that is a different experiment and this does not answer it.
11. **The KG-derived benchmark has had no expert validation.** TCM-CP items
    were filtered for machine-checkable discriminability, not reviewed by
    clinicians. Before publishing CP results, have two or three TCM physicians
    answer a sample independently: if their agreement is low, those items
    cannot support a model comparison either.
