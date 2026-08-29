# TCM-KG Agent Harness

A frozen agent harness for measuring what a Traditional Chinese Medicine
clinical knowledge graph adds to frontier LLMs, across two complementary
benchmarks:

- **TCMEval-SDT** — a *four-task* syndrome-differentiation benchmark
  (knowledge-enhanced *clinical reasoning*), scored by the benchmark's own
  weighted rules
- **TCMEval-PA** — 328 multiple-choice prescription-audit items across 19 rule
  families (knowledge-grounded *rule and tool reasoning*)

A third corpus, **TCM-SD** (43k EHR records over 148 syndrome labels, plus a
1,027-entry syndrome knowledge base), loads through the same interface for
scale-up work.

The same framework serves both. Only `model.generate()` differs between arms,
and a `framework_hash` proves it.

```
                         TCMEval
              ┌─────────────┴─────────────┐
         TCMEval-SDT                 TCMEval-PA
              │                           │
      Clinical Parser              Rule Classifier
              │                           │
              └────────────┬──────────────┘
                           ▼
                   Frozen Agent Runtime  (M0 · M1 · M2 · M3 · M4)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    Clinical Reasoning Domain   Prescription Safety Domain
    Department · Disease        Disease · Syndrome · Formula
    DiseaseSubtype · Syndrome   Herb · PatentMedicine
    PathwayStage · DocSource    SafetyContext · RestrictedItem
                                PharmacoPoeiaEntry · DocSource
              │                         │
              └────────────┬────────────┘
                           ▼
              8 KG tools + 5 deterministic checkers
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        Reasoning Agent            Rule Engine
              └────────────┬────────────┘
                           ▼
                  Verification · Trace · Score
```

---

## The benchmarks, as released

**TCMEval-SDT is four tasks, not one**, and its `scripts/evaluate.py` fixes the
scoring:

| task | field | form | weight |
|---|---|---|---|
| 1 | `Clinical Information` | `;`-separated findings | 0.2 |
| 2 | `Answers of TCM Pathogenesis` | letters, 10 options, often multi | 0.3 |
| 3 | `Answers of TCM Syndrome` | letters, 10 options, often multi | 0.4 |
| 4 | `Explanatory Summary` + `Syndrome Differentiation` | free text, ROUGE-L | 0.1 |

Three consequences the harness is built around:

- **Pathogenesis and syndrome are multiple choice.** The options arrive as
  *named* pathogeneses and syndromes, so the agent can look each candidate up
  in the graph rather than guess a name from the case text. That fits this
  graph far better than free-form naming.
- **The held-out splits ship blank answer columns.** Gold labels for
  validation and test live only in `Results/*.txt`, in the `@`-separated
  submission format. The loader merges them by record ID, so pointing at the
  test split yields a scorable dataset instead of silently scoring against
  empty strings.
- **Scoring follows the benchmark, quirks included.** `tcm_eval/official_sdt.py`
  reproduces the official rules and is checked against the vendored original on
  perfect, random and empty submissions (`tests/test_official_sdt.py`). Two
  quirks matter and are preserved: a wrong option **dilutes rather than
  forfeits** credit (`correct / (|gold| + n_wrong)`, so selecting all ten
  scores `|gold|/10`, not zero), and because the evaluator never strips line
  terminators an **empty explanation scores ~0.009 rather than 0**. `submit`
  writes the official format so a run can be scored by the benchmark's own
  evaluator rather than only by this harness's arithmetic.

**TCMEval-PA** is 328 items — 297 single-choice, 31 multiple-choice — each
tagged with one of 19 `Rule_ID` families. The distribution decides how much the
graph could possibly help; see below.

## What the knowledge graph actually is

9,350 nodes and 27,972 edges over **14 entity types** and **20 relation
types**, built from national TCM clinical protocols, clinical pathways and the
2025 Chinese Pharmacopoeia. Verified against the declared ontology on every
load (`validate_graph`), and it matches exactly.

| layer | entities |
|---|---|
| clinical organisation | Department (32) |
| disease | Disease (345), DiseaseSubtype (417) |
| differentiation | Syndrome (648), TreatmentPrinciple (1,296) |
| therapy | Formula (798), Herb (816), PatentMedicine (672), ExternalTherapy (1,432) |
| pathway | PathwayStage (1,440) |
| safety | SafetyContext (208), RestrictedItem (228) |
| regulation | PharmacoPoeiaEntry (386) |
| provenance | DocumentSource (632) |

**There is no `Symptom` node and no `Pathogenesis` node, and none was added.**
Symptoms, tongue, pulse and pathogenesis are *transient clinical features* the
model produces at run time. Pathogenesis in particular is treated as a latent
reasoning variable, never as something to look up — which is what makes any
SDT pathogenesis gain attributable to the graph narrowing the reasoning space
rather than to retrieving the answer.

### The finding that shapes the retrieval design

The graph stores no document bodies, so "semantic retrieval over
DocumentSource" cannot mean retrieving passages. But **312 of 648 Syndrome
nodes carry the protocol's verbatim definition sentence** in
`first_mention.sentence`:

> `1.心虚胆怯证：心悸，善惊易恐，坐卧不安，如恐人将捕之，多梦易醒，恶闻声响，食少纳呆。`

That sentence lists main symptoms, tongue and pulse. It is the symptom→syndrome
mapping, latent in the graph rather than absent from it. Retrieval is therefore
built over a **virtual document per entity**, assembled from names, aliases,
the definition sentence, type-specific structured fields, and the evidence
sentences of incident edges — which is also where preparation requirements
(先煎/后下/烊化) actually live.

---

## Design decisions that carry the experiment

### 1. Access domains, enforced at the tool boundary

An SDT agent that could see `Formula` would invert the syndrome→treatment
mapping and recover the answer from the prescription — not the ability SDT
means to measure. So the medication tools are not merely discouraged; they are
**unreachable**, and an attempt is recorded in the trace as an invalid call.

`TreatmentPrinciple` is exposed to SDT for *verification only*: a candidate the
model already produced may be checked against it, but it can never be used to
enumerate candidates.

### 2. No query language

Exposing `run_cypher` would make Cypher-writing skill a per-model confound in
what is meant to be a knowledge comparison. Every model sees the same 13 tools
with byte-identical descriptions, and a test asserts no tool description
mentions a query language.

### 3. Coverage is explicit — silence is not absence

Every tool returns a coverage verdict:

| verdict | meaning |
|---|---|
| `supported` | the graph encodes this, and the answer is grounded in it |
| `partial` | the graph encodes part of it; the gap is named in `caveats` |
| `empty` | the graph encodes this *class* of fact and has none for this query |
| `not_covered` | the graph does not encode this class of fact at all |

The distinction between `empty` and `not_covered` is the whole point. A
prescription-audit system that reports "no contraindication found" when it
never had a contraindication table produces a false negative, and false
negatives in drug safety are the dangerous kind.

### 4. Honest gaps, measured not asserted

`python -m runner.benchmark_runner coverage` audits the 19 PA rule families
against what the graph holds, then projects that onto the released item
distribution. The verdict: **3 grounded, 6 partial, 10 not grounded** — and
weighted by how many questions each family actually carries:

| graph verdict | PA items | share |
|---|---|---|
| not grounded | 166 | 50.6% |
| partial | 113 | 34.5% |
| grounded | 49 | 14.9% |

**Half the released PA set is in families this graph cannot ground at all**, and
the single largest family — A-003, single-herb dosage, 87 items (26.5%) — is
one of them. Any KG effect on PA is bounded to roughly the other half, which is
why the report splits PA accuracy by verdict instead of pooling it. Concretely,
this graph contains

- **no dosage field** — `用法用量` appears 0 times, so `check_dose` returns
  `NOT_COVERED` by construction;
- **no 十八反/十九畏 table** — 0 mentions, so `check_combination` cannot rule
  on incompatibility;
- **no 君臣佐使 role annotation** (0 mentions), no prescription-validity
  metadata (0), no controlled-substance schedule (0);
- **no PatentMedicine composition** (0 of 672), so component-level duplication
  involving a proprietary medicine is undecidable here.

A worked example of why this matters: **半夏 and 附子 are a genuine 十八反
incompatible pair, and they co-occur in three formulas in this graph.** A tool
that reported co-occurrence as evidence of compatibility would confidently
endorse a classical contraindication. `check_combination` therefore returns
`not_covered`, labels co-occurrence as weak compatibility evidence only, and
says the incompatibility table is absent. There is a test for exactly this.

### 4b. The SDT leakage check

The KG conditions hand the model a lookup over the ten named options, which is
only legitimate if the graph recognises distractors and gold answers at the
same rate. Measured across all three splits:

| split | options in graph | **gold** options in graph |
|---|---|---|
| Test | 161/500 (32%) | 25/81 (31%) |
| Validation | 183/500 (37%) | 25/77 (32%) |
| Train | 689/2000 (34%) | 115/335 (34%) |

The rates match, so a model cannot score by picking whichever option the graph
happens to know. Had gold coverage been materially higher, the KG arms would
have been measuring answer leakage rather than reasoning and the option-lookup
tool would have had to be withdrawn. A regression test enforces the gap stays
under 10 points.

### 5. Deterministic checks stay deterministic

`LLM = decides what to look up + integrates + explains`,
`KG = supplies facts`, `Rule engine = makes the decidable calls`.
`verify_tcm_decision` and the five `check_*` tools never call a model, so the
verification signal is identical across models and cannot itself become a
source of measured difference.

### 6. The framework hash

Everything a model could be advantaged by is hashed together: prompts, tool
descriptions and schemas, retrieval weights, top-k, graph hops, context budget,
tool budget, decode parameters, turn limits. `score` **refuses** to pool traces
whose hashes differ, rather than silently averaging incomparable runs.

```
framework_hash = sha256(prompts ‖ tools ‖ retrieval ‖ budgets ‖ decode ‖ limits)
```

---

## What was borrowed from the DeepSeek harness

The [dsh-eval](https://github.com/hccccc01333/dsh-eval) design contributed four
things:

| borrowed | here |
|---|---|
| declarative `benchmark.yaml` | `configs/experiment.*.yaml` — one file fixes models, arms, budgets, decode |
| trace-based metrics | metrics are derived *from* persisted traces, never accumulated during a run, so new metrics apply to old runs |
| keyless replay | `--replay` re-runs entirely from the request cache with no API key; the cache key hashes the full message list, so an edited prompt necessarily misses and replay can never misrepresent a prompt that no longer exists |
| paired A/B compare | `compare` reports win/lose/tie with signed deltas over matched cases |
| scripted grading vs LLM judge as separate layers | deterministic scorers and `SDTJudge` are independent; judge failure degrades to *unscored*, never to a silent zero |

Plus the split that makes iteration affordable: **generation and scoring are
separate phases**, so fixing a scorer costs nothing — no token is re-billed
across 300 cases × 5 models × 5 conditions.

---

## Experimental arms

| arm | structured prompt | KG evidence | model-chosen tools | verification |
|---|---|---|---|---|
| **M0** Base LLM | – | – | – | – |
| **M1** Structured | ✓ | – | – | – |
| **M2** KG-RAG | ✓ | static, deterministic | – | – |
| **M3** KG-Agent | ✓ | agentic | ✓ | – |
| **M4** KG-Agent + Verify | ✓ | agentic | ✓ | ✓ |

M2's retrieval is built from a **rule-based** query (demographics and
administrative fragments stripped identically for every model), so
`Δ(M3−M2)` isolates agentic tool choice rather than confounding it with a
better query. The contrasts decompose as:

```
Δ_structure    = M1 − M0      Δ_retrieval    = M2 − M1
Δ_agent        = M3 − M2      Δ_verification = M4 − M3
```

---

## Quick start

```bash
pip install -e .            # PyYAML is the only runtime dependency

# 1. See what you have: ontology, access domains, tool contract
python -m runner.benchmark_runner inspect

# 2. Audit what the graph can honestly ground
python -m runner.benchmark_runner coverage        # -> docs/kg_coverage.md

# 3. Check your dataset binds before spending tokens
python -m runner.benchmark_runner inspect \
  --dataset data/sdt/Test_TCM_Data_v1.json --dataset-kind sdt

# 4. Run, score, report
export DEEPSEEK_API_KEY=... GEMINI_API_KEY=... OPENAI_API_KEY=...
python -m runner.benchmark_runner run   configs/experiment.sdt.yaml
python -m runner.benchmark_runner score configs/experiment.sdt.yaml
python -m runner.benchmark_runner judge configs/experiment.sdt.yaml   # task-4 quality
python -m runner.benchmark_runner report configs/experiment.sdt.yaml \
                                          configs/experiment.pa.yaml --out runs/report.md

# 5. Emit an official submission file and score it with the benchmark's own script
python -m runner.benchmark_runner submit configs/experiment.sdt.yaml \
                                          --model gpt --condition M3

# Offline: replay a recorded run with no API key
python -m runner.benchmark_runner run configs/experiment.sdt.yaml --replay
```

Runs are **resumable** — an interrupted run picks up from the recorded traces,
matching on `framework_hash`, and every request is cached on the way through.

```bash
python -m unittest discover -s tests -t .    # 102 tests, fully offline
```

The suite passes on a clean checkout with **no benchmark files present**: the
end-to-end tests run against committed synthetic fixtures written in the
released schemas, and the tests that need the real data skip themselves (17 of
102). CI runs it on 3.10 and 3.12 and also re-validates the graph against the
declared ontology.

---

## Research questions this is built to answer

**RQ1** How far apart are frontier models on SDT and PA with no external
knowledge? → M0/M1 across five models.

**RQ2** Does the graph help, and for whom? → `Δ(M0→M3)` per model, exact
McNemar per contrast, Holm–Bonferroni across the family.

**RQ3** Static KG-RAG or dynamic KG-Agent? → `Δ(M2→M3)`, with `Δ(M3→M4)`
separating out verification.

**RQ4** Does the gain differ between knowledge the graph encodes explicitly and
reasoning it only constrains? SDT's four tasks give an unusually clean ladder,
because they are scored separately under one composite:

| target | relationship to the graph |
|---|---|
| PA, grounded families (49 items) | **explicit** — the fact is an entity or an edge |
| PA, ungroundable families (166 items) | **none** — a control: any gain here is not knowledge injection |
| SDT task 3, syndrome | **partial** — syndromes are entities, symptoms are not; 32% of options in graph |
| SDT task 2, pathogenesis | **implicit** — not a graph entity at all |
| SDT task 1, clinical extraction | **none** — a second control, drawn from the case text alone |

The ungroundable PA families and SDT task 1 are the controls that make the rest
interpretable. If the KG arms improve there too, the effect is prompting rather
than knowledge. If accuracy on task 2 improves even though the graph never
encodes a pathogenesis, the mechanism is constraint of the reasoning space — a
stronger result than a lookup win.

**Compensation.** `compensation_table` reports Spearman ρ(base score, Δ) across
models. A negative ρ means the framework compensates weaker base models, which
would be the most useful practical finding available here.

---

## Fairness: what is frozen

Swapping a model changes exactly one thing — `model.generate()`. Frozen and
hashed: prompts, KG, tool set, tool descriptions, retrieval weights and top-k,
graph hops, context budget, tool-call budget, output schema, verifier,
temperature/top-p/seed, dataset, scorer.

The tool-calling protocol is a **uniform text protocol** for every model rather
than each provider's native function-calling API. Native tool calling differs
by provider in schema handling, parallel-call behaviour and error surfaces;
using one text protocol removes that as a confound, at the cost of not
measuring native tool-calling quality. That is the right trade for a knowledge
comparison, and it is a deliberate limitation worth stating in a methods
section.

Formal experiments should run through `benchmark_runner.py`, **not** inside
Codex or Claude Code. Those harnesses contribute their own system prompts and
agent loops, which would confound a five-model comparison. Use them to build,
debug and analyse; use the runner to measure.

---

## Repository layout

```
tcm_kg/       ontology · store · normalisation · hybrid retrieval
tcm_tools/    8 KG tools + 5 deterministic checkers, domain-gated
tcm_agent/    frozen runtime, M0–M4, tasks, prompts (versioned text), traces
tcm_models/   provider adapters, generation cache, keyless replay
tcm_eval/     datasets · scorers · judge · trace metrics · statistics · report
runner/       CLI: run · score · judge · report · compare · inspect · coverage
scripts/      KG coverage audit
configs/      models.yaml + one experiment file per benchmark
kg/           tcm_knowledge_graph.json.gz (the committed artefact)
vendor/       the benchmark's own evaluate.py, vendored unmodified
docs/         kg_coverage.md (generated)
```

Datasets are not committed — see `data/*/README.md` for where to put them.
`tests/fixtures/` holds small synthetic stand-ins in the same schemas so the
suite and the CLI are exercised without them.

## Sources

- [TCMEval-SDT — Scientific Data (2025)](https://www.nature.com/articles/s41597-025-04772-9)
- [TCMEval-PA — Scientific Data (2025)](https://www.nature.com/articles/s41597-025-06387-6)
- TCM-SD — syndrome-differentiation corpus and syndrome knowledge base
- [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) · [dsh-eval](https://github.com/hccccc01333/dsh-eval)
