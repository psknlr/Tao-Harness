# Experiment release checklist

The audits have converged on the same question in different forms: *can any
reported gain be explained by something other than the intervention?* This is
that question as a list, with the mechanism that answers each item and how to
check it.

Run `python scripts/experiment_manifest.py` first — every number below should
agree with `docs/experiment_manifest.md`, which is generated rather than
written.

## Automated (the suite fails if these break)

| # | claim | mechanism | test |
|---|---|---|---|
| 1 | every case has its own key | builder gate + `enforce_unique_case_ids` | `V5_1_CaseIdentity` |
| 2 | no answer is readable off its own question | builder gate, punctuation- and width-normalised | `V5_1`, `V4_2` |
| 3 | no two options carry the same answer | builder gate on canonical form | `V5_1` |
| 4 | SDT cannot reach treatment entities | domain policy at the tool boundary | `test_tools` |
| 5 | the agent cannot reach the verifier | `ToolPhase` checked in `ToolRegistry.call` | `V5_2_PhaseIsEnforced` |
| 6 | nothing routes on the answer key | PA verifier reads the model's own category | `V3_1` |
| 7 | static and agentic arms hold the same *kinds* of knowledge | CP static context mirrors the tool's relations | `V5_3` |
| 8 | controls spend the same turns, per case | co-generation + per-case parity check | `V5_5`, `V4_4` |
| 9 | M3C and M4 come from one trajectory | `branch_group` required by the contrast | `V5_6` |
| 10 | resume cannot mix runs | `run_signature` + `design_signature` | `V5_5`, `V5_7` |
| 11 | scoring fails closed | model set, design, condition, sample, case set | `V6_3` (below) |
| 12 | point estimate and CI share an estimand | hierarchical macro bootstrap | `V4_6`, `V5_4` |
| 13 | the CP endpoint weights capabilities equally | six families, CP4G excluded | `V6_2` |
| 14 | CP4 gold is this disease's own guideline | disease-specific only | `V6_2` |
| 15 | subtype knowledge is reachable | two-hop traversal | `V6_4` |
| 16 | a crash cannot corrupt a paid run | atomic writes, tolerant reader | `V6_5` |
| 17 | the suite runs from a source archive | no `.git` required | `TestSuiteRunsOnACleanCheckout` |
| 18 | the contamination detector works | planted positives and rewrites | `V6_1_ContaminationDetector` |

## Manual, before generation

- [ ] `python scripts/build_tcm_cp.py --out data/cp/TCM-CP.json` — refuses on any
      invariant violation, so a clean exit *is* the dataset check
- [ ] `benchmark_runner contaminate <config>` for **every** effectiveness
      benchmark. SDT and PA measured 100% / 99.4% clean; anything worse needs
      the clean-subset analysis read before the headline number
- [ ] `benchmark_runner smoke` — every provider answers, holds the JSON
      contract, reports token usage. Fix failures here: an adapter defect
      becomes a per-model confound no paired test removes
- [ ] pin exact model snapshots in `configs/models.yaml`. The IDs shipped are
      placeholders on purpose; filling them in changes the framework hash,
      which is correct — a different registry is a different run
- [ ] `python scripts/experiment_manifest.py` and commit the result
- [ ] freeze: graph, datasets, configs, model snapshots, code. Record the
      commit. The harness's value now is that it does not move

## Manual, before publication

- [ ] read the turn-parity table: calls match per case, token ratios are
      *reported*, and the claim in the text is "turn-matched"
- [ ] read the verification-stratum split: an M4 gain concentrated in
      `not_applicable` is a second-turn effect, not verification
- [ ] read the contamination sensitivity table: the clean-subset delta is the
      one to defend
- [ ] read the PA coverage strata: 10 of 19 rule families are ungroundable, so
      report verification benefit as conditional on coverage
- [ ] label TCM-CP an instrument-capability benchmark everywhere, and never
      pool its contrasts with SDT or PA

## Not claimed

Two things this harness does not establish, stated so a reader does not have
to infer them:

- **TCM-CP does not show clinical correctness.** Its gold comes from the same
  graph the KG arms read. It shows an agent can execute the pathway encoded
  there. Independent expert review of a sample — case realism, answerability,
  gold correctness, distractor plausibility, safety and transition correctness,
  with inter-rater agreement — is the missing evidence, and an expert-built
  test set with no derivation from this graph would be better still.
- **There is no longitudinal patient state.** No `PatientEpisode`, no
  T0→T1→T2 carrying treatment response, monitoring results or variance. CP
  items are independent decisions with the history written into the vignette.
  The accurate description is *graph-grounded, verification-aware clinical
  pathway reasoning*, not *autonomous longitudinal pathway management*.
