# Vendored benchmark scripts

Third-party files, kept **unmodified**, so that this harness can be checked
against the benchmarks' own definitions rather than against our reading of
them.

## `tcmeval_sdt/`

From the TCMEval-SDT release (`scripts/`), CC-BY 4.0:

- `official_evaluate.py` — the benchmark's scoring script. `tcm_eval/official_sdt.py`
  is a re-implementation of its scoring functions, and
  `tests/test_official_sdt.py` runs this original over the released answer
  files to assert the two agree on perfect, random and empty submissions. If
  they ever disagree, the benchmark is right and the test fails loudly.
- `official_generate_options.py` — the ten-option multiple-choice generator,
  kept as the record of how the option pool was constructed (gold answers plus
  distractors sampled from the global pool, shuffled).

Attribution: *TCMEval-SDT: a benchmark dataset for syndrome differentiation
thought of traditional Chinese medicine*, Scientific Data (2025),
<https://doi.org/10.1038/s41597-025-04772-9>.

These scripts import `numpy` and `pandas` at module scope but use neither in
any scoring function. The test stubs them when absent rather than adding them
as harness dependencies, and asserts the stubs are never actually called.
