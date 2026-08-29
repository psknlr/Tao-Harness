# TCMEval-PA

Place the benchmark file here, e.g. `TCMEval-PA.json` (328 items: 297
single-choice, 31 multiple-choice).

Source: *TCMEval-PA: a question-answering benchmark dataset for the
prescription audit of Traditional Chinese Medicine*, Scientific Data (2025),
<https://doi.org/10.1038/s41597-025-06387-6>.

Scoring uses `rule_id` (A-001…A-009, N-001…N-009, C-001) for the per-rule
breakdown that RQ4 turns on. If your copy names that field differently, add the
key to `PA_ALIASES` in `tcm_eval/datasets.py`; without it the per-rule analysis
degrades to an overall accuracy and the explicit-vs-implicit contrast is lost.

Check the binding first:

```bash
python -m runner.benchmark_runner inspect \
  --dataset data/pa/TCMEval-PA.json --dataset-kind pa
```
