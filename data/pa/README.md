# TCMEval-PA

Place `TCMEval-PA.xlsx` here (the released workbook; `.json` also works).

Source: *TCMEval-PA: a question-answering benchmark dataset for the
prescription audit of Traditional Chinese Medicine*, Scientific Data (2025),
<https://doi.org/10.1038/s41597-025-06387-6>.

328 items — 297 single-choice, 31 multiple-choice — with columns `ID`,
`Question`, `Candidate Answers`, `Answer`, `Explanation`, `Category`,
`Rule_ID`, `Rule_summary`. The workbook is read with a stdlib-only parser
(`tcm_eval/xlsx.py`), so no pandas or openpyxl is required.

`Rule_ID` drives the per-rule analysis that RQ4 turns on. The released
distribution, against what this knowledge graph can ground:

| rule | items | graph verdict |
|---|---|---|
| A-003 single-herb dosage | 87 | not grounded |
| A-007 contraindications | 64 | partial |
| N-003 special decoction | 36 | **grounded** |
| A-001 appropriateness concepts | 35 | not grounded |
| A-005 administration | 20 | partial |
| … | | |

Pooled: **166 items (51%) not grounded, 113 partial, 49 grounded.** Report PA
accuracy split by verdict; pooling hides the effect. Regenerate the full table
with `python -m runner.benchmark_runner coverage`.

```bash
python -m runner.benchmark_runner inspect \
  --dataset data/pa/TCMEval-PA.xlsx --dataset-kind pa
```
