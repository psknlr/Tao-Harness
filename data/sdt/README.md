# TCMEval-SDT

Expected layout (as released):

```
data/sdt/
  Train_TCM_Data_v1.json          200 cases, fully annotated
  Validation_TCM_Data_v1.json      50 cases, answer columns blank
  Test_TCM_Data_v1.json            50 cases, answer columns blank
  Results/
    Validation_data_result.txt     gold labels for the validation split
    Test_data_result.txt           gold labels for the test split
```

Source: *TCMEval-SDT: a benchmark dataset for syndrome differentiation thought
of traditional Chinese medicine*, Scientific Data (2025),
<https://doi.org/10.1038/s41597-025-04772-9>. CC-BY 4.0.

## It is four tasks, not one

`scripts/evaluate.py` in the release scores four tasks under one weighted
composite:

| task | field | form | weight |
|---|---|---|---|
| 1 | `Clinical Information` | `;`-separated findings | 0.2 |
| 2 | `Answers of TCM Pathogenesis` | letters over 10 options | 0.3 |
| 3 | `Answers of TCM Syndrome` | letters over 10 options | 0.4 |
| 4 | `Explanatory Summary` + `Syndrome Differentiation` | free text, ROUGE-L | 0.1 |

Tasks 2 and 3 frequently have several correct options (27 of 50 test cases do
for syndrome).

## The held-out splits ship blank answers

Validation and test carry empty answer columns; the labels exist only in
`Results/*.txt`, in the same `@`-separated format the official evaluator reads.
The loader merges them by `Medical Record ID` automatically — put the `Results/`
directory where it was released and nothing else is needed.

Check the binding before spending tokens:

```bash
python -m runner.benchmark_runner inspect \
  --dataset data/sdt/Test_TCM_Data_v1.json --dataset-kind sdt
```

`records: 50   with gold answers: 50` means the merge worked. If it says
`with gold answers: 0`, the `Results/` file was not found and every score would
come out zero.

## Submitting

`benchmark_runner submit` writes the official `case@t1@t2@t3@t4` format, so a
run can be scored by the benchmark's own `evaluate.py` rather than only by this
harness. A submission covers every case in the split, blank where the run had
no answer.
