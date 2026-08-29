# TCMEval-SDT

Place the benchmark files here:

- `Train_TCM_Data_v1.json` (200 cases)
- `Validation_TCM_Data_v1.json` (50 cases)
- `Test_TCM_Data_v1.json` (50 cases)

Source: Zhang et al., *TCMEval-SDT: a benchmark dataset for syndrome
differentiation thought of traditional Chinese medicine*, Scientific Data
(2025), <https://doi.org/10.1038/s41597-025-04772-9>. CC-BY 4.0.

The loader binds fields by alias, so English (`TCM_syndrome`,
`clinical_data`, `TCM_pathogenesis`, …) and Chinese (`证候`, `病例`, `中医病机`, …)
key styles both work. Check what it bound before a real run:

```bash
python -m runner.benchmark_runner inspect \
  --dataset data/sdt/Test_TCM_Data_v1.json --dataset-kind sdt
```

If a field reports as MISSING, add its key to `SDT_ALIASES` in
`tcm_eval/datasets.py` rather than renaming the distributed file — the alias
list is the record of what the harness accepts.
