# TCM-SD

A third corpus, distributed separately from TCMEval:

```
data/tcmsd/
  train.json                 43,180 records (JSON Lines despite the extension)
  dev.json                    5,486 records
  syndrome_vocab.txt            148 normalised syndrome labels
  syndrome_knowledge.json     1,027 syndrome definitions
```

Each record is a real EHR extract (`chief_complaint`, `description`,
`detection`) with a `norm_syndrome` label — single-label classification over
148 syndromes, quite different in shape from TCMEval-SDT's four tasks.

Loads through the same interface:

```bash
python -m runner.benchmark_runner inspect \
  --dataset data/tcmsd/dev.json --dataset-kind tcmsd
```

## On `syndrome_knowledge.json`

1,027 syndrome definitions with typical presentations and associated diseases —
i.e. symptom→syndrome text of exactly the kind the clinical knowledge graph
lacks. It is deliberately **not** wired into any headline arm:
`load_syndrome_knowledge` exposes it for a separately-reported ablation.
Folding a second corpus into the KG conditions would confound "what the
knowledge graph adds" with "what more text adds", which is the one question
this study design exists to keep separate.
