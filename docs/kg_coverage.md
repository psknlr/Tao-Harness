# Knowledge-graph coverage audit

Every figure below is computed from the delivered graph by `scripts/kg_coverage.py`; regenerate with `python -m runner.benchmark_runner coverage`.

## Graph at a glance

```json
{
  "n_nodes": 9350,
  "n_edges": 27972,
  "node_types": {
    "Department": 32,
    "Disease": 345,
    "DiseaseSubtype": 417,
    "DocumentSource": 632,
    "ExternalTherapy": 1432,
    "Formula": 798,
    "Herb": 816,
    "PatentMedicine": 672,
    "PathwayStage": 1440,
    "PharmacoPoeiaEntry": 386,
    "RestrictedItem": 228,
    "SafetyContext": 208,
    "Syndrome": 648,
    "TreatmentPrinciple": 1296
  },
  "edge_types": {
    "ALIAS_OF": 274,
    "BELONGS_TO_DEPARTMENT": 346,
    "CAUTION_FOR": 73,
    "CITES_DOCUMENT": 632,
    "CONTAINS_HERB": 9681,
    "CONTRAINDICATED_FOR": 269,
    "DERIVED_FROM": 170,
    "HAS_PATHWAY_STAGE": 1440,
    "HAS_SUBTYPE": 417,
    "HAS_SYNDROME": 1298,
    "NEXT_STAGE": 1118,
    "PROCESSED_FROM": 95,
    "REGISTERED_IN_PHARMACOPOEIA": 729,
    "SAME_AS": 364,
    "SUBTYPE_HAS_SYNDROME": 271,
    "TREATED_BY_PRINCIPLE": 1433,
    "USES_EXTERNAL_THERAPY": 6685,
    "USES_FORMULA": 1145,
    "USES_HERB_DIRECT": 258,
    "USES_PATENT_MEDICINE": 1274
  },
  "n_documents": 632,
  "n_identity_clusters": 222,
  "content_hash": "c94e3dd59d776323"
}
```

## What the graph can and cannot ground

Of the 19 TCMEval-PA rule families: **3 grounded**, **6 partial**, **10 not grounded**.

A rule marked *not grounded* is not a defect to be papered over. The corresponding tool returns `NOT_COVERED` and says so, and the model is instructed to answer from its own knowledge while stating that the graph gave no support. Returning a confident "no problem found" for a dosage or compatibility question the graph never encoded would be a false negative in a safety system.

| rule | title | verdict | evidence in this graph | tool |
|---|---|---|---|---|
| A-001 | 处方适宜性概念 | not grounded | DocumentSource 只存元数据（标题/类型/科室/版本），不存条文正文。 | — |
| A-002 | 用药与病名/证型相符 | grounded | HAS_SYNDROME 1298、TREATED_BY_PRINCIPLE 1433、USES_FORMULA 1145、USES_PATENT_MEDICINE 1274 条边完整支撑该链路。 | retrieve_clinical_context, retrieve_medication_knowledge |
| A-003 | 单味药剂量 | not grounded | 药典条目 386 条，无用法用量字段（“用法用量”出现 0 次）。 | check_dose → NOT_COVERED |
| A-004 | 总剂量/药味数量 | partial | 药味数可数（791/798 方有组成，中位 11 味）；总剂量无数据。 | retrieve_medication_knowledge |
| A-005 | 用法合理 | partial | 74/816 味药有原文煎法括注；服法（每日几剂、饭前后）无数据。 | check_decoction_requirement |
| A-006 | 品种选择 | grounded | 药典条目含基原部位、性味归经、功能主治与别名；222 个别名/炮制品聚类可做品种归一。 | retrieve_pharmacopeia_entry |
| A-007 | 使用禁忌 | partial | 禁忌边以饮食调护为主：{'herb': 1, 'other_restriction': 88, 'diet': 195, 'procedure': 58}。这是诊疗方案的调护禁忌，不是药品说明书禁忌症。 | retrieve_safety_constraints, check_restricted_item |
| A-008 | 重复用药 | partial | 汤剂可判（组成 + 222 个别名聚类）；中成药不可判（0/672 有组成）。 | check_duplicate_medication |
| A-009 | 联合用药/配伍禁忌 | not grounded | “十八反/十九畏”出现 0 次；无配伍禁忌表。（半夏与附子在图谱方剂中共现 3 次——正说明共现不可当作安全性证据。） | check_combination → NOT_COVERED |
| N-001 | 处方完整性 | not grounded | 图谱不含处方实体，无科别/年龄/临床诊断等处方字段。 | — |
| N-002 | 君臣佐使 | not grounded | “君臣佐使/君药”出现 0 次；CONTAINS_HERB 边不带角色标注。 | — |
| N-003 | 特殊煎煮 | grounded | 74 味药有原文括注，按位置归属，不会把同句其他药材的括注错记。 | check_decoction_requirement |
| N-004 | 剂量单位 | not grounded | 无剂量数据即无单位数据。 | — |
| N-005 | 处方用量 | not grounded | 无用量数据。 | — |
| N-006 | 处方效期 | not grounded | “处方效期/有效期”出现 0 次。 | — |
| N-007 | 特殊药品 | partial | 药典有毒标注 27/386 条；“毒性/麻醉/精神药品”管理术语出现 0 次，无分级管理数据。 | retrieve_pharmacopeia_entry |
| N-008 | 开具规范 | not grounded | 图谱不含法规条文正文。 | — |
| N-009 | 新生儿/婴幼儿 | partial | “新生儿/婴幼儿”出现 4 次；儿科相关 SafetyContext 5 个，妊娠相关 18 个；无儿童剂量折算规则。 | retrieve_safety_constraints |
| C-001 | 基本概念 | not grounded | 同 A-001。 | — |

## Consequences for the study design

- **SDT anchoring.** 312 of 648 Syndrome nodes (48%) carry the protocol's verbatim definition sentence, which lists main symptoms, tongue and pulse. That sentence — not any Symptom entity — is what a case description can actually match against, and it is why the retrieval index is built over per-entity virtual documents rather than over entity names.
- **The other 336 syndromes** are name-only. `retrieve_syndrome_evidence` reports `PARTIAL` for these so a model can weigh them accordingly.
- **PA is the explicit-knowledge probe** only for the grounded and partial families. Reporting PA accuracy split by verdict is the cleanest test of RQ4: if the KG gain concentrates in the grounded families, the mechanism is knowledge injection rather than a general prompting effect.
- **No entity was added to the ontology to suit either benchmark.** There is no Symptom node and no Pathogenesis node. Pathogenesis is treated as a latent reasoning variable produced by the model, which makes any SDT pathogenesis gain attributable to the graph narrowing the reasoning space rather than to retrieving the answer.


## Where the gaps land on the released PA set

328 items, by rule family:

| rule | graph verdict | items | share |
|---|---|---|---|
| A-003 | not grounded | 87 | 26.5% |
| A-007 | partial | 64 | 19.5% |
| N-003 | grounded | 36 | 11.0% |
| A-001 | not grounded | 35 | 10.7% |
| A-005 | partial | 20 | 6.1% |
| N-007 | partial | 14 | 4.3% |
| N-001 | not grounded | 11 | 3.4% |
| N-002 | not grounded | 10 | 3.0% |
| A-006 | grounded | 9 | 2.7% |
| A-004 | partial | 9 | 2.7% |
| C-001 | not grounded | 6 | 1.8% |
| N-006 | not grounded | 6 | 1.8% |
| N-005 | not grounded | 5 | 1.5% |
| A-002 | grounded | 4 | 1.2% |
| A-008 | partial | 4 | 1.2% |
| A-009 | not grounded | 4 | 1.2% |
| N-009 | partial | 2 | 0.6% |
| N-008 | not grounded | 1 | 0.3% |
| N-004 | not grounded | 1 | 0.3% |

Pooled by verdict:

| graph verdict | items | share |
|---|---|---|
| not grounded | 166 | 50.6% |
| partial | 113 | 34.5% |
| grounded | 49 | 14.9% |

**51% of released PA items fall in rule families this graph cannot ground at all.** The single largest family, A-003 (single-herb dosage, 87 items), is one of them. Any KG effect on PA is therefore bounded to roughly the remaining half, and PA results should be reported split by verdict rather than pooled.


## SDT option coverage, and the leakage check

| split | cases | options in graph | **gold** options in graph |
|---|---|---|---|
| Test_TCM_Data_v1 | 50 | 161/500 (32%) | 25/81 (31%) |
| Validation_TCM_Data_v1 | 50 | 183/500 (37%) | 25/77 (32%) |
| Train_TCM_Data_v1 | 200 | 689/2000 (34%) | 115/335 (34%) |

The two rates are the point. They match closely, which means the graph is **not** biased toward the correct options: a model cannot score by picking whichever option the graph happens to recognise. Had the gold rate been materially higher, the KG conditions would have been measuring answer leakage rather than reasoning, and the option-lookup tool would have had to be withdrawn.


## Measured facts

```json
{
  "n_syndromes": 648,
  "n_syndromes_with_definition": 312,
  "n_herbs": 816,
  "n_herbs_with_preparation": 74,
  "n_formulas": 798,
  "n_formulas_with_composition": 791,
  "n_pharmacopoeia_entries": 386,
  "n_toxicity_flagged": 27,
  "n_patent_medicines": 672,
  "n_patent_with_composition": 0,
  "n_identity_clusters": 222,
  "safety_edge_kinds": {
    "herb": 1,
    "other_restriction": 88,
    "diet": 195,
    "procedure": 58
  },
  "n_pregnancy_contexts": 18,
  "n_paediatric_contexts": 5,
  "mentions_junchen": 0,
  "mentions_shibafan": 0,
  "mentions_dosage_field": 0,
  "mentions_validity": 0,
  "mentions_controlled": 0,
  "mentions_neonate": 4
}
```
