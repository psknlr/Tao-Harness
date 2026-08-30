"""Regressions for the V7 audit, plus the MiniMax and Poe adapters.

The contamination tests are the ones that matter. The V6 audit reported SDT
100% clean and PA 99.4% clean; those numbers were produced by a detector that
indexed about half the reachable text, measured overlap in one direction only
and concatenated multi-select gold before matching. Each defect has a test
here that fails against the V6 code.
"""

import json
import unittest
from pathlib import Path
from typing import Any, List

import tcm_tools  # noqa: F401
from tcm_kg import load_kg
from tcm_kg.schema import EdgeType, NodeType
from tcm_models.base import DecodeParams, LLMError, Message, ModelSpec, RetryableError

REPO = Path(__file__).resolve().parent.parent

_KG: List[Any] = []
_GRAPH: List[Any] = []


def kg():
    if not _KG:
        _KG.append(load_kg())
    return _KG[0]


def graph_text():
    if not _GRAPH:
        from tcm_eval.contamination import GraphText

        _GRAPH.append(GraphText.from_kg(kg()))
    return _GRAPH[0]


def audit(text, cid="probe", opts=None, letters=None):
    from tcm_eval.contamination import audit_case

    return audit_case(
        {"id": cid, "clinical_data": text, "options": opts or {"A": "x"},
         "answer_letters": letters or ["A"]},
        graph_text(),
        "sdt",
    )


class V7_1_CorpusCoversWhatAnAgentReads(unittest.TestCase):
    """The corpus must be reachability, not convenience."""

    def test_no_node_attribute_text_is_outside_the_corpus(self):
        from tcm_eval.contamination import normalise

        covered = {normalise(p) for p in graph_text().passages}

        def leaves(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from leaves(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from leaves(item)

        missing = 0
        checked = 0
        for node in kg().nodes.values():
            for value in (node.attrs or {}).values():
                for text in leaves(value):
                    key = normalise(text)
                    if len(key) < 5:
                        continue
                    checked += 1
                    if key not in covered:
                        missing += 1
        self.assertGreater(checked, 10000)
        self.assertEqual(missing, 0, f"{missing} reachable strings outside the corpus")

    def test_a_pathway_stage_field_is_findable(self):
        """99.8% of PathwayStage text used to be invisible to the audit."""
        stage = next(
            n for n in kg().nodes.values()
            if str(n.type) == "PathwayStage" and n.attrs.get("monitoring_items")
        )
        item = str(stage.attrs["monitoring_items"][0])
        row = audit("患者今日查房。", "stage", {"A": item}, ["A"])
        self.assertTrue(row["answer_exact_in_graph"])
        self.assertEqual(row["stratum"], "likely")

    def test_a_pharmacopoeia_field_is_findable(self):
        """100% of PharmacoPoeiaEntry text used to be invisible -- PA reads it."""
        herb = next(
            (n for n in kg().nodes.values()
             if isinstance(n.attrs.get("pharmacopoeia_entry"), dict)
             and n.attrs["pharmacopoeia_entry"].get("pharmacopoeial_functions")),
            None,
        )
        if herb is None:
            self.skipTest("no pharmacopoeia functions in the graph")
        text = str(herb.attrs["pharmacopoeia_entry"]["pharmacopoeial_functions"])
        row = audit("处方审核。", "pharm", {"A": text}, ["A"])
        self.assertTrue(row["answer_exact_in_graph"])

    def test_the_corpus_is_built_from_what_retrieval_indexes(self):
        """Tied to virtual_document so the audit cannot fall behind retrieval."""
        origins = {o.split(":")[0] for o in graph_text().origins}
        self.assertIn("vdoc", origins)
        self.assertIn("node", origins)
        self.assertIn("edge", origins)


class V7_2_BidirectionalContainment(unittest.TestCase):
    def test_a_case_that_is_an_excerpt_of_a_graph_passage_is_caught(self):
        passage = max((p for p in graph_text().passages if len(p) > 300), key=len)
        for n in (40, 60, 90):
            with self.subTest(length=n):
                row = audit(passage[15 : 15 + n], f"excerpt{n}")
                self.assertEqual(row["case_in_graph"], 1.0)
                self.assertEqual(row["stratum"], "likely")

    def test_both_directions_are_reported_separately(self):
        """They answer different questions and either being high is a leak."""
        passage = max((p for p in graph_text().passages if len(p) > 200), key=len)
        row = audit(passage[10:70], "dir")
        self.assertIn("graph_in_case", row)
        self.assertIn("case_in_graph", row)
        self.assertGreater(row["case_in_graph"], row["graph_in_case"])

    def test_containment_helper_is_asymmetric(self):
        from tcm_eval.contamination import containment, ngrams

        short, long = ngrams("舌淡红苔薄白脉细弱"), ngrams("患者五十二岁，舌淡红苔薄白脉细弱，余无异常。")
        self.assertEqual(containment(short, long), 1.0)
        self.assertLess(containment(long, short), 1.0)


class V7_3_PartWiseGoldMatching(unittest.TestCase):
    def test_multi_select_gold_is_matched_part_by_part(self):
        """Two options each verbatim in the graph used to report clean."""
        a, b = [p for p in graph_text().passages if 30 < len(p) < 80][:2]
        row = audit("患者主诉不适。", "multi", {"A": a, "B": b}, ["A", "B"])
        self.assertEqual(row["n_gold_parts"], 2)
        self.assertEqual(row["n_gold_parts_exact_in_graph"], 2)
        self.assertEqual(row["stratum"], "likely")

    def test_one_leaked_option_among_several_is_enough(self):
        a = next(p for p in graph_text().passages if 30 < len(p) < 80)
        row = audit("患者主诉不适。", "one", {"A": a, "B": "完全无关的选项文本内容"}, ["A", "B"])
        self.assertEqual(row["n_gold_parts_exact_in_graph"], 1)
        self.assertEqual(row["stratum"], "likely")

    def test_a_clean_multi_select_case_stays_clean(self):
        row = audit(
            "The quick brown fox. " * 6, "cleanmulti",
            {"A": "Entirely unrelated option one", "B": "Entirely unrelated option two"},
            ["A", "B"],
        )
        self.assertEqual(row["n_gold_parts_exact_in_graph"], 0)
        self.assertEqual(row["stratum"], "clean")


class V7_4_AuditIdentity(unittest.TestCase):
    def test_the_identity_covers_everything_that_invalidates_an_audit(self):
        from tcm_eval.contamination import audit_identity, identity_conflicts

        base = dict(kg_hash="k", dataset_hash="d", case_set_hash="c", gold_hash="g")
        first = audit_identity(**base)
        for field, value in (
            ("kg_hash", "k2"), ("dataset_hash", "d2"),
            ("case_set_hash", "c2"), ("gold_hash", "g2"),
        ):
            with self.subTest(field):
                other = audit_identity(**{**base, field: value})
                self.assertTrue(identity_conflicts(first, other), f"{field} not covered")

    def test_an_identical_identity_has_no_conflicts(self):
        from tcm_eval.contamination import audit_identity, identity_conflicts

        base = dict(kg_hash="k", dataset_hash="d", case_set_hash="c", gold_hash="g")
        self.assertEqual(
            identity_conflicts(audit_identity(**base), audit_identity(**base)), []
        )

    def test_the_audit_version_is_part_of_the_identity(self):
        from tcm_eval.contamination import AUDIT_VERSION, audit_identity

        identity = audit_identity(kg_hash="k", dataset_hash="d", case_set_hash="c")
        self.assertEqual(identity["audit_version"], AUDIT_VERSION)

    def test_the_report_does_not_overclaim(self):
        from tcm_eval.contamination import format_report

        text = format_report({
            "dataset": "sdt", "n_cases": 1, "n_graph_passages": 10,
            "strata": {"clean": 1, "possible": 0, "likely": 0},
            "share_clean": 1.0, "n_answer_exact": 0,
            "thresholds": {"likely": 0.4, "possible": 0.2, "ngram": 5},
            "cases": [],
        })
        self.assertIn("less consistent with", text)
        self.assertIn("not proof", text)
        self.assertNotIn("cannot be explained as retrieving", text)


class V7_5_ResolvedGoldIsFrozen(unittest.TestCase):
    def test_the_loader_reports_the_gold_file_it_discovered(self):
        from tcm_eval.datasets import load_dataset

        path = REPO / "data" / "sdt" / "Test_TCM_Data_v1.json"
        if not path.exists():
            self.skipTest("SDT dataset not present")
        dataset = load_dataset(path, "sdt")
        self.assertIsNotNone(dataset.gold_path, "discovered gold was not recorded")
        self.assertTrue(Path(dataset.gold_path).exists())

    def test_the_gold_path_survives_subsetting(self):
        from tcm_eval.datasets import load_dataset

        path = REPO / "data" / "sdt" / "Test_TCM_Data_v1.json"
        if not path.exists():
            self.skipTest("SDT dataset not present")
        dataset = load_dataset(path, "sdt")
        self.assertEqual(dataset.subset(5).gold_path, dataset.gold_path)

    def test_the_gold_hash_is_immutable_in_the_manifest(self):
        from tcm_eval.provenance import IMMUTABLE_MANIFEST_KEYS

        self.assertIn("dataset_results_sha256", IMMUTABLE_MANIFEST_KEYS)


class V7_6_SubtypeKnowledgeEndToEnd(unittest.TestCase):
    def test_syndrome_contexts_works_when_entered_at_a_subtype(self):
        """syndromes_of returned empty for all 116 such subtypes."""
        subtypes = [
            n for n in kg().of_type(NodeType.DISEASE_SUBTYPE.value)
            if kg().out_edges(n.id, {EdgeType.SUBTYPE_HAS_SYNDROME.value})
        ]
        self.assertTrue(subtypes)
        empty = [s.name for s in subtypes if not kg().syndrome_contexts(s.id)]
        self.assertEqual(empty[:3], [], f"{len(empty)} subtypes still return nothing")

    def test_a_subtype_entered_directly_knows_its_parent_disease(self):
        subtype = next(
            n for n in kg().of_type(NodeType.DISEASE_SUBTYPE.value)
            if kg().out_edges(n.id, {EdgeType.SUBTYPE_HAS_SYNDROME.value})
        )
        rows = kg().syndrome_contexts(subtype.id)
        self.assertTrue(all(r["disease"] for r in rows))
        self.assertTrue(all(r["via"] == "subtype" for r in rows))

    def test_contexts_are_not_deduplicated_by_syndrome(self):
        """One syndrome under two subtypes is two clinical facts, not one."""
        for disease in kg().of_type(NodeType.DISEASE.value):
            rows = kg().syndrome_contexts(disease.id)
            by_syndrome: dict = {}
            for row in rows:
                by_syndrome.setdefault(row["syndrome_id"], set()).add(row["subtype"])
            if any(len(v) > 1 for v in by_syndrome.values()):
                self.assertGreater(len(rows), len(by_syndrome))
                return
        self.skipTest("no syndrome spans two subtypes in this graph")

    def test_the_clinical_tool_shows_a_subtype_its_syndromes(self):
        from tcm_kg.index import KGRetriever
        from tcm_tools.base import REGISTRY, ToolBudget, ToolContext
        from tcm_kg.schema import Domain

        retriever = KGRetriever(kg())
        retriever.warm()
        found = kg().find_by_name("消渴病肾病（糖尿病肾病）", [NodeType.DISEASE.value])
        if not found:
            self.skipTest("disease absent from the graph")
        result = REGISTRY.call(
            "retrieve_clinical_context",
            ToolContext(kg(), retriever, Domain.CLINICAL, ToolBudget()),
            {"entity": found[0].name},
        )
        subtypes = (result.data or {}).get("subtypes") or []
        self.assertTrue(subtypes)
        self.assertTrue(
            any(s.get("syndromes") for s in subtypes),
            "every subtype was shown with an empty syndrome list",
        )

    def test_the_static_context_reaches_subtype_syndromes(self):
        """Else M1/M2/M2C see nothing where M3's tools see knowledge."""
        import inspect

        from tcm_agent import tasks

        self.assertIn("syndrome_contexts", inspect.getsource(tasks))


class V7_MiniMaxAdapter(unittest.TestCase):
    """MiniMax reports API errors with HTTP 200."""

    def setUp(self):
        import tcm_models.providers as providers

        self.providers = providers
        self.spec = ModelSpec(
            key="minimax", provider="minimax", model_id="m",
            base_url="https://x/v1", api_key_env="",
        )
        self._original = providers.post_json

    def tearDown(self):
        self.providers.post_json = self._original

    def _respond(self, payload):
        self.providers.post_json = lambda *a, **k: payload

    def _call(self):
        return self.providers.MiniMaxClient(self.spec)._generate(
            [Message("user", "hi")], DecodeParams()
        )

    def test_an_error_delivered_with_http_200_is_not_a_blank_answer(self):
        self._respond({
            "base_resp": {"status_code": 1004, "status_msg": "invalid api key"},
            "choices": [{"message": {"content": ""}}],
        })
        with self.assertRaises(LLMError):
            self._call()

    def test_a_rate_limit_is_retryable_rather_than_fatal(self):
        self._respond({
            "base_resp": {"status_code": 1002, "status_msg": "rate limited"},
            "choices": [{"message": {"content": ""}}],
        })
        with self.assertRaises(RetryableError):
            self._call()

    def test_a_total_only_usage_report_is_recorded_not_guessed(self):
        self._respond({
            "base_resp": {"status_code": 0}, "usage": {"total_tokens": 412},
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        })
        usage = self._call().usage
        self.assertEqual(usage.total_tokens, 412)
        self.assertEqual(usage.prompt_tokens, 0)  # visibly incomplete, not invented

    def test_a_normal_response_is_untouched(self):
        self._respond({
            "base_resp": {"status_code": 0},
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        })
        usage = self._call().usage
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens), (100, 20))


class V7_PoeAdapter(unittest.TestCase):
    def test_a_gateway_answer_is_marked_as_one(self):
        import tcm_models.providers as providers

        original = providers.post_json
        try:
            providers.post_json = lambda *a, **k: {
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            }
            spec = ModelSpec(key="poe", provider="poe", model_id="GPT-4o",
                             base_url="https://api.poe.com/v1", api_key_env="")
            completion = providers.PoeClient(spec)._generate(
                [Message("user", "hi")], DecodeParams()
            )
            self.assertEqual(completion.raw.get("via_gateway"), "poe")
        finally:
            providers.post_json = original

    def test_both_providers_are_registered(self):
        from tcm_models.registry import PROVIDERS

        self.assertIn("minimax", PROVIDERS)
        self.assertIn("poe", PROVIDERS)

    def test_the_shipped_config_uses_the_dedicated_adapters(self):
        from runner.config import load_models

        specs = load_models(REPO / "configs" / "models.yaml")
        self.assertEqual(specs["minimax"].provider, "minimax")
        self.assertEqual(specs["poe"].provider, "poe")


class V7_ModelFingerprint(unittest.TestCase):
    def test_retry_policy_is_part_of_the_apparatus(self):
        """max_retries=1 vs 10 changes the unanswered rate, which scores zero."""
        base = dict(key="m", provider="openai_compat", model_id="x",
                    base_url="https://x", api_key_env="")
        one = ModelSpec(**base, max_retries=1)
        ten = ModelSpec(**base, max_retries=10)
        self.assertNotEqual(one.fingerprint(), ten.fingerprint())
        slow = ModelSpec(**base, timeout_s=600.0)
        self.assertNotEqual(ModelSpec(**base).fingerprint(), slow.fingerprint())


if __name__ == "__main__":
    unittest.main()
