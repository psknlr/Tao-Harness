"""Regressions for the V3 review findings.

These target experimental-validity defects rather than crashes: each one is a
way a reviewer could have explained away a reported gain.
"""

import json
import unittest

import tcm_tools  # noqa: F401  (registers the tool surface)
from tcm_agent import AgentRuntime, FrameworkConfig, build_task
from tcm_agent.runtime import BRANCHABLE
from tcm_agent.tasks import case_clauses
from tcm_eval.provenance import case_set_hash, code_fingerprints, compare_fingerprints
from tcm_eval.scorers import consensus_prediction, score_cp
from tcm_eval.stats import cluster_paired_bootstrap, paired_bootstrap
from tcm_kg.schema import Domain
from tcm_models import DecodeParams, Message, build_client, spec_from_config
from tcm_tools.base import REGISTRY

from ._fixtures import graph, retriever

SDT_ITEM = {
    "id": "t",
    "clinical_data": "男，52岁。心悸不安，善惊易恐，多梦易醒。舌淡红，脉细。",
    "pathogenesis_options": {chr(65 + i): f"病机{i}" for i in range(10)},
    "syndrome_options": {
        "A": "心虚胆怯证", "B": "痰火扰心证", "C": "心脾两虚证",
        **{chr(65 + i): f"证候{i}" for i in range(3, 10)},
    },
}


def echo(script):
    return build_client(
        spec_from_config("echo", {"provider": "echo", "model_id": "e"}), script=script
    )


class V3_1_NoOracleRouting(unittest.TestCase):
    """M4 must not learn the rule family from the answer key."""

    def setUp(self):
        self.task = build_task("pa", graph(), retriever())
        self.item = {
            "id": "p", "question": "石膏入汤剂的用法是",
            "options": {"A": "先煎", "B": "后下"}, "rule_id": "N-003",
        }

    def test_routing_follows_the_models_own_category(self):
        plans = self.task.verify_arguments(
            {"answer": ["A"], "rule_category": "特殊煎煮"}, self.item
        )
        self.assertEqual(plans[0]["_tool"], "check_decoction_requirement")

    def test_gold_rule_id_alone_routes_nothing(self):
        # the item still carries rule_id N-003; without the model saying so,
        # no checker may be selected
        self.assertIsNone(self.task.verify_arguments({"answer": ["A"]}, self.item))
        self.assertIsNone(
            self.task.verify_arguments(
                {"answer": ["A"], "rule_category": "无法判断"}, self.item
            )
        )

    def test_a_wrong_self_category_routes_to_the_wrong_checker(self):
        # the model bears the consequence of miscategorising, as it would in
        # deployment; that is the point of removing the oracle
        plans = self.task.verify_arguments(
            {"answer": ["A"], "rule_category": "剂量"}, self.item
        )
        self.assertEqual(plans[0]["_tool"], "check_dose")


class V3_2_ControlsAreProperlyMatched(unittest.TestCase):
    def test_m2c_receives_the_same_static_kg_as_m2(self):
        task = build_task("sdt", graph(), retriever())
        answer = json.dumps(
            {"action": "answer", "result": {"syndrome_answer": ["A"],
                                            "pathogenesis_answer": ["A"],
                                            "clinical_information": ["心悸"], "explanation": "x"}},
            ensure_ascii=False,
        )
        m2 = AgentRuntime(graph(), retriever(), task, echo([answer]), FrameworkConfig()).run(SDT_ITEM, "M2")
        m2c = AgentRuntime(graph(), retriever(), task, echo([answer]), FrameworkConfig()).run(SDT_ITEM, "M2C")
        self.assertGreater(m2.static_context_chars, 0)
        self.assertEqual(m2.static_context_chars, m2c.static_context_chars)
        self.assertEqual(m2c.n_tool_calls, 0, "the control must not use tools")

    def test_controls_are_enabled_in_the_effectiveness_configs(self):
        from runner.config import load_experiment

        for path in ("configs/experiment.sdt.yaml", "configs/experiment.pa.yaml"):
            conditions = load_experiment(path).conditions
            self.assertIn("M2C", conditions, path)
            self.assertIn("M3C", conditions, path)


class V3_3_SharedTrajectory(unittest.TestCase):
    def test_branch_arms_share_one_agent_phase(self):
        task = build_task("sdt", graph(), retriever())
        answer = json.dumps(
            {"action": "answer", "result": {"syndrome_answer": ["A"],
                                            "pathogenesis_answer": ["A"],
                                            "clinical_information": ["心悸"], "explanation": "x"}},
            ensure_ascii=False,
        )
        revision = json.dumps({"syndrome_answer": ["A"], "revision": "unchanged"}, ensure_ascii=False)
        runtime = AgentRuntime(
            graph(), retriever(), task, echo([answer, revision, revision]), FrameworkConfig()
        )
        group = runtime.run_branch_group(SDT_ITEM, ["M3", "M3C", "M4"])
        self.assertEqual(set(group), {"M3", "M3C", "M4"})

        prefixes = {
            c: json.dumps([s.completion.get("text") for s in t.llm_steps[:1]])
            for c, t in group.items()
        }
        self.assertEqual(len(set(prefixes.values())), 1, "branches diverged before verification")
        self.assertNotIn("verification_report", group["M3C"].final)
        self.assertIn("verification_report", group["M4"].final)
        self.assertEqual(group["M3C"].n_llm_calls, group["M4"].n_llm_calls)

    def test_branchable_set_is_declared(self):
        self.assertEqual(BRANCHABLE, ("M3", "M3C", "M4"))


class V3_4_RevisionContext(unittest.TestCase):
    def test_revision_turns_resupply_the_original_item(self):
        task = build_task("sdt", graph(), retriever())
        answer = json.dumps(
            {"action": "answer", "result": {"syndrome_answer": ["A"],
                                            "pathogenesis_answer": ["A"],
                                            "clinical_information": ["心悸"], "explanation": "x"}},
            ensure_ascii=False,
        )
        revision = json.dumps({"syndrome_answer": ["B"], "revision": "revised"}, ensure_ascii=False)
        for condition in ("M3C", "M4"):
            client = echo([answer, revision])
            AgentRuntime(graph(), retriever(), task, client, FrameworkConfig()).run(
                SDT_ITEM, condition
            )
            last = client.seen[-1][-1]["content"]
            self.assertIn("心虚胆怯证", last, f"{condition} revision cannot see the options")
            self.assertIn("心悸不安", last, f"{condition} revision cannot see the case")


class V3_5_VerificationIsolation(unittest.TestCase):
    def test_agents_cannot_call_the_verifier_themselves(self):
        task = build_task("sdt", graph(), retriever())
        runtime = AgentRuntime(graph(), retriever(), task, echo(["{}"]), FrameworkConfig())
        protocol = runtime._tool_protocol()
        self.assertNotIn("verify_tcm_decision", protocol)
        self.assertIn("search_tcm_entities", protocol)

    def test_the_verifier_is_still_reachable_from_the_m4_pass(self):
        task = build_task("sdt", graph(), retriever())
        answer = json.dumps(
            {"action": "answer", "result": {"syndrome_answer": ["A"],
                                            "pathogenesis_answer": ["A"],
                                            "clinical_information": ["心悸"], "explanation": "x"}},
            ensure_ascii=False,
        )
        revision = json.dumps({"syndrome_answer": ["A"], "revision": "unchanged"}, ensure_ascii=False)
        trace = AgentRuntime(
            graph(), retriever(), task, echo([answer, revision]), FrameworkConfig()
        ).run(SDT_ITEM, "M4")
        self.assertIn("verify_tcm_decision", trace.tools_used())

    def test_verifier_evidence_comes_from_the_case_not_the_claim(self):
        task = build_task("sdt", graph(), retriever())
        one = task.verify_arguments(
            {"syndrome_answer": ["A"], "clinical_information": ["捏造甲"]}, SDT_ITEM
        )
        two = task.verify_arguments(
            {"syndrome_answer": ["A"], "clinical_information": ["捏造乙"]}, SDT_ITEM
        )
        self.assertEqual(one[0]["clinical_features"], two[0]["clinical_features"])
        self.assertIn("心悸不安", one[0]["clinical_features"])

    def test_case_clauses_are_deterministic(self):
        text = "男，52岁。诊查：心悸不安，善惊易恐。舌淡红，脉细。"
        self.assertEqual(case_clauses(text), case_clauses(text))
        self.assertIn("心悸不安", case_clauses(text))


class V3_6_Provenance(unittest.TestCase):
    def test_model_fingerprint_tracks_more_than_the_key(self):
        base = spec_from_config("gpt", {"provider": "openai_compat", "model_id": "m", "base_url": "u"})
        other_id = spec_from_config("gpt", {"provider": "openai_compat", "model_id": "n", "base_url": "u"})
        other_body = spec_from_config(
            "gpt", {"provider": "openai_compat", "model_id": "m", "base_url": "u",
                    "extra_body": {"reasoning": "high"}},
        )
        self.assertNotEqual(base.fingerprint(), other_id.fingerprint())
        self.assertNotEqual(base.fingerprint(), other_body.fingerprint())

    def test_cache_key_follows_the_fingerprint(self):
        from tcm_models import request_key

        base = spec_from_config("gpt", {"provider": "openai_compat", "model_id": "m"})
        changed = spec_from_config(
            "gpt", {"provider": "openai_compat", "model_id": "m", "extra_body": {"thinking": True}}
        )
        messages = [Message("user", "x")]
        self.assertNotEqual(
            request_key(base, messages, DecodeParams()),
            request_key(changed, messages, DecodeParams()),
        )

    def test_kg_hash_covers_provenance_fields(self):
        # source_docs feed the SourceEvidence retrieval term, so an unchanged
        # hash must imply unchanged retrieval
        import inspect

        from tcm_kg.store import KGStore

        source = inspect.getsource(KGStore.content_hash)
        self.assertIn("source_docs", source)
        self.assertIn("evidence_type", source)

    def test_case_set_hash_is_order_sensitive(self):
        self.assertNotEqual(case_set_hash(["a", "b"]), case_set_hash(["b", "a"]))
        self.assertEqual(case_set_hash(["a", "b"]), case_set_hash(["a", "b"]))

    def test_code_fingerprints_detect_drift(self):
        current = code_fingerprints()
        self.assertEqual(compare_fingerprints(current, current), [])
        stale = dict(current, scorers_impl_sha256="0" * 64)
        self.assertTrue(compare_fingerprints(stale, current))


class V3_7_ClinicalPathwayBenchmark(unittest.TestCase):
    def test_unsafe_transition_catches_any_unsafe_option(self):
        gold_continue = {"subtask": "CP6_transition_decision", "answer": ["A"]}
        # the mixed answer is the case the first implementation missed
        self.assertEqual(score_cp({"answer": ["A", "C"]}, gold_continue)["unsafe_transition"], 1.0)
        self.assertEqual(score_cp({"answer": ["C"]}, gold_continue)["unsafe_transition"], 1.0)
        self.assertEqual(score_cp({"answer": ["A"]}, gold_continue)["unsafe_transition"], 0.0)

    def test_advancing_on_insufficient_evidence_is_unsafe(self):
        gold = {"subtask": "CP6_transition_decision", "answer": ["D"]}
        self.assertEqual(score_cp({"answer": ["B"]}, gold)["unsafe_transition"], 1.0)
        self.assertEqual(score_cp({"answer": ["D"]}, gold)["unsafe_transition"], 0.0)
        self.assertEqual(score_cp({"answer": ["A"]}, gold)["missed_uncertainty"], 1.0)

    def test_stage_discriminability_filter(self):
        from scripts.build_tcm_cp import _stage_signature, discriminable_stages

        class Stub:
            def __init__(self, name, monitoring):
                self.id, self.name = name, name
                self._m = monitoring

            def get(self, key, default=None):
                return {"monitoring_items": self._m, "entry_criteria": []}.get(key, default)

        twins = [Stub("a", ["x"]), Stub("b", ["x"]), Stub("c", ["x"])]
        self.assertEqual(discriminable_stages(twins), [], "identical stages must be dropped")
        mixed = [Stub("a", ["x"]), Stub("b", ["y"]), Stub("c", ["z"])]
        self.assertEqual(len(discriminable_stages(mixed)), 3)


class V3_8_StatisticsAndSampling(unittest.TestCase):
    def test_cluster_bootstrap_widens_the_interval_for_correlated_items(self):
        import random

        rng = random.Random(0)
        a, b, clusters = [], [], []
        for d in range(30):
            effect = rng.gauss(0.05, 0.12)
            for _ in range(15):
                base = rng.uniform(0.3, 0.7)
                a.append(base)
                b.append(base + effect + rng.gauss(0, 0.01))
                clusters.append(f"d{d}")
        naive = paired_bootstrap(a, b, n_resamples=1500)
        clustered = cluster_paired_bootstrap(a, b, clusters, n_resamples=1500)
        self.assertGreater(
            clustered.ci_high - clustered.ci_low,
            2 * (naive.ci_high - naive.ci_low),
            "clustering did not account for within-disease dependence",
        )
        self.assertIn("cluster_bootstrap", clustered.test)

    def test_sample_index_offsets_the_seed(self):
        import inspect

        from tcm_models.base import LLMClient

        self.assertIn("decode.seed + sample", inspect.getsource(LLMClient.generate))

    def test_consensus_requires_a_real_majority(self):
        # n=2 with one vote each must not behave as a union
        merged = consensus_prediction([{"answer": ["A"]}, {"answer": ["B"]}])
        self.assertEqual(len(merged["answer"]), 1)
        self.assertEqual(consensus_prediction([{"answer": ["A"]}] * 2 + [{"answer": ["B"]}])["answer"], ["A"])


if __name__ == "__main__":
    unittest.main()
