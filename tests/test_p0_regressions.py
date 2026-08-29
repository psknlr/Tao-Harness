"""Regressions for the P0 defects found in code review.

Each test names the defect it locks down. These are the ones that would have
invalidated conclusions rather than merely crashed, so they are kept together
and stated in terms of the scientific claim each protects.
"""

import json
import unittest

import tcm_tools  # noqa: F401  (registers the tool surface)
from tcm_agent import AgentRuntime, FrameworkConfig, build_task
from tcm_agent.runtime import COMPUTE_MATCHED, CONDITIONS
from tcm_eval.metrics import pathogenesis_probe_rate
from tcm_eval.scorers import consensus_prediction
from tcm_eval.stats import is_binary, paired_test
from tcm_kg.schema import Domain
from tcm_models import build_client, spec_from_config
from tcm_tools.base import REGISTRY

from ._fixtures import graph, retriever


def echo(script):
    return build_client(
        spec_from_config("echo", {"provider": "echo", "model_id": "e"}), script=script
    )


class P0_1_StatisticalTestSelection(unittest.TestCase):
    """The SDT composite is continuous; McNemar on it tests the wrong thing."""

    def test_continuous_metrics_do_not_get_mcnemar(self):
        composite_a = [0.31, 0.45, 0.52, 0.28, 0.60] * 8
        composite_b = [0.40, 0.51, 0.55, 0.39, 0.66] * 8
        result = paired_test(composite_a, composite_b)
        self.assertEqual(result.test, "paired_bootstrap")
        # and it must produce an effect size with an interval, which McNemar cannot
        self.assertEqual(result.ci_low, result.ci_low)  # not NaN
        self.assertGreater(result.delta, 0)

    def test_binary_metrics_still_get_exact_mcnemar(self):
        a = [0, 1, 0, 1, 0] * 8
        b = [1, 1, 0, 1, 1] * 8
        self.assertEqual(paired_test(a, b).test, "mcnemar_exact")

    def test_is_binary_discriminates(self):
        self.assertTrue(is_binary([0.0, 1.0, 1.0]))
        self.assertFalse(is_binary([0.0, 0.5, 1.0]))


class P0_2_PathogenesisLeakage(unittest.TestCase):
    """Task-2 options must never be probed against the graph.

    The study's cleanest claim is that a pathogenesis gain reflects constrained
    reasoning, because the graph holds no pathogenesis entity. Probing the
    options against Syndrome nodes would give a signal correlated with the
    answer and destroy that claim.
    """

    def setUp(self):
        from tcm_eval.datasets import load_dataset

        self.task = build_task("sdt", graph(), retriever())
        try:
            self.item = load_dataset(
                "data/sdt/Test_TCM_Data_v1.json", "sdt"
            ).items[0]
        except FileNotFoundError:
            self.item = {
                "clinical_data": "心悸不安",
                "pathogenesis_options": {"A": "心胆气虚", "B": "痰热扰心"},
                "syndrome_options": {"A": "心虚胆怯证", "B": "痰火扰心证"},
            }

    def test_static_context_contains_no_pathogenesis_lookup(self):
        context = self.task.static_context(self.item)
        for key in context:
            self.assertNotIn(
                "pathogenesis", key, f"{key} exposes a pathogenesis probe"
            )

    def test_lookup_helper_no_longer_accepts_a_pathogenesis_mode(self):
        import inspect

        signature = inspect.signature(self.task._lookup_options)
        self.assertNotIn("as_syndrome", signature.parameters)

    def test_agent_option_probing_is_measurable(self):
        from tcm_agent.trace import ToolStep, Trace

        options = {"A": "心胆气虚", "B": "痰热扰心"}
        trace = Trace("r", "c", "sdt", "M3", "m", "h")
        trace.tool_steps.append(
            ToolStep(0, "search_tcm_entities", {"query": "心悸 善惊易恐"}, "supported", True, 1.0, 3)
        )
        self.assertEqual(pathogenesis_probe_rate(trace, options), 0.0)
        trace.tool_steps.append(
            ToolStep(1, "search_tcm_entities", {"query": "心胆气虚"}, "supported", True, 1.0, 1)
        )
        self.assertEqual(pathogenesis_probe_rate(trace, options), 0.5)


class P0_3_and_4_Verification(unittest.TestCase):
    """M4 must verify the whole answer, and must actually verify for PA."""

    def _sdt_item(self):
        return {
            "id": "t",
            "clinical_data": "心悸，善惊易恐",
            "pathogenesis_options": {chr(65 + i): f"病机{i}" for i in range(10)},
            "syndrome_options": {
                "A": "心虚胆怯证", "B": "痰火扰心证", "C": "心脾两虚证",
                **{chr(65 + i): f"证候{i}" for i in range(3, 10)},
            },
        }

    def test_every_selected_syndrome_is_verified(self):
        item = self._sdt_item()
        task = build_task("sdt", graph(), retriever())
        plans = task.verify_arguments({"syndrome_answer": ["A", "B", "C"]}, item)
        self.assertEqual(len(plans), 3)
        self.assertEqual(
            [p["syndrome"] for p in plans], ["心虚胆怯证", "痰火扰心证", "心脾两虚证"]
        )

    def test_m4_runs_one_check_per_selected_option(self):
        item = self._sdt_item()
        answer = json.dumps(
            {"action": "answer", "result": {"syndrome_answer": ["A", "B"],
                                            "pathogenesis_answer": ["A"],
                                            "clinical_information": ["心悸"],
                                            "explanation": "x"}},
            ensure_ascii=False,
        )
        revision = json.dumps({"syndrome_answer": ["A", "B"], "revision": "unchanged"}, ensure_ascii=False)
        trace = AgentRuntime(
            graph(), retriever(), build_task("sdt", graph(), retriever()),
            echo([answer, revision]), FrameworkConfig(),
        ).run(item, "M4")
        verifications = [s for s in trace.tool_steps if s.tool == "verify_tcm_decision"]
        self.assertEqual(len(verifications), 2)

    def test_pa_verification_runs_a_real_checker(self):
        item = {
            "id": "p1",
            "question": "石膏入汤剂的用法是",
            "options": {"A": "先煎", "B": "后下", "C": "包煎", "D": "冲服"},
            "rule_id": "N-003",
        }
        task = build_task("pa", graph(), retriever())
        plans = task.verify_arguments({"answer": ["A"]}, item)
        self.assertIsNotNone(plans, "PA M4 produced no deterministic check")
        self.assertEqual(plans[0]["_tool"], "check_decoction_requirement")
        # the claim must be passed through, or the checker adjudicates nothing
        self.assertEqual(plans[0]["claimed_requirement"], "先煎")

    def test_pa_falls_back_to_coverage_audit_when_no_checker_applies(self):
        item = {"id": "p2", "question": "处方审核的定义是", "options": {"A": "x"}, "rule_id": "C-001"}
        task = build_task("pa", graph(), retriever())
        self.assertIsNone(task.verify_arguments({"answer": ["A"]}, item))


class P0_8_and_9_Reproducibility(unittest.TestCase):
    def test_framework_hash_separates_tasks_and_domains(self):
        base = FrameworkConfig(task="sdt", domain="clinical_reasoning")
        other_task = FrameworkConfig(task="pa", domain="prescription_safety")
        self.assertNotEqual(base.framework_hash(), other_task.framework_hash())

    def test_framework_hash_tracks_kg_and_dataset_content(self):
        base = FrameworkConfig(task="sdt", kg_hash="a", dataset_hash="b")
        self.assertNotEqual(
            base.framework_hash(),
            FrameworkConfig(task="sdt", kg_hash="CHANGED", dataset_hash="b").framework_hash(),
        )
        self.assertNotEqual(
            base.framework_hash(),
            FrameworkConfig(task="sdt", kg_hash="a", dataset_hash="CHANGED").framework_hash(),
        )

    def test_kg_content_hash_is_semantic_and_stable(self):
        kg = graph()
        self.assertEqual(kg.content_hash(), kg.content_hash())
        self.assertEqual(len(kg.content_hash()), 64)


class P0_10_SelfConsistency(unittest.TestCase):
    def test_options_are_voted_per_option(self):
        samples = [
            {"syndrome_answer": ["A", "C"]},
            {"syndrome_answer": ["A"]},
            {"syndrome_answer": ["A", "B"]},
        ]
        merged = consensus_prediction(samples)
        # A appears in 3/3, B and C in 1/3 each
        self.assertEqual(merged["syndrome_answer"], ["A"])

    def test_never_returns_an_empty_answer_when_samples_answered(self):
        merged = consensus_prediction([{"answer": ["A"]}, {"answer": ["B"]}])
        self.assertTrue(merged["answer"])

    def test_all_unanswered_returns_none(self):
        self.assertIsNone(consensus_prediction([None, None]))


class P0_11_JudgeSchema(unittest.TestCase):
    def test_judge_reads_the_four_task_schema(self):
        from tcm_eval.judge import _letters_to_text

        options = {"A": "心虚胆怯证", "J": "阴寒内盛"}
        self.assertEqual(_letters_to_text(["J"], options), "阴寒内盛")
        self.assertEqual(_letters_to_text(["A", "J"], options), "心虚胆怯证；阴寒内盛")

    def test_judge_does_not_regrade_the_official_tasks(self):
        from tcm_eval.judge import JUDGE_FIELDS

        self.assertNotIn("pathogenesis", JUDGE_FIELDS)
        self.assertNotIn("syndrome", JUDGE_FIELDS)


class P0_12_ComputeMatchedControls(unittest.TestCase):
    def test_control_arms_are_declared(self):
        self.assertIn("M2C", CONDITIONS)
        self.assertIn("M3C", CONDITIONS)
        self.assertEqual(COMPUTE_MATCHED, {"M2C": "M3", "M3C": "M4"})

    def test_m2c_gets_extra_turns_and_no_graph_access(self):
        item = {"id": "t", "clinical_data": "心悸",
                "pathogenesis_options": {"A": "x"}, "syndrome_options": {"A": "y"}}
        think = json.dumps({"action": "think", "note": "梳理"}, ensure_ascii=False)
        answer = json.dumps(
            {"action": "answer", "result": {"syndrome_answer": ["A"], "pathogenesis_answer": ["A"],
                                            "clinical_information": ["心悸"], "explanation": "x"}},
            ensure_ascii=False,
        )
        trace = AgentRuntime(
            graph(), retriever(), build_task("sdt", graph(), retriever()),
            echo([think, think, answer]), FrameworkConfig(),
        ).run(item, "M2C")
        self.assertEqual(trace.n_tool_calls, 0, "the compute control touched the graph")
        self.assertGreater(trace.n_llm_calls, 1, "the control got no extra compute")
        self.assertEqual(trace.final["syndrome_answer"], ["A"])

    def test_m3c_revises_without_verification_evidence(self):
        item = {"id": "t", "clinical_data": "心悸",
                "pathogenesis_options": {"A": "x"}, "syndrome_options": {"A": "心虚胆怯证"}}
        answer = json.dumps(
            {"action": "answer", "result": {"syndrome_answer": ["A"], "pathogenesis_answer": ["A"],
                                            "clinical_information": ["心悸"], "explanation": "x"}},
            ensure_ascii=False,
        )
        revision = json.dumps({"syndrome_answer": ["A"], "revision": "unchanged"}, ensure_ascii=False)
        task = build_task("sdt", graph(), retriever())
        sham = AgentRuntime(graph(), retriever(), task, echo([answer, revision]), FrameworkConfig()).run(item, "M3C")
        real = AgentRuntime(graph(), retriever(), task, echo([answer, revision]), FrameworkConfig()).run(item, "M4")
        self.assertEqual(sham.n_llm_calls, real.n_llm_calls, "controls must match on compute")
        self.assertNotIn("verification_report", sham.final)
        self.assertIn("verification_report", real.final)


class P0_5_6_7_ClinicalPathway(unittest.TestCase):
    def test_pathway_domain_exists_and_sdt_stays_isolated(self):
        pathway_tools = REGISTRY.names_for(Domain.PATHWAY)
        self.assertIn("retrieve_pathway_stage", pathway_tools)
        self.assertIn("evaluate_pathway_transition", pathway_tools)
        self.assertIn("retrieve_treatment_plan", pathway_tools)
        clinical = REGISTRY.names_for(Domain.CLINICAL)
        for leaky in ("retrieve_treatment_plan", "retrieve_medication_knowledge"):
            self.assertNotIn(leaky, clinical, "SDT lost its treatment isolation")

    def test_stage_tool_exposes_the_action_fields(self):
        from tcm_tools.base import ToolBudget, ToolContext

        kg = graph()
        disease = next(
            d for d in kg.of_type("Disease")
            if any(
                s.get("day_actions")
                for _e, s in kg.neighbours(d.id, {"HAS_PATHWAY_STAGE"})
            )
        )
        ctx = ToolContext(kg, retriever(), Domain.PATHWAY, ToolBudget())
        result = REGISTRY.call("retrieve_pathway_stage", ctx, {"disease": disease.name})
        self.assertTrue(result.ok)
        stage = result.data["stages"][0]
        self.assertIn("day_actions", stage)
        self.assertIn("next_stages", stage)

    def test_transition_evaluator_traverses_and_respects_criteria(self):
        from tcm_tools.base import ToolBudget, ToolContext

        kg = graph()
        stage = next(
            s for s in kg.of_type("PathwayStage")
            if s.get("exit_criteria")
        )
        ctx = ToolContext(kg, retriever(), Domain.PATHWAY, ToolBudget())

        met = REGISTRY.call(
            "evaluate_pathway_transition", ctx,
            {"stage_id": stage.id, "findings": [str(c) for c in stage.get("exit_criteria")]},
        )
        self.assertIn(met.data["recommendation"], {"exit", "advance"})

        unmet = REGISTRY.call(
            "evaluate_pathway_transition", ctx,
            {"stage_id": stage.id, "findings": ["症状无改善", "病情反复"]},
        )
        self.assertEqual(unmet.data["recommendation"], "continue")

        # no findings must never yield a discharge recommendation
        silent = REGISTRY.call("evaluate_pathway_transition", ctx, {"stage_id": stage.id})
        self.assertEqual(silent.data["recommendation"], "insufficient_evidence")

    def test_unsafe_transition_is_scored(self):
        from tcm_eval.scorers import score_cp

        gold = {"subtask": "CP6_transition_decision", "answer": ["A"]}  # continue
        self.assertEqual(score_cp({"answer": ["C"]}, gold)["unsafe_transition"], 1.0)
        self.assertEqual(score_cp({"answer": ["A"]}, gold)["unsafe_transition"], 0.0)


if __name__ == "__main__":
    unittest.main()
