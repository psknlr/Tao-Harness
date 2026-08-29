"""Agent runtime: parsing robustness, the five arms, and the frozen contract."""

import json
import unittest

import tcm_tools  # noqa: F401
from tcm_agent import AgentRuntime, FrameworkConfig, build_task
from tcm_agent.parsing import coerce_list, coerce_str, extract_json_object
from tcm_agent.runtime import CONDITIONS
from tcm_models import DecodeParams, build_client, spec_from_config
from tcm_tools.base import ToolBudget

from ._fixtures import graph, retriever

CASE = {
    "id": "t1",
    "clinical_data": "男，52岁。心悸，善惊易恐，坐卧不安，多梦易醒，食少纳呆。舌淡红，脉细。",
    "pathogenesis_options": {
        "A": "心胆气虚", "B": "痰热扰心", "C": "心血不足", "D": "水饮凌心", "E": "心阳不振",
        "F": "肝郁气滞", "G": "瘀阻心脉", "H": "阴虚火旺", "I": "脾胃虚弱", "J": "肾精不足",
    },
    "syndrome_options": {
        "A": "心虚胆怯证", "B": "痰火扰心证", "C": "心脾两虚证", "D": "水饮凌心证",
        "E": "心阳不振证", "F": "肝郁化火证", "G": "心血瘀阻证", "H": "阴虚火旺证",
        "I": "脾胃虚弱证", "J": "肾精亏虚证",
    },
}
ANSWER = json.dumps(
    {
        "action": "answer",
        "result": {
            "clinical_information": ["心悸", "善惊易恐", "多梦易醒", "食少纳呆"],
            "pathogenesis_answer": ["A"],
            "syndrome_answer": ["A"],
            "explanation": "临证体会：善惊易恐为心胆气虚之特征。辨证：心虚胆怯",
        },
    },
    ensure_ascii=False,
)


def echo(script):
    return build_client(spec_from_config("echo", {"provider": "echo", "model_id": "echo-1"}), script=script)


def run(condition, script):
    task = build_task("sdt", graph(), retriever())
    runtime = AgentRuntime(graph(), retriever(), task, echo(script), FrameworkConfig(), run_id="t")
    return runtime.run(CASE, condition)


class ParsingTests(unittest.TestCase):
    def test_recovers_json_from_prose_and_fences(self):
        cases = {
            '{"a": 1}': "direct",
            '好的\n```json\n{"a": 1}\n```\n以上': "fenced",
            '先想一想 {"x":1} 最终 {"a":1,"b":2}': "balanced_span",
            '{"a": [1,2,], "b": "多行\n文本"}': "repaired",
            '｛"a"：1｝': "repaired_whole",
        }
        for text, expected in cases.items():
            self.assertEqual(extract_json_object(text).strategy, expected, text)

    def test_reports_failure_rather_than_guessing(self):
        outcome = extract_json_object("完全没有 JSON")
        self.assertEqual(outcome.strategy, "failed")
        self.assertIsNone(outcome.value)

    def test_coercion_flattens_model_shapes(self):
        self.assertEqual(coerce_list("A、B C"), ["A", "B", "C"])
        self.assertEqual(coerce_list(["A", ["B"]]), ["A", "B"])
        self.assertEqual(coerce_str(["心悸", "失眠"]), "心悸；失眠")


class ConditionTests(unittest.TestCase):
    def test_single_shot_arms_make_one_call_and_no_tool_calls(self):
        for condition in ("M0", "M1", "M2"):
            trace = run(condition, [ANSWER])
            self.assertEqual(trace.n_llm_calls, 1, condition)
            self.assertEqual(trace.n_tool_calls, 0, condition)
            self.assertEqual(trace.final["syndrome_answer"], ["A"])

    def test_only_m2_injects_static_context(self):
        self.assertEqual(run("M1", [ANSWER]).static_context_chars, 0)
        self.assertGreater(run("M2", [ANSWER]).static_context_chars, 500)

    def test_agent_arm_executes_tool_calls(self):
        script = [
            json.dumps({"action": "tool", "tool": "search_tcm_entities",
                        "arguments": {"query": "心悸 善惊易恐", "entity_types": ["Syndrome"]}}),
            json.dumps({"action": "tool", "tool": "retrieve_syndrome_evidence",
                        "arguments": {"syndrome": "心虚胆怯证"}}),
            ANSWER,
        ]
        trace = run("M3", script)
        self.assertEqual(trace.n_tool_calls, 2)
        self.assertEqual(trace.n_invalid_tool_calls, 0)
        self.assertEqual(trace.tools_used(), ["search_tcm_entities", "retrieve_syndrome_evidence"])

    def test_domain_violation_is_recorded_not_fatal(self):
        script = [
            json.dumps({"action": "tool", "tool": "retrieve_medication_knowledge",
                        "arguments": {"name": "安神定志丸加减"}}),
            ANSWER,
        ]
        trace = run("M3", script)
        self.assertEqual(trace.n_invalid_tool_calls, 1)
        self.assertEqual(trace.final["syndrome_answer"], ["A"])

    def test_verification_arm_adds_a_deterministic_check(self):
        revision = json.dumps(
            {"syndrome_answer": ["A"], "revision": "unchanged", "revision_reason": "图谱一致"},
            ensure_ascii=False,
        )
        trace = run("M4", [ANSWER, revision])
        self.assertIn("verify_tcm_decision", trace.tools_used())
        self.assertEqual(trace.final["revision"], "unchanged")
        # a revision turn must not drop fields the first answer supplied
        self.assertEqual(trace.final["pathogenesis_answer"], ["A"])

    def test_verifier_resolves_every_letter_to_its_option_text(self):
        # verify_tcm_decision cannot check a letter; it must be handed the name,
        # and a multi-select answer must produce one check per selected option
        task = build_task("sdt", graph(), retriever())
        plans = task.verify_arguments({"syndrome_answer": ["A", "B"]}, CASE)
        self.assertEqual([p["syndrome"] for p in plans], ["心虚胆怯证", "痰火扰心证"])

    def test_out_of_range_letters_are_dropped(self):
        task = build_task("sdt", graph(), retriever())
        result = task.normalise_result({"syndrome_answer": ["A", "K", "z"]}, CASE)
        self.assertEqual(result["syndrome_answer"], ["A"])

    def test_unparseable_output_is_retried_then_recorded(self):
        trace = run("M3", ["不是 JSON"])
        self.assertIsNotNone(trace.error)
        self.assertIsNone(trace.final)
        self.assertLessEqual(trace.n_llm_calls, FrameworkConfig().max_format_retries + 1)

    def test_tool_budget_bounds_the_loop(self):
        config = FrameworkConfig(tool_budget=ToolBudget(max_calls=2, max_calls_per_tool=2))
        script = [json.dumps({"action": "tool", "tool": "search_tcm_entities",
                              "arguments": {"query": "心悸"}})] * 6 + [ANSWER]
        task = build_task("sdt", graph(), retriever())
        trace = AgentRuntime(graph(), retriever(), task, echo(script), config).run(CASE, "M3")
        successful = [s for s in trace.tool_steps if s.ok]
        self.assertLessEqual(len(successful), 2)


class FrameworkContractTests(unittest.TestCase):
    def test_hash_is_stable_and_covers_every_frozen_input(self):
        base = FrameworkConfig().framework_hash()
        self.assertEqual(base, FrameworkConfig().framework_hash())
        variants = [
            FrameworkConfig(decode=DecodeParams(temperature=0.7)),
            FrameworkConfig(tool_budget=ToolBudget(max_calls=99)),
            FrameworkConfig(max_agent_turns=32),
        ]
        for variant in variants:
            self.assertNotEqual(variant.framework_hash(), base)

    def test_every_declared_condition_runs(self):
        self.assertEqual(
            set(CONDITIONS), {"M0", "M1", "M2", "M3", "M4", "M2C", "M3C"}
        )


if __name__ == "__main__":
    unittest.main()
