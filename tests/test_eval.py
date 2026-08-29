"""Evaluation layer: dataset binding, scorers, statistics, trace metrics."""

import json
import tempfile
import unittest
from pathlib import Path

from tcm_agent.trace import LLMStep, ToolStep, Trace
from tcm_eval.datasets import load_dataset
from tcm_eval.metrics import coverage_honesty, tool_selection_accuracy, trace_metrics
from tcm_eval.scorers import ScoredItem, aggregate, majority_vote, score_pa, score_sdt
from tcm_eval.stats import holm_bonferroni, mcnemar, paired_bootstrap, spearman

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class DatasetTests(unittest.TestCase):
    def test_english_schema_binds(self):
        dataset = load_dataset(FIXTURES / "sdt_sample.json", "sdt")
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.mapping.bound["syndrome"], "TCM_syndrome")
        self.assertEqual(dataset.items[0]["syndrome"], "心虚胆怯证")

    def test_chinese_schema_binds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cn.json"
            path.write_text(
                json.dumps([{"编号": "1", "病例": "心悸", "证候": "心虚胆怯证",
                             "中医病机": "心胆气虚", "临床信息": "心悸", "辨证分析": "x"}],
                           ensure_ascii=False),
                encoding="utf-8",
            )
            dataset = load_dataset(path, "sdt")
            self.assertEqual(dataset.items[0]["syndrome"], "心虚胆怯证")
            self.assertEqual(dataset.mapping.bound["id"], "编号")

    def test_pa_options_normalise_from_list_or_dict(self):
        dataset = load_dataset(FIXTURES / "pa_sample.json", "pa")
        self.assertEqual(dataset.items[0]["options"]["A"], "石膏")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pa.json"
            path.write_text(
                json.dumps([{"id": "x", "question": "q",
                             "options": ["A. 甲", "B. 乙"], "answer": "A"}], ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertEqual(load_dataset(path, "pa").items[0]["options"], {"A": "甲", "B": "乙"})


class SDTScorerTests(unittest.TestCase):
    def test_exact_match_ignores_enumeration_and_punctuation(self):
        result = score_sdt({"syndrome": "1.心虚胆怯证"}, {"syndrome": "心虚胆怯证"})
        self.assertEqual(result["syndrome_exact"], 1.0)

    def test_compound_syndrome_earns_atom_credit_but_not_exact(self):
        result = score_sdt({"syndrome": "痰阻血瘀证"}, {"syndrome": "痰阻血瘀，湿郁化热证"})
        self.assertEqual(result["syndrome_exact"], 0.0)
        self.assertAlmostEqual(result["syndrome_atom_f1"], 2 / 3, places=3)

    def test_unanswered_case_scores_zero_not_missing(self):
        result = score_sdt(None, {"syndrome": "心虚胆怯证", "pathogenesis": "x"})
        self.assertEqual(result["answered"], 0.0)
        self.assertEqual(result["syndrome_exact"], 0.0)
        self.assertEqual(result["pathogenesis_f1"], 0.0)

    def test_step_absent_from_gold_is_not_scored(self):
        # scoring an unannotated reference as a perfect match would inflate
        # every model equally and hide the split's real coverage
        result = score_sdt({"syndrome": "A", "pathogenesis": "x"}, {"syndrome": "A"})
        self.assertNotIn("pathogenesis_f1", result)


class PAScorerTests(unittest.TestCase):
    def test_single_choice(self):
        self.assertEqual(score_pa({"answer": "A"}, {"answer": "A"})["exact"], 1.0)
        self.assertEqual(score_pa({"answer": "答案是 B"}, {"answer": "B"})["exact"], 1.0)

    def test_multi_choice_partial_credit_rules(self):
        under = score_pa({"answer": ["A"]}, {"answer": "AB"})
        self.assertEqual(under["exact"], 0.0)
        self.assertEqual(under["partial_credit"], 0.5)
        wrong = score_pa({"answer": ["A", "C"]}, {"answer": "AB"})
        self.assertEqual(wrong["partial_credit"], 0.0)  # any wrong option forfeits
        exact = score_pa({"answer": ["B", "A"]}, {"answer": "AB"})
        self.assertEqual(exact["exact"], 1.0)

    def test_majority_vote_across_samples(self):
        votes = [{"answer": "A"}, {"answer": "A"}, {"answer": "B"}]
        self.assertEqual(majority_vote(votes, "answer"), "A")


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.a = [0, 0, 1, 1, 0, 1, 0, 0, 1, 0] * 5
        self.b = [1, 0, 1, 1, 1, 1, 0, 1, 1, 0] * 5

    def test_paired_bootstrap_finds_the_improvement(self):
        result = paired_bootstrap(self.a, self.b, n_resamples=2000)
        self.assertAlmostEqual(result.delta, 0.3, places=6)
        self.assertLess(result.p_value, 0.05)
        self.assertLess(result.ci_low, result.delta < result.ci_high)

    def test_mcnemar_counts_only_discordant_pairs(self):
        result = mcnemar(self.a, self.b)
        self.assertEqual(result.wins, 15)
        self.assertEqual(result.losses, 0)
        self.assertLess(result.p_value, 0.05)

    def test_identical_vectors_are_not_significant(self):
        self.assertEqual(mcnemar(self.a, self.a).p_value, 1.0)

    def test_holm_is_monotone_and_controls_the_family(self):
        corrected = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.6})
        self.assertTrue(corrected["a"]["reject"])
        self.assertFalse(corrected["b"]["reject"])
        self.assertGreaterEqual(corrected["c"]["p_adjusted"], corrected["b"]["p_adjusted"])

    def test_spearman_detects_the_compensation_pattern(self):
        rho, n = spearman([0.9, 0.8, 0.7, 0.6, 0.5], [0.01, 0.03, 0.05, 0.09, 0.12])
        self.assertEqual(n, 5)
        self.assertLess(rho, -0.9)  # weaker base model, larger gain

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 0], [1])


class TraceMetricTests(unittest.TestCase):
    def _trace(self, coverage, reasoning):
        trace = Trace("r", "c", "pa", "M3", "m", "h")
        trace.tool_steps.append(
            ToolStep(0, "check_dose", {}, coverage, True, 1.0, 1, result_chars=100)
        )
        trace.llm_steps.append(
            LLMStep(0, "answer", 10, {"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        )
        trace.final = {"answer": ["A"], "reasoning": reasoning}
        return trace

    def test_metrics_are_derived_from_the_trace(self):
        metrics = trace_metrics(self._trace("supported", "x"))
        self.assertEqual(metrics["n_tool_calls"], 1)
        self.assertEqual(metrics["total_tokens"], 15)
        self.assertEqual(metrics["tool_success_rate"], 1.0)

    def test_coverage_honesty_penalises_unearned_graph_claims(self):
        dishonest = self._trace("not_covered", "根据图谱，该剂量合适。")
        self.assertEqual(coverage_honesty(dishonest), 0.0)
        honest = self._trace("not_covered", "图谱未收录剂量数据，依据药典知识判断。")
        self.assertEqual(coverage_honesty(honest), 1.0)

    def test_coverage_honesty_is_none_when_nothing_was_uncovered(self):
        self.assertIsNone(coverage_honesty(self._trace("supported", "根据图谱")))

    def test_tool_selection_accuracy_is_rule_aware(self):
        trace = Trace("r", "c", "pa", "M3", "m", "h")
        trace.tool_steps.append(ToolStep(0, "check_decoction_requirement", {}, "supported", True, 1.0, 1))
        self.assertEqual(tool_selection_accuracy(trace, "N-003"), 1.0)
        self.assertEqual(tool_selection_accuracy(trace, "A-007"), 0.0)
        self.assertIsNone(tool_selection_accuracy(trace, "UNKNOWN"))


class AggregationTests(unittest.TestCase):
    def test_absent_metric_aggregates_to_none_not_zero(self):
        items = [ScoredItem("a", "sdt", "M0", "m", 0, {"syndrome_exact": 1.0})]
        summary = aggregate(items, ["syndrome_exact", "pathogenesis_f1"])
        self.assertEqual(summary["syndrome_exact"], 1.0)
        self.assertIsNone(summary["pathogenesis_f1"])


if __name__ == "__main__":
    unittest.main()
