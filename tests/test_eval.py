"""Evaluation layer: dataset binding, scorers, statistics, trace metrics.

Runs against the released files, so a schema change in a re-download fails here
rather than silently producing a column of zeros.
"""

import json
import tempfile
import unittest
from pathlib import Path

from tcm_agent.trace import LLMStep, ToolStep, Trace
from tcm_eval.datasets import (
    load_dataset,
    load_syndrome_knowledge,
    parse_letters,
    parse_options,
    parse_result_file,
)
from tcm_eval.metrics import coverage_honesty, tool_selection_accuracy, trace_metrics
from tcm_eval.scorers import ScoredItem, aggregate, majority_vote, score_pa, score_sdt
from tcm_eval.stats import holm_bonferroni, mcnemar, paired_bootstrap, spearman

REPO = Path(__file__).resolve().parent.parent
SDT_TEST = REPO / "data" / "sdt" / "Test_TCM_Data_v1.json"
SDT_TRAIN = REPO / "data" / "sdt" / "Train_TCM_Data_v1.json"
PA_XLSX = REPO / "data" / "pa" / "TCMEval-PA.xlsx"
TCMSD_DEV = REPO / "data" / "tcmsd" / "dev.json"

has_data = SDT_TEST.exists() and PA_XLSX.exists()


class ParsingTests(unittest.TestCase):
    def test_sdt_option_block_parses(self):
        options = parse_options("A:痰浊;B:耗损心气和心阴;C:食郁于胃")
        self.assertEqual(options["B"], "耗损心气和心阴")
        self.assertEqual(len(options), 3)

    def test_pa_option_block_parses(self):
        options = parse_options("A. 中药名称\nB. 中药数量\nC. 中药的煎法")
        self.assertEqual(options["C"], "中药的煎法")

    def test_answer_letters_parse_from_both_conventions(self):
        self.assertEqual(parse_letters("H;J"), ["H", "J"])   # SDT
        self.assertEqual(parse_letters("ABCD"), ["A", "B", "C", "D"])  # PA
        self.assertEqual(parse_letters(""), [])

    def test_result_file_parses_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.txt"
            path.write_text(
                "﻿病例1@a;b@H;J@B@说明\n病例1@x@Z@Z@重复应被忽略\n", encoding="utf-8"
            )
            gold = parse_result_file(path)
            self.assertEqual(list(gold), ["病例1"])  # BOM stripped, dup ignored
            self.assertEqual(gold["病例1"]["pathogenesis_answer"], "H;J")


@unittest.skipUnless(has_data, "released datasets not present")
class DatasetTests(unittest.TestCase):
    def test_sdt_test_split_binds_and_merges_gold_from_results(self):
        dataset = load_dataset(SDT_TEST, "sdt")
        self.assertEqual(len(dataset), 50)
        # the JSON ships blank answers; the labels come from Results/*.txt
        self.assertEqual(dataset.n_labelled, 50)
        item = dataset.items[0]
        self.assertEqual(item["id"], "病例247")
        self.assertEqual(item["syndrome_letters"], ["J"])
        self.assertEqual(len(item["syndrome_options"]), 10)
        self.assertTrue(item["clinical_information_list"])

    def test_sdt_train_split_carries_its_own_labels(self):
        dataset = load_dataset(SDT_TRAIN, "sdt")
        self.assertEqual(len(dataset), 200)
        item = dataset.items[0]
        self.assertEqual(item["pathogenesis_letters"], ["H", "J"])
        self.assertEqual(item["syndrome_letters"], ["B", "I"])
        self.assertEqual(item["pathogenesis_text"], "热伤肺络;血热不固")

    def test_sdt_gold_letters_index_real_options(self):
        for split in (SDT_TEST, SDT_TRAIN):
            for item in load_dataset(split, "sdt").items:
                for letter in item["syndrome_letters"]:
                    self.assertIn(letter, item["syndrome_options"], item["id"])
                for letter in item["pathogenesis_letters"]:
                    self.assertIn(letter, item["pathogenesis_options"], item["id"])

    def test_pa_workbook_loads_with_the_published_counts(self):
        dataset = load_dataset(PA_XLSX, "pa")
        self.assertEqual(len(dataset), 328)
        multi = sum(1 for i in dataset.items if i["is_multi"])
        self.assertEqual(multi, 31)          # published: 297 single + 31 multi
        self.assertEqual(len(dataset) - multi, 297)
        self.assertTrue(all(i["rule_id"] for i in dataset.items))

    def test_pa_answers_index_real_options(self):
        for item in load_dataset(PA_XLSX, "pa").items:
            for letter in item["answer_letters"]:
                self.assertIn(letter, item["options"], item["id"])

    @unittest.skipUnless(TCMSD_DEV.exists(), "TCM-SD not present")
    def test_tcmsd_loads_as_json_lines(self):
        dataset = load_dataset(TCMSD_DEV, "tcmsd")
        self.assertGreater(len(dataset), 5000)
        item = dataset.items[0]
        self.assertTrue(item["clinical_data"])
        self.assertEqual(len(item["label_space"]), 148)

    @unittest.skipUnless(
        (REPO / "data" / "tcmsd" / "syndrome_knowledge.json").exists(), "KB not present"
    )
    def test_syndrome_knowledge_base_loads(self):
        kb = load_syndrome_knowledge(REPO / "data" / "tcmsd" / "syndrome_knowledge.json")
        self.assertEqual(len(kb), 1027)
        self.assertIn("Definition", kb["风寒袭肺证"])


@unittest.skipUnless(has_data, "released datasets not present")
class SDTScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.item = load_dataset(SDT_TEST, "sdt").items[0]

    def _perfect(self):
        return {
            "clinical_information": self.item["clinical_information_list"],
            "pathogenesis_answer": self.item["pathogenesis_letters"],
            "syndrome_answer": self.item["syndrome_letters"],
            "explanation": self.item["explanation_reference"],
        }

    def test_perfect_answer_scores_one(self):
        scores = score_sdt(self._perfect(), self.item)
        self.assertAlmostEqual(scores["sdt_composite"], 1.0, places=9)
        self.assertEqual(scores["syndrome_exact_set"], 1.0)

    def test_hedging_is_visible_in_the_diagnostics(self):
        # the official rule dilutes rather than forfeits, so selecting every
        # option still scores |gold|/10 -- selection count is what exposes it
        hedged = dict(self._perfect(), syndrome_answer=list("ABCDEFGHIJ"))
        scores = score_sdt(hedged, self.item)
        self.assertAlmostEqual(scores["task3_syndrome"], 0.1)
        self.assertEqual(scores["n_syndrome_selected"], 10.0)
        self.assertEqual(scores["syndrome_exact_set"], 0.0)

    def test_unanswered_case_scores_zero_on_every_task(self):
        scores = score_sdt(None, self.item)
        self.assertEqual(scores["answered"], 0.0)
        self.assertEqual(scores["sdt_composite"], 0.0)


class PAScorerTests(unittest.TestCase):
    def test_single_choice(self):
        self.assertEqual(score_pa({"answer": "A"}, {"answer": "A"})["exact"], 1.0)
        self.assertEqual(score_pa({"answer": "答案是 B"}, {"answer": "B"})["exact"], 1.0)

    def test_multi_choice_partial_credit_rules(self):
        under = score_pa({"answer": ["A"]}, {"answer": "AB"})
        self.assertEqual(under["exact"], 0.0)
        self.assertEqual(under["partial_credit"], 0.5)
        wrong = score_pa({"answer": ["A", "C"]}, {"answer": "AB"})
        self.assertEqual(wrong["partial_credit"], 0.0)
        self.assertEqual(score_pa({"answer": ["B", "A"]}, {"answer": "AB"})["exact"], 1.0)

    def test_contiguous_letter_answers_parse(self):
        self.assertEqual(score_pa({"answer": "ABCDE"}, {"answer": "ABCDE"})["exact"], 1.0)

    def test_majority_vote_across_samples(self):
        self.assertEqual(majority_vote([{"answer": "A"}, {"answer": "A"}, {"answer": "B"}], "answer"), "A")


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.a = [0, 0, 1, 1, 0, 1, 0, 0, 1, 0] * 5
        self.b = [1, 0, 1, 1, 1, 1, 0, 1, 1, 0] * 5

    def test_paired_bootstrap_finds_the_improvement(self):
        result = paired_bootstrap(self.a, self.b, n_resamples=2000)
        self.assertAlmostEqual(result.delta, 0.3, places=6)
        self.assertLess(result.p_value, 0.05)

    def test_mcnemar_counts_only_discordant_pairs(self):
        result = mcnemar(self.a, self.b)
        self.assertEqual((result.wins, result.losses), (15, 0))
        self.assertLess(result.p_value, 0.05)

    def test_identical_vectors_are_not_significant(self):
        self.assertEqual(mcnemar(self.a, self.a).p_value, 1.0)

    def test_holm_is_monotone(self):
        corrected = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.6})
        self.assertTrue(corrected["a"]["reject"])
        self.assertFalse(corrected["b"]["reject"])
        self.assertGreaterEqual(corrected["c"]["p_adjusted"], corrected["b"]["p_adjusted"])

    def test_spearman_detects_the_compensation_pattern(self):
        rho, n = spearman([0.9, 0.8, 0.7, 0.6, 0.5], [0.01, 0.03, 0.05, 0.09, 0.12])
        self.assertEqual(n, 5)
        self.assertLess(rho, -0.9)

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 0], [1])


class TraceMetricTests(unittest.TestCase):
    def _trace(self, coverage, reasoning):
        trace = Trace("r", "c", "pa", "M3", "m", "h")
        trace.tool_steps.append(ToolStep(0, "check_dose", {}, coverage, True, 1.0, 1, result_chars=100))
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
        self.assertEqual(coverage_honesty(self._trace("not_covered", "根据图谱，该剂量合适。")), 0.0)
        self.assertEqual(
            coverage_honesty(self._trace("not_covered", "图谱未收录剂量数据，依据药典知识判断。")), 1.0
        )

    def test_coverage_honesty_is_none_when_nothing_was_uncovered(self):
        self.assertIsNone(coverage_honesty(self._trace("supported", "根据图谱")))

    def test_tool_selection_accuracy_is_rule_aware(self):
        trace = Trace("r", "c", "pa", "M3", "m", "h")
        trace.tool_steps.append(
            ToolStep(0, "check_decoction_requirement", {}, "supported", True, 1.0, 1)
        )
        self.assertEqual(tool_selection_accuracy(trace, "N-003"), 1.0)
        self.assertEqual(tool_selection_accuracy(trace, "A-007"), 0.0)
        self.assertIsNone(tool_selection_accuracy(trace, "UNKNOWN"))


class AggregationTests(unittest.TestCase):
    def test_absent_metric_aggregates_to_none_not_zero(self):
        items = [ScoredItem("a", "sdt", "M0", "m", 0, {"task3_syndrome": 1.0})]
        summary = aggregate(items, ["task3_syndrome", "task2_pathogenesis"])
        self.assertEqual(summary["task3_syndrome"], 1.0)
        self.assertIsNone(summary["task2_pathogenesis"])


if __name__ == "__main__":
    unittest.main()
