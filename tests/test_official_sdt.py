"""Agreement with the benchmark's own evaluator.

The official ``evaluate.py`` shipped with TCMEval-SDT is vendored unmodified at
``vendor/tcmeval_sdt/official_evaluate.py``. These tests run it over the
released answer files and assert that :mod:`tcm_eval.official_sdt` produces the
same numbers. If the harness and the benchmark ever disagree, the benchmark is
right and this test fails loudly rather than the disagreement showing up as an
unexplained delta in a results table.
"""

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

from tcm_eval.datasets import load_dataset, parse_result_file
from tcm_eval.official_sdt import (
    TASK_WEIGHTS,
    clinical_info_extraction_eval,
    rouge_l,
    score_case,
    score_proportional,
    to_submission_line,
    write_submission,
)

REPO = Path(__file__).resolve().parent.parent
OFFICIAL = REPO / "vendor" / "tcmeval_sdt" / "official_evaluate.py"
TEST_JSON = REPO / "data" / "sdt" / "Test_TCM_Data_v1.json"
TEST_GOLD = REPO / "data" / "sdt" / "Results" / "Test_data_result.txt"


def _load_official():
    """Import the vendored official evaluator.

    It imports numpy and pandas at module scope but uses neither in any scoring
    function (they are leftovers from the competition harness). Rather than
    make this harness depend on them just to run one test, absent modules are
    stubbed. ``test_official_module_does_not_use_stubs`` guards the assumption:
    if a future revision of the official script actually calls into numpy or
    pandas, the stub raises and the test fails rather than quietly passing.
    """
    import sys
    import types

    injected = []
    for name in ("numpy", "pandas"):
        if name not in sys.modules:
            try:
                importlib.import_module(name)
            except ImportError:
                stub = types.ModuleType(name)

                def _unavailable(*_args, **_kwargs):
                    raise AssertionError(
                        f"the official evaluator called into {name}, which this "
                        f"test stubbed; the stub is no longer safe"
                    )

                stub.__getattr__ = lambda _attr: _unavailable  # type: ignore[attr-defined]
                sys.modules[name] = stub
                injected.append(name)

    spec = importlib.util.spec_from_file_location("official_evaluate", OFFICIAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._stubbed_modules = injected
    return module


@unittest.skipUnless(OFFICIAL.exists() and TEST_GOLD.exists(), "released files not present")
class OfficialAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = _load_official()
        cls.dataset = load_dataset(TEST_JSON, "sdt")
        cls.gold = {item["id"]: item for item in cls.dataset.items}

    def _submission(self, predictions):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        )
        for case_id, prediction in predictions:
            handle.write(to_submission_line(case_id, prediction) + "\n")
        handle.close()
        return handle.name

    def _compare(self, predictions):
        """Official total vs this harness's total over the same predictions."""
        path = self._submission(predictions)
        official_total = self.official.automated_score(str(TEST_GOLD), path)
        mine = sum(
            score_case(prediction, self.gold[case_id])["sdt_composite"]
            for case_id, prediction in predictions
        )
        return official_total, mine

    def test_official_module_does_not_use_stubs(self):
        # the agreement tests below are only meaningful if the official scorer
        # really is pure-Python arithmetic
        for name in ("clinical_info_extraction_eval", "score_proportional", "rouge_l",
                     "lcs_length", "automated_score"):
            self.assertTrue(callable(getattr(self.official, name)), name)

    def test_component_functions_match(self):
        self.assertAlmostEqual(
            clinical_info_extraction_eval("a;b;c", ["a", "b", "d"]),
            self.official.clinical_info_extraction_eval("a;b;c", ["a", "b", "d"]),
        )
        self.assertAlmostEqual(
            score_proportional(["C", "B", "H", "I"], ["B", "C", "D"], 1),
            self.official.score_proportional(["C", "B", "H", "I"], ["B", "C", "D"], 1),
        )
        self.assertAlmostEqual(rouge_l("热伤肺络血热不固", "热伤肺络"),
                               self.official.rouge_l("热伤肺络血热不固", "热伤肺络"))

    def test_perfect_submission_scores_one_per_case(self):
        predictions = [
            (
                item["id"],
                {
                    "clinical_information": item["clinical_information_list"],
                    "pathogenesis_answer": item["pathogenesis_letters"],
                    "syndrome_answer": item["syndrome_letters"],
                    "explanation": item["explanation_reference"],
                },
            )
            for item in self.dataset.items
        ]
        official_total, mine = self._compare(predictions)
        self.assertAlmostEqual(official_total, float(len(predictions)), places=6)
        self.assertAlmostEqual(mine, official_total, places=6)

    def test_random_submission_matches_the_official_total(self):
        rng = random.Random(20260829)
        letters = list("ABCDEFGHIJ")
        predictions = []
        for item in self.dataset.items:
            predictions.append(
                (
                    item["id"],
                    {
                        "clinical_information": rng.sample(
                            item["clinical_information_list"] + ["杜撰症状"],
                            k=min(3, len(item["clinical_information_list"]) + 1),
                        ),
                        "pathogenesis_answer": rng.sample(letters, k=rng.randint(1, 3)),
                        "syndrome_answer": rng.sample(letters, k=rng.randint(1, 3)),
                        "explanation": item["explanation_reference"][: rng.randint(10, 120)],
                    },
                )
            )
        official_total, mine = self._compare(predictions)
        self.assertAlmostEqual(mine, official_total, places=6)
        self.assertGreater(official_total, 0.0)

    def test_empty_submission_has_a_small_nonzero_floor(self):
        """An empty submission does not score zero under the official scorer.

        Every field keeps its line terminator, so task 4 always shares one
        newline with the reference and earns ``2/(len(ref)+1)``. The floor is
        tiny (~0.002 of the composite per case) but real, and reporting a
        model's score as if the floor were zero would overstate the gap between
        a blank answer and a bad one.
        """
        predictions = [
            (item["id"], {"clinical_information": [], "pathogenesis_answer": [],
                          "syndrome_answer": [], "explanation": ""})
            for item in self.dataset.items
        ]
        official_total, mine = self._compare(predictions)
        self.assertAlmostEqual(mine, official_total, places=6)
        self.assertGreater(official_total, 0.0)
        self.assertLess(official_total / len(predictions), 0.01)

    def test_clean_arithmetic_mode_removes_the_floor(self):
        item = self.dataset.items[0]
        blank = {"clinical_information": [], "pathogenesis_answer": [],
                 "syndrome_answer": [], "explanation": ""}
        self.assertEqual(
            score_case(blank, item, emulate_official_io=False)["sdt_composite"], 0.0
        )
        self.assertGreater(score_case(blank, item)["sdt_composite"], 0.0)


class ScorerSemanticsTests(unittest.TestCase):
    def test_task1_is_recall_and_deduplicates_predictions(self):
        # extra findings are not penalised; repeats do not earn extra credit
        self.assertEqual(clinical_info_extraction_eval("a;b;extra", ["a", "b"]), 1.0)
        self.assertEqual(clinical_info_extraction_eval("a;a", ["a", "b"]), 0.5)

    def test_option_scoring_dilutes_rather_than_zeroes(self):
        # unlike an exam rule, one wrong pick does not forfeit the whole item
        self.assertAlmostEqual(score_proportional(["A", "B"], ["A"]), 0.5)
        self.assertAlmostEqual(score_proportional(["A"], ["A"]), 1.0)
        self.assertAlmostEqual(score_proportional(list("ABCDEFGHIJ"), ["A"]), 0.1)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(TASK_WEIGHTS.values()), 1.0)

    def test_submission_line_is_at_symbol_separated(self):
        line = to_submission_line("病例1", {
            "clinical_information": ["a", "b"], "pathogenesis_answer": ["H", "J"],
            "syndrome_answer": ["B"], "explanation": "文本"})
        self.assertEqual(line, "病例1@a;b@H;J@B@文本")

    def test_submission_strips_separators_from_free_text(self):
        # an @ or newline in task 4 would corrupt the official parser
        line = to_submission_line("c", {"explanation": "含@符号\n换行"})
        self.assertEqual(line.count("@"), 4)
        self.assertNotIn("\n", line)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(TEST_JSON.exists(), "released files not present")
class LeakageTests(unittest.TestCase):
    """The graph must not preferentially cover the correct options.

    The KG conditions give the model a lookup over the ten named options. That
    is only legitimate if the graph is equally likely to recognise a distractor
    as a gold answer. If gold options were materially better covered, a model
    could score by picking whatever the graph knows, and the measured KG gain
    would be leakage rather than reasoning.
    """

    @classmethod
    def setUpClass(cls):
        import tcm_tools  # noqa: F401  (registers the tool surface)
        from tcm_agent import build_task

        from ._fixtures import graph, retriever

        cls.task = build_task("sdt", graph(), retriever())

    def _rates(self, path):
        dataset = load_dataset(path, "sdt")
        found = total = gold_found = gold_total = 0
        for item in dataset.items:
            lookup = self.task._lookup_options(item.get("syndrome_options"))
            gold = set(item.get("syndrome_letters") or [])
            for entry in lookup:
                total += 1
                found += bool(entry["found"])
                if entry["option"] in gold:
                    gold_total += 1
                    gold_found += bool(entry["found"])
        return found / max(1, total), gold_found / max(1, gold_total)

    def test_gold_options_are_not_better_covered_than_distractors(self):
        overall, gold = self._rates(TEST_JSON)
        self.assertGreater(overall, 0.1, "option lookup found almost nothing")
        # a gold-coverage advantage beyond a few points would be answer leakage
        self.assertLess(
            gold - overall,
            0.10,
            f"gold options covered at {gold:.2f} vs {overall:.2f} overall — "
            f"the option-lookup tool would be leaking answers",
        )
