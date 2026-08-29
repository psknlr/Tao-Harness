"""End-to-end: the CLI's run -> score -> report -> compare -> replay cycle.

Runs entirely offline against the echo provider, so CI needs no API key and no
network. This is the test that would catch a break in the seam between the
runtime, the trace format and the scorer.
"""

import json
import shutil
import unittest
from pathlib import Path

from runner.benchmark_runner import main
from runner.config import load_experiment
from tcm_agent import read_traces

REPO = Path(__file__).resolve().parent.parent
SDT_CONFIG = "tests/fixtures/experiment.test.yaml"
PA_CONFIG = "tests/fixtures/experiment.pa.test.yaml"

SDT_ANSWER = json.dumps(
    {
        "action": "answer",
        "result": {
            "clinical_information": "心悸；善惊易恐；多梦易醒；食少纳呆",
            "pathogenesis": "心胆气虚，心神失养",
            "syndrome": "心虚胆怯证",
            "explanation": "善惊易恐为心胆气虚之特征表现",
        },
    },
    ensure_ascii=False,
)
PA_ANSWER = json.dumps(
    {
        "action": "answer",
        "result": {
            "rule_category": "特殊煎煮",
            "option_analysis": "石膏为矿物类，需先煎",
            "answer": ["A"],
            "reasoning": "图谱记载石膏先煎",
        },
    },
    ensure_ascii=False,
)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = load_experiment(SDT_CONFIG).output_dir
        if cls.out.exists():
            shutil.rmtree(cls.out)

    def test_01_run_produces_one_trace_per_case_and_condition(self):
        code = main(["run", SDT_CONFIG, "--echo-script", SDT_ANSWER])
        self.assertEqual(code, 0)
        traces = read_traces(self.out / "traces.sdt.echo.jsonl")
        self.assertEqual(len(traces), 2 * 5)  # 2 cases x 5 conditions
        self.assertEqual(len({t.framework_hash for t in traces}), 1)
        self.assertTrue(all(t.final is not None for t in traces))

    def test_02_run_is_resumable(self):
        before = read_traces(self.out / "traces.sdt.echo.jsonl")
        self.assertEqual(main(["run", SDT_CONFIG, "--echo-script", SDT_ANSWER]), 0)
        after = read_traces(self.out / "traces.sdt.echo.jsonl")
        self.assertEqual(len(before), len(after))

    def test_03_score_produces_metrics(self):
        self.assertEqual(main(["score", SDT_CONFIG]), 0)
        path = self.out / "scores.sdt.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 10)
        # case 1's gold syndrome is 心虚胆怯证, case 2's is 痰瘀郁肺证
        by_case = {}
        for row in rows:
            by_case.setdefault(row["case_id"], []).append(row["metrics"]["syndrome_exact"])
        self.assertEqual(set(by_case["sdt_demo_001"]), {1.0})
        self.assertEqual(set(by_case["sdt_demo_002"]), {0.0})

    def test_04_report_renders(self):
        target = self.out / "report.md"
        self.assertEqual(main(["report", SDT_CONFIG, "--out", str(target)]), 0)
        report = target.read_text(encoding="utf-8")
        self.assertIn("## SDT", report)
        self.assertIn("framework_hash", report)
        self.assertIn("M3 KG-Agent", report)

    def test_05_replay_needs_no_key(self):
        # the recorded cache must satisfy a second run with providers disabled
        self.assertEqual(
            main(["run", SDT_CONFIG, "--overwrite", "--replay"]), 0
        )
        traces = read_traces(self.out / "traces.sdt.echo.jsonl")
        self.assertEqual(len(traces), 10)
        self.assertTrue(all(t.final is not None for t in traces), "replay lost answers")

    def test_06_compare_is_paired(self):
        scores = self.out / "scores.sdt.jsonl"
        self.assertEqual(main(["compare", str(scores), str(scores), "--metric", "syndrome_exact"]), 0)

    def test_07_pa_pipeline_runs_and_scores(self):
        self.assertEqual(main(["run", PA_CONFIG, "--echo-script", PA_ANSWER]), 0)
        self.assertEqual(main(["score", PA_CONFIG]), 0)
        rows = [
            json.loads(line)
            for line in (self.out / "scores.pa.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 3 * 2)
        by_case = {r["case_id"]: r["metrics"]["exact"] for r in rows}
        self.assertEqual(by_case["pa_demo_001"], 1.0)  # answer A, gold A
        self.assertEqual(by_case["pa_demo_003"], 0.0)  # answer A, gold AB

    def test_08_mixed_framework_hashes_are_refused(self):
        path = self.out / "traces.sdt.echo.jsonl"
        original = path.read_text(encoding="utf-8")
        try:
            lines = original.splitlines()
            payload = json.loads(lines[0])
            payload["framework_hash"] = "0000000000000000"
            lines[0] = json.dumps(payload, ensure_ascii=False)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(main(["score", SDT_CONFIG]), 2)
        finally:
            path.write_text(original, encoding="utf-8")

    def test_09_inspect_and_coverage_run(self):
        self.assertEqual(main(["inspect", "--dataset", "tests/fixtures/pa_sample.json",
                               "--dataset-kind", "pa"]), 0)
        self.assertEqual(main(["coverage", "--out", str(self.out / "coverage.md")]), 0)


if __name__ == "__main__":
    unittest.main()
