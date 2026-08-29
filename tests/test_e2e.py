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

#: A fixed answer for 病例247 (gold: pathogenesis D, syndrome J), so the scored
#: numbers below are checkable by hand against the released answer file.
SDT_ANSWER = json.dumps(
    {
        "action": "answer",
        "result": {
            "clinical_information": ["咳逆不能平卧", "唾白色泡沫痰", "短气"],
            "pathogenesis_answer": ["D"],
            "syndrome_answer": ["J"],
            "explanation": "临证体会：少阴伤寒，阴寒内盛。辨证：少阴伤寒，阴寒内盛",
        },
    },
    ensure_ascii=False,
)
PA_ANSWER = json.dumps(
    {
        "action": "answer",
        "result": {
            "rule_category": "基础概念",
            "option_analysis": "逐项分析",
            "answer": ["E"],
            "reasoning": "图谱未收录该法规条文，依据自身知识判断",
        },
    },
    ensure_ascii=False,
)
N_SDT_CASES = 4
N_PA_CASES = 6


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
        self.assertEqual(len(traces), N_SDT_CASES * 5)
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
        self.assertEqual(len(rows), N_SDT_CASES * 5)
        # 病例247's gold answers are pathogenesis D and syndrome J, which the
        # scripted answer matches exactly; every other case it gets wrong
        by_case = {}
        for row in rows:
            by_case.setdefault(row["case_id"], []).append(row["metrics"])
        hit = by_case["病例247"][0]
        self.assertEqual(hit["task2_pathogenesis"], 1.0)
        self.assertEqual(hit["task3_syndrome"], 1.0)
        self.assertIn("sdt_composite", hit)

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
        self.assertEqual(len(traces), N_SDT_CASES * 5)
        self.assertTrue(all(t.final is not None for t in traces), "replay lost answers")

    def test_06_compare_is_paired(self):
        scores = self.out / "scores.sdt.jsonl"
        self.assertEqual(
            main(["compare", str(scores), str(scores), "--metric", "sdt_composite"]), 0
        )

    def test_06b_submission_file_is_official_format(self):
        target = self.out / "sub.txt"
        self.assertEqual(
            main(["submit", SDT_CONFIG, "--model", "echo", "--condition", "M3",
                  "--out", str(target)]), 0
        )
        lines = [l for l in target.read_text(encoding="utf-8").splitlines() if l]
        # a submission must cover every case in the split, not only those run
        self.assertEqual(len(lines), 50)
        self.assertTrue(all(line.count("@") == 4 for line in lines))
        self.assertTrue(lines[0].startswith("病例247@"))

    def test_07_pa_pipeline_runs_and_scores(self):
        self.assertEqual(main(["run", PA_CONFIG, "--echo-script", PA_ANSWER]), 0)
        self.assertEqual(main(["score", PA_CONFIG]), 0)
        rows = [
            json.loads(line)
            for line in (self.out / "scores.pa.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), N_PA_CASES * 2)
        by_case = {r["case_id"]: r["metrics"] for r in rows}
        self.assertEqual(by_case["1"]["exact"], 1.0)   # gold E, answered E
        self.assertEqual(by_case["2"]["exact"], 0.0)   # gold D, answered E
        self.assertEqual(by_case["1"]["rule_id"], "C-001")

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
        self.assertEqual(
            main(["inspect", "--dataset", "data/pa/TCMEval-PA.xlsx", "--dataset-kind", "pa"]), 0
        )
        self.assertEqual(main(["coverage", "--out", str(self.out / "coverage.md")]), 0)


if __name__ == "__main__":
    unittest.main()
