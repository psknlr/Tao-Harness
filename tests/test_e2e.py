"""End-to-end: the CLI's run -> score -> report -> submit -> replay cycle.

Runs entirely offline against the echo provider and against committed synthetic
fixtures written in the released schemas, so this passes on a clean checkout
with no API key, no network and no benchmark files. That matters: the datasets
are gitignored, and a suite that only passed on a machine where someone had
already downloaded them would be no use in CI or to a reviewer.

This is the test that catches a break in the seam between the runtime, the
trace format and the scorer.
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

#: A fixed answer matching fixture 案例001 exactly (gold: pathogenesis A,
#: syndrome A) and wrong for the other two, so every scored number below is
#: checkable by hand against tests/fixtures/sdt_mini/Results/.
SDT_ANSWER = json.dumps(
    {
        "action": "answer",
        "result": {
            "clinical_information": ["心悸不安", "善惊易恐", "多梦易醒"],
            "pathogenesis_answer": ["A"],
            "syndrome_answer": ["A"],
            "explanation": "临证体会：心胆气虚，心神失养，故善惊易恐。辨证：心虚胆怯",
        },
    },
    ensure_ascii=False,
)
PA_ANSWER = json.dumps(
    {
        "action": "answer",
        "result": {
            "rule_category": "特殊煎煮",
            "option_analysis": "逐项分析",
            "answer": ["A"],
            "reasoning": "图谱记载石膏先煎",
        },
    },
    ensure_ascii=False,
)
N_SDT_CASES = 3
N_PA_CASES = 4


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
        # 案例001's gold answers are pathogenesis A and syndrome A, which the
        # scripted answer matches exactly; the other two cases it gets wrong
        by_case = {}
        for row in rows:
            by_case.setdefault(row["case_id"], []).append(row["metrics"])
        hit = by_case["案例001"][0]
        self.assertEqual(hit["task2_pathogenesis"], 1.0)
        self.assertEqual(hit["task3_syndrome"], 1.0)
        self.assertGreater(hit["sdt_composite"], 0.7)
        miss = by_case["案例002"][0]
        self.assertEqual(miss["task3_syndrome"], 0.0)

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
        self.assertEqual(len(lines), N_SDT_CASES)
        self.assertTrue(all(line.count("@") == 4 for line in lines))
        self.assertTrue(lines[0].startswith("案例001@"))

    def test_07_pa_pipeline_runs_and_scores(self):
        self.assertEqual(main(["run", PA_CONFIG, "--echo-script", PA_ANSWER]), 0)
        self.assertEqual(main(["score", PA_CONFIG]), 0)
        rows = [
            json.loads(line)
            for line in (self.out / "scores.pa.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), N_PA_CASES * 2)
        by_case = {r["case_id"]: r["metrics"] for r in rows}
        self.assertEqual(by_case["m1"]["exact"], 1.0)   # gold A, answered A
        self.assertEqual(by_case["m4"]["exact"], 0.0)   # gold C, answered A
        self.assertEqual(by_case["m1"]["rule_id"], "N-003")

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

    def test_08b_mixed_run_signatures_are_refused(self):
        """The framework hash is equal across models; the signature is not."""
        path = self.out / "traces.sdt.echo.jsonl"
        original = path.read_text(encoding="utf-8")
        try:
            lines = original.splitlines()
            payload = json.loads(lines[0])
            payload["run_signature"] = "0000000000000000"
            lines[0] = json.dumps(payload, ensure_ascii=False)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(main(["score", SDT_CONFIG]), 2)
            # ...and --allow-drift scores it anyway, with the drift recorded
            self.assertEqual(main(["score", SDT_CONFIG, "--allow-drift"]), 0)
            scored = json.loads(
                (self.out / "scored_manifest.sdt.json").read_text(encoding="utf-8")
            )
            self.assertTrue(scored["allow_drift"])
            self.assertTrue(any("run signatures" in d for d in scored["drift"]))
        finally:
            path.write_text(original, encoding="utf-8")
            main(["score", SDT_CONFIG])  # restore a clean scored manifest

    def test_08c_traces_from_another_run_are_not_resumed(self):
        """Resuming on the framework hash kept traces a resume must regenerate."""
        path = self.out / "traces.sdt.echo.jsonl"
        original = path.read_text(encoding="utf-8")
        try:
            lines = original.splitlines()
            for i, line in enumerate(lines):
                payload = json.loads(line)
                payload["run_signature"] = "not-this-run"
                lines[i] = json.dumps(payload, ensure_ascii=False)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(main(["run", SDT_CONFIG, "--echo-script", SDT_ANSWER]), 0)
            traces = read_traces(path)
            self.assertEqual(len(traces), N_SDT_CASES * 5, "traces were duplicated")
            self.assertFalse(
                any(t.run_signature == "not-this-run" for t in traces),
                "a trace from a different run survived the resume",
            )
        finally:
            path.write_text(original, encoding="utf-8")

    def test_08d_a_manifest_for_another_experiment_is_not_overwritten(self):
        manifest_path = self.out / "manifest.sdt.json"
        original = manifest_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(original)
            payload["dataset_sha256"] = "0" * 64
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaises(SystemExit) as caught:
                main(["run", SDT_CONFIG, "--echo-script", SDT_ANSWER])
            self.assertIn("dataset_sha256", str(caught.exception))
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_sha256"],
                "0" * 64,
                "the conflicting manifest was overwritten",
            )
            # --new-run archives it rather than losing it
            self.assertEqual(
                main(["run", SDT_CONFIG, "--new-run", "--echo-script", SDT_ANSWER]), 0
            )
            archived = list(self.out.glob("manifest.sdt.*.json"))
            self.assertTrue(archived, "the previous manifest was not archived")
            self.assertTrue(
                any(
                    json.loads(p.read_text(encoding="utf-8")).get("dataset_sha256")
                    == "0" * 64
                    for p in archived
                )
            )
            for stale in archived:
                stale.unlink()
        finally:
            manifest_path.write_text(original, encoding="utf-8")

    def test_09_inspect_and_coverage_run(self):
        self.assertEqual(
            main(["inspect", "--dataset", "tests/fixtures/pa_mini.json", "--dataset-kind", "pa"]), 0
        )
        # the coverage audit degrades to the graph-only sections when the
        # released datasets are absent, rather than failing
        self.assertEqual(main(["coverage", "--out", str(self.out / "coverage.md")]), 0)


if __name__ == "__main__":
    unittest.main()
