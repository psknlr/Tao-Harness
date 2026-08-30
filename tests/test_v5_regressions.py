"""Regressions for the V5 review findings.

The V5 audit went looking for conditions the tests did not cover, and found
seven. Each one below fails on the code as it stood.
"""

import importlib.util
import json
import unittest
from pathlib import Path
from typing import Any, Dict, List

import tcm_tools  # noqa: F401  (registers the tool surface)
from tcm_agent import AgentRuntime, FrameworkConfig, build_task
from tcm_agent.trace import Trace
from tcm_eval.scorers import ScoredItem, cp_family
from tcm_kg.schema import Domain
from tcm_models import build_client, spec_from_config
from tcm_tools.base import REGISTRY, ToolBudget, ToolContext, ToolPhase

from ._fixtures import graph, retriever

REPO = Path(__file__).resolve().parent.parent

SDT_ITEM = {
    "id": "t",
    "clinical_data": "男，52岁。心悸不安，善惊易恐，多梦易醒。舌淡红，脉细。",
    "pathogenesis_options": {chr(65 + i): f"病机{i}" for i in range(10)},
    "syndrome_options": {chr(65 + i): f"证候{i}" for i in range(10)},
}
ANSWER = json.dumps(
    {
        "action": "answer",
        "result": {
            "syndrome_answer": ["A"],
            "pathogenesis_answer": ["A"],
            "clinical_information": ["心悸"],
            "explanation": "x",
        },
    },
    ensure_ascii=False,
)
REVISION = json.dumps({"syndrome_answer": ["A"], "revision": "unchanged"}, ensure_ascii=False)
TOOL_CALL = json.dumps(
    {"action": "tool", "tool": "search_tcm_entities", "arguments": {"query": "心悸"}},
    ensure_ascii=False,
)


def echo(script):
    return build_client(
        spec_from_config("echo", {"provider": "echo", "model_id": "e"}), script=script
    )


def builder():
    spec = importlib.util.spec_from_file_location(
        "build_tcm_cp", REPO / "scripts" / "build_tcm_cp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BUILT: List[Dict[str, Any]] = []


def built_items() -> List[Dict[str, Any]]:
    """The benchmark as the builder produces it, from the committed graph.

    Deliberately *not* ``data/cp/TCM-CP.json``: that file is generated and
    gitignored, so a test reading it passes on a machine where someone has run
    the builder and fails on a clean checkout -- which is what CI is. Building
    here also means these tests check the builder rather than whatever happens
    to be on disk.
    """
    if not _BUILT:
        from tcm_kg import load_kg

        items, _report = builder().build(load_kg(), seed=20260829)
        _BUILT.extend(items)
    return _BUILT


class V5_1_CaseIdentity(unittest.TestCase):
    """case_id is the primary key of scoring, resume and paired analysis."""

    def test_cp4_ids_include_the_disease(self):
        """One syndrome serves many diseases; the id has to say which."""
        cp4 = [i for i in built_items() if i["subtask"].startswith("CP4")]
        self.assertTrue(cp4)
        ids = [i["id"] for i in cp4]
        self.assertEqual(len(ids), len(set(ids)), "CP4 ids still collide")
        for item in cp4[:20]:
            self.assertIn("::Disease::", item["id"])

    def test_the_whole_build_has_unique_ids(self):
        ids = [i["id"] for i in built_items()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_the_builder_refuses_a_set_with_a_duplicate_id(self):
        b = builder()
        item = {
            "id": "x", "subtask": "CP1_pathway_eligibility", "question": "q",
            "options": {"A": "aa", "B": "bb", "C": "cc"}, "answer": ["A"], "vignette": "v",
        }
        self.assertEqual(b.validate([item]), {})
        failures = b.validate([item, dict(item)])
        self.assertIn("duplicate_id", failures)

    def test_the_builder_refuses_two_options_with_the_same_answer(self):
        """A near-duplicate node made 20 items unanswerable."""
        b = builder()
        item = {
            "id": "x", "subtask": "CP4_treatment_principle", "question": "q",
            # the graph really does hold both spellings as separate nodes
            "options": {"A": "镇惊定志，养心安神", "B": "镇惊定志，养心安神。", "C": "cc"},
            "answer": ["A"], "vignette": "v",
        }
        self.assertIn("duplicate_option_text", b.validate([item]))

    def test_the_builder_refuses_an_answer_echoed_in_the_question(self):
        b = builder()
        item = {
            "id": "x", "subtask": "CP4_treatment_principle", "question": "本证的治法是？",
            "options": {"A": "活血化瘀，行气止痛", "B": "bb", "C": "cc"}, "answer": ["A"],
            "vignette": "四诊所见：活血化瘀，行气止痛",
        }
        self.assertIn("answer_echoed_in_question", b.validate([item]))

    def test_echo_detection_is_punctuation_and_width_insensitive(self):
        b = builder()
        item = {
            "id": "x", "subtask": "CP3_stage_actions", "question": "q",
            "options": {"A": "询问病史，检查乳房，中医舌脉诊", "B": "bb", "C": "cc"},
            "answer": ["A"],
            # same sentence, different marks -- a raw substring check misses it
            "vignette": "已完成：询问病史；检查乳房、中医舌脉诊",
        }
        self.assertIn("answer_echoed_in_question", b.validate([item]))

    def test_the_loader_imposes_unique_keys_and_says_so(self):
        """TCM-SD ships 178 repeated user_ids; a released corpus is not ours to fix."""
        from tcm_eval.datasets import Dataset, FieldMapping, enforce_unique_case_ids

        data = Dataset("x", [{"id": "a"}, {"id": "a"}, {"id": "b"}], FieldMapping())
        renamed = enforce_unique_case_ids(data)
        self.assertEqual(renamed, 1)
        self.assertEqual([i["id"] for i in data.items], ["a", "a#2", "b"])
        self.assertEqual(data.items[1]["_original_id"], "a")
        self.assertTrue(any("unique" in n for n in data.mapping.notes))

    def test_renaming_is_deterministic(self):
        from tcm_eval.datasets import Dataset, FieldMapping, enforce_unique_case_ids

        def ids():
            data = Dataset("x", [{"id": "a"} for _ in range(4)], FieldMapping())
            enforce_unique_case_ids(data)
            return [i["id"] for i in data.items]

        self.assertEqual(ids(), ids())


class V5_2_PhaseIsEnforced(unittest.TestCase):
    """Withholding a tool from the prompt is obscurity, not isolation."""

    def _ctx(self, phase):
        return ToolContext(graph(), retriever(), Domain.SAFETY, ToolBudget(), phase=phase)

    def test_an_agent_context_cannot_run_a_verification_tool(self):
        for name, args in (
            ("check_dose", {"items": [{"herb": "石膏", "dose": "30g"}]}),
            ("verify_tcm_decision", {"syndrome": "心虚胆怯证"}),
        ):
            result = REGISTRY.call(name, self._ctx(ToolPhase.AGENT), args)
            self.assertFalse(result.ok, f"{name} executed from the agent phase")
            self.assertIn("phase", result.error or "")

    def test_a_verification_context_cannot_run_an_agent_tool(self):
        result = REGISTRY.call(
            "retrieve_medication_knowledge", self._ctx(ToolPhase.VERIFICATION), {"name": "石膏"}
        )
        self.assertFalse(result.ok)

    def test_both_phase_tools_are_reachable_from_either_side(self):
        spec = REGISTRY.spec("evaluate_pathway_transition")
        self.assertIs(spec.phase, ToolPhase.BOTH)
        for phase in (ToolPhase.AGENT, ToolPhase.VERIFICATION):
            ctx = ToolContext(graph(), retriever(), Domain.PATHWAY, ToolBudget(), phase=phase)
            result = REGISTRY.call("evaluate_pathway_transition", ctx, {"stage": "第1天"})
            self.assertNotIn("phase", result.error or "")

    def test_an_m3_agent_naming_a_checker_is_refused_at_the_boundary(self):
        """The model can produce a name it was never shown."""
        task = build_task("pa", graph(), retriever())
        item = {"id": "x", "question": "q", "options": {"A": "a", "B": "b"},
                "category": "处方书写"}
        sneak = json.dumps(
            {"action": "tool", "tool": "check_dose",
             "arguments": {"items": [{"herb": "石膏", "dose": "30g"}]}},
            ensure_ascii=False,
        )
        answer = json.dumps(
            {"action": "answer",
             "result": {"rule_category": "处方书写", "answer": ["A"], "reasoning": "r"}},
            ensure_ascii=False,
        )
        trace = AgentRuntime(
            graph(), retriever(), task, echo([sneak, answer]), FrameworkConfig()
        ).run(item, "M3")
        step = trace.tool_steps[0]
        self.assertEqual(step.tool, "check_dose")
        self.assertFalse(step.ok, "M3 reached a verification-phase checker")
        self.assertEqual(step.phase, "agent")

    def test_the_m4_verifier_still_reaches_its_checkers(self):
        task = build_task("sdt", graph(), retriever())
        trace = AgentRuntime(
            graph(), retriever(), task, echo([ANSWER, REVISION]), FrameworkConfig()
        ).run(SDT_ITEM, "M4")
        verifier = [s for s in trace.tool_steps if s.phase == "verification"]
        self.assertTrue(verifier, "the verification pass could not call anything")
        self.assertTrue(any(s.ok for s in verifier))


class V5_3_DiseaseConditionedTreatment(unittest.TestCase):
    """Syndrome -> Treatment is a global edge; the clinical fact is ternary."""

    def test_treatments_split_by_whether_this_disease_attests_them(self):
        from tcm_kg import load_kg

        kg = load_kg()
        for disease in kg.of_type("Disease"):
            rows = kg.syndromes_of(disease.id)
            if not rows:
                continue
            plan = kg.treatments_of(rows[0]["id"], disease_id=disease.id)
            self.assertEqual(plan["scope"], "disease_conditioned")
            self.assertEqual(plan["disease"], disease.name)
            for row in plan["disease_specific"]:
                self.assertTrue(row.get("shared_docs"))
            return
        self.skipTest("no disease with syndromes in the graph")

    def test_an_unscoped_lookup_claims_nothing(self):
        from tcm_kg import load_kg

        kg = load_kg()
        syndrome = next(iter(kg.of_type("Syndrome")))
        plan = kg.treatments_of(syndrome.id)
        self.assertEqual(plan["scope"], "unscoped")
        self.assertEqual(plan["disease_specific"], [])

    def test_cp4_prefers_a_disease_grounded_gold_and_labels_the_rest(self):
        cp4 = [i for i in built_items() if i["subtask"].startswith("CP4")]
        provenance = {i["treatment_provenance"] for i in cp4}
        self.assertTrue(provenance <= {"disease_specific", "cross_disease_general"})
        grounded = sum(1 for i in cp4 if i["treatment_provenance"] == "disease_specific")
        # was 45.55% before the fix
        self.assertGreater(grounded / len(cp4), 0.80)

    def test_the_provenance_label_reaches_the_report(self):
        """A label the report cannot read is a label that does not exist."""
        from tcm_eval.report import cp4_provenance_table
        from tcm_eval.scorers import score_cp

        row = score_cp(
            {"answer": ["A"]},
            {"subtask": "CP4_formula", "answer": ["A"],
             "treatment_provenance": "cross_disease_general"},
        )
        self.assertEqual(row["treatment_provenance"], "cross_disease_general")

        items = [
            ScoredItem(f"{scope}{i}", "cp", "M3", "m", 0,
                       {"exact": 1.0 if scope == "d" else 0.0,
                        "subtask": "CP4_formula", "cp_family": "CP4",
                        "treatment_provenance":
                            "disease_specific" if scope == "d" else "cross_disease_general"})
            for scope in ("d", "x")
            for i in range(5)
        ]
        table = cp4_provenance_table(items)
        self.assertIn("disease_specific", table)
        self.assertIn("cross_disease_general", table)

    def test_the_treatment_tool_requires_a_disease(self):
        spec = REGISTRY.spec("retrieve_treatment_plan")
        self.assertIn("disease", spec.parameters["required"])
        self.assertIn("disease", spec.parameters["properties"])

    def test_the_static_context_offers_what_the_tool_offers(self):
        """Else M2C->M3 measures missing knowledge, not adaptive retrieval."""
        from tcm_kg import load_kg
        from tcm_kg.index import KGRetriever

        kg = load_kg()
        r = KGRetriever(kg)
        r.warm()
        task = build_task("cp", kg, r)
        tool_keys = {key for _relation, key in kg.TREATMENT_EDGES}

        seen: set = set()
        for item in built_items():
            if not item["subtask"].startswith("CP4"):
                continue
            treatment = task.static_context(item).get("treatment")
            if treatment:
                for scope in ("disease_specific", "cross_disease_general"):
                    seen.update(treatment[scope])
            if seen >= tool_keys:
                break
        self.assertEqual(
            seen & tool_keys, tool_keys,
            f"the static context never offers {sorted(tool_keys - seen)}",
        )


class V5_4_MacroEstimand(unittest.TestCase):
    """The interval must describe the number printed above it."""

    def test_the_hierarchical_macro_weights_families_not_subtasks(self):
        from tcm_eval.stats import hierarchical_macro_bootstrap

        a, b, strata, families = [], [], [], []
        subtasks = [("CP4_a", "CP4"), ("CP4_b", "CP4"), ("CP4_c", "CP4"), ("CP4_d", "CP4"),
                    ("CP1", "CP1"), ("CP2", "CP2"), ("CP3", "CP3"),
                    ("CP5", "CP5"), ("CP6", "CP6")]
        for subtask, family in subtasks:
            for _ in range(20):
                a.append(0.0)
                b.append(1.0 if family == "CP4" else 0.0)
                strata.append(subtask)
                families.append(family)
        result = hierarchical_macro_bootstrap(a, b, strata, families, n_resamples=300)
        self.assertAlmostEqual(result.delta, 1 / 6, places=3)
        self.assertNotAlmostEqual(result.delta, 4 / 9, places=2)

    def test_a_flat_macro_is_the_one_level_case(self):
        from tcm_eval.stats import hierarchical_macro_bootstrap, stratified_macro_bootstrap

        a = [0.0] * 40
        b = [1.0] * 20 + [0.0] * 20
        strata = ["s1"] * 20 + ["s2"] * 20
        flat = stratified_macro_bootstrap(a, b, strata, n_resamples=200)
        same = hierarchical_macro_bootstrap(a, b, strata, strata, n_resamples=200)
        self.assertAlmostEqual(flat.delta, same.delta, places=6)
        self.assertAlmostEqual(flat.delta, 0.5, places=6)

    def test_a_subtask_cannot_belong_to_two_families(self):
        from tcm_eval.stats import hierarchical_macro_bootstrap

        with self.assertRaises(ValueError):
            hierarchical_macro_bootstrap(
                [0.0, 0.0], [1.0, 1.0], ["s", "s"], ["CP1", "CP2"]
            )

    def test_every_cp_subtask_the_builder_emits_has_a_family(self):
        for subtask in {i["subtask"] for i in built_items()}:
            self.assertNotEqual(
                cp_family(subtask), "other", f"{subtask} is outside CP_FAMILIES"
            )


class V5_5_PerCaseComputeMatching(unittest.TestCase):
    """A mean is not a match for a paired test."""

    def _group(self, n_tool_calls):
        task = build_task("sdt", graph(), retriever())
        script = [TOOL_CALL] * n_tool_calls + [ANSWER] + [REVISION] * 2 + [ANSWER] * 8
        runtime = AgentRuntime(
            graph(), retriever(), task, echo(script), FrameworkConfig()
        )
        return runtime.run_branch_group(SDT_ITEM, ["M2C", "M3", "M3C", "M4"])

    def test_m2c_spends_exactly_what_its_m3_twin_spent(self):
        for n in (1, 2, 3):
            group = self._group(n)
            self.assertEqual(
                group["M2C"].n_llm_calls, group["M3"].n_llm_calls,
                f"M2C and M3 diverged with {n} tool call(s)",
            )
            self.assertEqual(group["M2C"].parity_error, "")

    def test_m2c_would_otherwise_answer_on_its_first_turn(self):
        """Without the pin the arms differ by two calls on this very case."""
        task = build_task("sdt", graph(), retriever())
        unpinned = AgentRuntime(
            graph(), retriever(), task, echo([ANSWER] * 6), FrameworkConfig()
        ).run(SDT_ITEM, "M2C")
        self.assertEqual(unpinned.n_llm_calls, 1)
        self.assertEqual(self._group(2)["M2C"].n_llm_calls, 3)

    def test_m2c_is_co_generated_with_the_branch_group(self):
        from tcm_agent.runtime import CO_GENERATED, GROUPED

        self.assertIn("M2C", CO_GENERATED)
        self.assertIn("M2C", GROUPED)
        group = self._group(2)
        self.assertEqual(len({t.branch_group for t in group.values()}), 1)

    def test_m2c_alone_runs_unpinned_rather_than_failing(self):
        task = build_task("sdt", graph(), retriever())
        runtime = AgentRuntime(
            graph(), retriever(), task, echo([ANSWER] * 4), FrameworkConfig()
        )
        group = runtime.run_branch_group(SDT_ITEM, ["M2C"])
        self.assertEqual(set(group), {"M2C"})


class V5_6_SharedTrajectoryIsAnInvariant(unittest.TestCase):
    def test_a_contrast_refuses_arms_from_different_groups(self):
        from tcm_eval.report import index_items, paired_vectors

        def item(case, condition, value, group):
            return ScoredItem(
                case, "sdt", condition, "m", 0, {"exact": value},
                {"parity_error": "", "branch_group": group},
            )

        items = [
            item("a", "M3C", 0.0, "a#0"), item("a", "M4", 1.0, "a#0"),      # same run
            item("b", "M3C", 0.0, "b#0"), item("b", "M4", 1.0, "b#RESUMED"),  # split
        ]
        xs, _ = paired_vectors(index_items(items), "m", "M3C", "M4", "exact")
        self.assertEqual(len(xs), 1, "a split-trajectory pair was scored")

    def test_a_missing_group_id_is_not_a_match(self):
        from tcm_eval.report import index_items, paired_vectors

        def item(condition, group):
            return ScoredItem(
                "a", "sdt", condition, "m", 0, {"exact": 1.0},
                {"parity_error": "", "branch_group": group},
            )

        xs, _ = paired_vectors(
            index_items([item("M3C", ""), item("M4", "")]), "m", "M3C", "M4", "exact"
        )
        self.assertEqual(xs, [])

    def test_unrelated_contrasts_are_not_group_checked(self):
        """M0 and M1 are independent by design; requiring a group would drop all."""
        from tcm_eval.report import index_items, paired_vectors

        def item(condition):
            return ScoredItem("a", "sdt", condition, "m", 0, {"exact": 1.0}, {})

        xs, _ = paired_vectors(
            index_items([item("M0"), item("M1")]), "m", "M0", "M1", "exact"
        )
        self.assertEqual(len(xs), 1)


class V5_7_ExperimentDesignFreeze(unittest.TestCase):
    def test_the_design_signature_covers_what_the_run_signature_omits(self):
        from tcm_eval.provenance import design_signature

        base = dict(run_signature="apparatus", conditions=["M0", "M1"], samples=1,
                    limit=100, stratify="subtask")
        first = design_signature(**base)
        for field, value in (
            ("conditions", ["M0", "M1", "M2"]),
            ("samples", 3),
            ("limit", 200),
            ("stratify", None),
        ):
            self.assertNotEqual(
                first, design_signature(**{**base, field: value}), f"{field} not covered"
            )

    def test_condition_order_does_not_change_the_design(self):
        from tcm_eval.provenance import design_signature

        a = design_signature(run_signature="x", conditions=["M0", "M1"], samples=1)
        b = design_signature(run_signature="x", conditions=["M1", "M0"], samples=1)
        self.assertEqual(a, b)

    def test_the_design_signature_survives_a_trace_round_trip(self):
        trace = Trace("r", "c", "sdt", "M0", "m", "fw", design_signature="d1")
        again = Trace.from_dict(json.loads(json.dumps(trace.to_dict(), ensure_ascii=False)))
        self.assertEqual(again.design_signature, "d1")

    def test_framework_hash_stays_blind_to_both_signatures(self):
        a, b = FrameworkConfig(), FrameworkConfig()
        a.run_signature, a.design_signature = "one", "d1"
        b.run_signature, b.design_signature = "two", "d2"
        self.assertEqual(a.framework_hash(), b.framework_hash())


class V5_9_ConsensusProvenance(unittest.TestCase):
    def test_a_merged_trace_keeps_its_provenance(self):
        from runner.benchmark_runner import _consensus_traces

        def sample(i, parity=""):
            t = Trace("r", "c", "sdt", "M4", "m", "fw",
                      run_signature="sig", design_signature="dsg", sample=i)
            t.branch_group = "c#0"
            t.verification_stratum = "deterministic"
            t.parity_error = parity
            t.final = {"syndrome_answer": ["A"]}
            return t

        merged = _consensus_traces([sample(0), sample(1), sample(2)])[0]
        self.assertEqual(merged.sample, -1)
        self.assertEqual(merged.run_signature, "sig")
        self.assertEqual(merged.design_signature, "dsg")
        self.assertEqual(merged.branch_group, "c#0")
        self.assertEqual(merged.verification_stratum, "deterministic")

    def test_a_parity_break_in_any_sample_invalidates_the_merge(self):
        from runner.benchmark_runner import _consensus_traces

        def sample(i, parity=""):
            t = Trace("r", "c", "sdt", "M4", "m", "fw", sample=i)
            t.branch_group = "c#0"
            t.parity_error = parity
            t.final = {"syndrome_answer": ["A"]}
            return t

        merged = _consensus_traces([sample(0), sample(1, "broken"), sample(2)])[0]
        self.assertIn("broken", merged.parity_error)


class TestSuiteRunsOnACleanCheckout(unittest.TestCase):
    """No test may depend on a file the repository does not ship.

    This has now been the cause of two CI failures. The benchmark datasets are
    gitignored, so a test that reads one passes on a machine where someone has
    already downloaded or generated it and fails on a clean checkout -- which
    is what CI is, and what a reviewer has. The suite is meant to prove the
    harness is sound without the data; a test that quietly needs it breaks that
    promise and does not announce itself.

    Tests that genuinely need a released corpus must skip themselves (see the
    ``skipTest`` calls elsewhere), not read the path and hope.
    """

    #: Generated or downloaded data, from .gitignore.
    FORBIDDEN = ("data/cp/", "data/sdt/", "data/pa/", "data/tcmsd/")

    #: A reference is fine when the surrounding code copes with the file being
    #: absent -- by skipping, by catching, or by checking first.
    GUARDS = ("skipTest", "FileNotFoundError", "exists()", "try:")

    #: How many lines around a reference are searched for one of those.
    WINDOW = 8

    @staticmethod
    def tracked_files() -> set:
        """Files the repository ships, without requiring a git checkout.

        A source archive -- GitHub "Download ZIP", a Zenodo deposit, a journal
        supplementary bundle -- carries no ``.git``, so shelling out to
        ``git ls-files`` made this test fail for exactly the reader most likely
        to run the suite cold, and on a defect that is not in the harness.
        Falls back to walking the tree minus the directories .gitignore
        excludes, which is the same set for this purpose.
        """
        import subprocess

        if (REPO / ".git").exists():
            try:
                return set(
                    subprocess.run(
                        ["git", "ls-files"],
                        cwd=REPO,
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout.split()
                )
            except (OSError, subprocess.CalledProcessError):
                pass
        skip = {".git", "runs", "__pycache__", ".index", "node_modules", ".venv", "data"}
        return {
            str(path.relative_to(REPO))
            for path in REPO.rglob("*")
            if path.is_file() and not any(part in skip for part in path.parts)
        }

    def test_no_test_reads_an_unguarded_ungitted_data_path(self):
        import tokenize

        tracked = self.tracked_files()

        triple = ('"' * 3, "'" * 3)
        offenders = []
        for path in sorted((REPO / "tests").rglob("*.py")):
            lines = path.read_text(encoding="utf-8").splitlines()
            # Only real string literals count -- a docstring explaining this
            # very rule is not a file read, and neither is a comment.
            with open(path, "rb") as handle:
                literals = [
                    (tok.start[0], tok.string)
                    for tok in tokenize.tokenize(handle.readline)
                    if tok.type == tokenize.STRING
                ]
            for lineno, literal in literals:
                if literal.startswith(triple):
                    continue  # a docstring, not a path
                prefix = next((p for p in self.FORBIDDEN if p in literal), None)
                if prefix is None:
                    continue
                # naming a file the repository actually ships is fine
                if any(t.startswith(prefix) and t in literal for t in tracked):
                    continue
                window = "\n".join(
                    lines[max(0, lineno - 1 - self.WINDOW) : lineno + self.WINDOW]
                )
                if any(guard in window for guard in self.GUARDS):
                    continue
                offenders.append(f"{path.name}:{lineno}: {literal[:80]}")
        self.assertEqual(
            offenders,
            [],
            "these tests read a gitignored data file with no fallback, so they "
            "pass here and fail on a clean checkout",
        )


if __name__ == "__main__":
    unittest.main()
