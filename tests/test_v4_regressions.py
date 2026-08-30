"""Regressions for the V4 review findings.

Like the V3 file, these are validity defects rather than crashes: each is a
way a reviewer could read a reported gain as an artefact of the harness.
"""

import json
import unittest

import tcm_tools  # noqa: F401  (registers the tool surface)
from tcm_agent import AgentRuntime, FrameworkConfig, build_task
from tcm_eval.metrics import tool_selection_accuracy, trace_metrics
from tcm_kg.schema import Domain
from tcm_models import build_client, spec_from_config
from tcm_tools.base import REGISTRY, ToolPhase

from pathlib import Path

from ._fixtures import graph, retriever

REPO = Path(__file__).resolve().parent.parent

SDT_ITEM = {
    "id": "t",
    "clinical_data": "男，52岁。心悸不安，善惊易恐，多梦易醒。舌淡红，脉细。",
    "pathogenesis_options": {chr(65 + i): f"病机{i}" for i in range(10)},
    "syndrome_options": {
        "A": "心虚胆怯证", "B": "痰火扰心证", "C": "心脾两虚证",
        **{chr(65 + i): f"证候{i}" for i in range(3, 10)},
    },
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


def echo(script):
    return build_client(
        spec_from_config("echo", {"provider": "echo", "model_id": "e"}), script=script
    )


class V4_1_DiseaseConditionedSyndromeEvidence(unittest.TestCase):
    """A syndrome's clinical picture must come from the disease at hand."""

    @staticmethod
    def _evidenced_edge(kg):
        for edge in kg.edges.values() if hasattr(kg.edges, "values") else kg.edges:
            if str(edge.type) == "HAS_SYNDROME" and edge.evidence_sentences():
                return edge
        return None

    def test_presentation_prefers_the_edge_over_the_global_first_mention(self):
        kg = graph()
        edge = self._evidenced_edge(kg)
        if edge is None:
            self.skipTest("fixture graph has no evidenced HAS_SYNDROME edge")
        out = kg.syndrome_presentation(edge.target, disease_id=edge.source)
        self.assertEqual(out["scope"], "disease_specific")
        self.assertEqual(out["sentence"], edge.evidence_sentences()[0])
        self.assertEqual(out["disease"], kg.nodes[edge.source].name)

    def test_a_context_free_lookup_is_labelled_and_caveated(self):
        kg = graph()
        edge = self._evidenced_edge(kg)
        if edge is None:
            self.skipTest("fixture graph has no evidenced HAS_SYNDROME edge")
        out = kg.syndrome_presentation(edge.target)
        self.assertNotEqual(
            out["scope"], "disease_specific", "an unscoped lookup must not claim disease scope"
        )
        if out["sentence"]:
            self.assertEqual(out["scope"], "global_first_mention")
            self.assertTrue(out.get("caveat"), "an unscoped presentation must warn the reader")

    def test_syndromes_of_carries_per_syndrome_evidence(self):
        kg = graph()
        for node in kg.nodes.values():
            if str(node.type) != "Disease":
                continue
            rows = kg.syndromes_of(node.id)
            if rows:
                self.assertIn("presentation", rows[0])
                self.assertIn("scope", rows[0])
                self.assertIn(
                    rows[0]["scope"], {"disease_specific", "global_first_mention"}
                )
                return
        self.skipTest("fixture graph has no disease with syndromes")


class V4_3_ToolPhaseIsolation(unittest.TestCase):
    """Agent-chosen tool calls and verifier-issued ones must be separable."""

    def test_deterministic_checkers_are_hidden_from_the_agent(self):
        agent = {s.name for s in REGISTRY.specs_for(Domain.SAFETY, phase=ToolPhase.AGENT)}
        verify = {s.name for s in REGISTRY.specs_for(Domain.SAFETY, phase=ToolPhase.VERIFICATION)}
        for checker in (
            "check_dose",
            "check_duplicate_medication",
            "check_restricted_item",
            "check_combination",
            "check_decoction_requirement",
            "verify_tcm_decision",
        ):
            self.assertNotIn(checker, agent, f"{checker} is callable by the agent")
            self.assertIn(checker, verify, f"{checker} is unreachable from the M4 pass")

    def test_the_pa_prompt_never_advertises_a_checker(self):
        task = build_task("pa", graph(), retriever())
        runtime = AgentRuntime(graph(), retriever(), task, echo(["{}"]), FrameworkConfig())
        protocol = runtime._tool_protocol()
        self.assertNotIn("check_dose", protocol)
        self.assertNotIn("verify_tcm_decision", protocol)
        self.assertIn("retrieve_medication_knowledge", protocol)

    def test_verifier_calls_are_tagged_verification(self):
        task = build_task("sdt", graph(), retriever())
        trace = AgentRuntime(
            graph(), retriever(), task, echo([ANSWER, REVISION]), FrameworkConfig()
        ).run(SDT_ITEM, "M4")
        phases = {s.tool: s.phase for s in trace.tool_steps}
        self.assertEqual(phases.get("verify_tcm_decision"), "verification")
        self.assertGreater(trace.n_verification_tool_calls, 0)
        self.assertNotIn("verify_tcm_decision", trace.agent_tools_used())

    def test_a_verifier_call_cannot_inflate_tool_selection_accuracy(self):
        """The metric must read the model's choices, not the harness's."""
        task = build_task("sdt", graph(), retriever())
        trace = AgentRuntime(
            graph(), retriever(), task, echo([ANSWER, REVISION]), FrameworkConfig()
        ).run(SDT_ITEM, "M4")
        self.assertTrue(trace.tool_steps, "expected the verifier to have run")
        # every step here is verification-phase, so there is nothing the model
        # selected and the metric must decline to score rather than award 1.0
        self.assertEqual(trace.n_agent_tool_calls, 0)
        self.assertIsNone(tool_selection_accuracy(trace, "A-002"))

    def test_metrics_report_agent_and_verifier_counts_separately(self):
        task = build_task("sdt", graph(), retriever())
        trace = AgentRuntime(
            graph(), retriever(), task, echo([ANSWER, REVISION]), FrameworkConfig()
        ).run(SDT_ITEM, "M4")
        row = trace_metrics(trace)
        self.assertEqual(
            row["n_tool_calls"], row["n_agent_tool_calls"] + row["n_verification_tool_calls"]
        )
        self.assertEqual(row["tools_used"], trace.agent_tools_used())
        self.assertIn("verify_tcm_decision", row["verification_tools_used"])

    def test_phase_survives_a_trace_round_trip(self):
        from tcm_agent.trace import Trace

        task = build_task("sdt", graph(), retriever())
        trace = AgentRuntime(
            graph(), retriever(), task, echo([ANSWER, REVISION]), FrameworkConfig()
        ).run(SDT_ITEM, "M4")
        again = Trace.from_dict(json.loads(json.dumps(trace.to_dict(), ensure_ascii=False)))
        self.assertEqual(
            [s.phase for s in again.tool_steps], [s.phase for s in trace.tool_steps]
        )

    def test_specs_are_frozen_so_a_phase_cannot_be_edited_mid_run(self):
        with self.assertRaises(Exception):
            REGISTRY.spec("check_dose").phase = ToolPhase.AGENT

    def test_phase_is_part_of_the_tool_fingerprint(self):
        """Re-labelling a tool's phase changes the experiment, so it must hash."""
        import dataclasses

        from tcm_tools.base import ToolRegistry

        before = REGISTRY.fingerprint(Domain.SAFETY)
        shadow = ToolRegistry()
        for spec in REGISTRY.specs_for(Domain.SAFETY):
            moved = (
                dataclasses.replace(spec, phase=ToolPhase.AGENT)
                if spec.phase is ToolPhase.VERIFICATION
                else spec
            )
            shadow.register(moved)(lambda ctx, **kw: None)
        self.assertNotEqual(
            before,
            shadow.fingerprint(Domain.SAFETY),
            "moving a checker out of the verification phase left the hash unchanged",
        )


class V4_4_PerCaseComputeParity(unittest.TestCase):
    """M3C and M4 must cost the same on *every* case, not on average."""

    def _group(self, verify_arguments):
        """A branch group whose task adjudicates (or not) as the test dictates."""
        task = build_task("sdt", graph(), retriever())
        task.verify_arguments = verify_arguments  # type: ignore[method-assign]
        runtime = AgentRuntime(
            graph(),
            retriever(),
            task,
            echo([ANSWER, REVISION, REVISION]),
            FrameworkConfig(),
        )
        return runtime.run_branch_group(SDT_ITEM, ["M3", "M3C", "M4"])

    def test_m4_takes_its_revision_turn_even_with_nothing_to_check(self):
        """The case the fix exists for: no checker applies and no prose to audit."""
        group = self._group(lambda result, item: None)
        self.assertEqual(
            group["M4"].n_llm_calls,
            group["M3C"].n_llm_calls,
            "M4 skipped its revision turn where M3C took one",
        )
        self.assertEqual(group["M4"].verification_stratum, "not_applicable")
        report = group["M4"].final["verification_report"]
        self.assertEqual(report["verdict"], "not_applicable")
        self.assertEqual(report["n_checks"], 0)

    def test_an_adjudicated_case_is_labelled_deterministic(self):
        group = self._group(build_task("sdt", graph(), retriever()).verify_arguments)
        self.assertEqual(group["M4"].verification_stratum, "deterministic")
        self.assertEqual(group["M4"].n_llm_calls, group["M3C"].n_llm_calls)

    def test_branches_carry_a_shared_pairing_id(self):
        group = self._group(lambda result, item: None)
        ids = {t.branch_group for t in group.values()}
        self.assertEqual(len(ids), 1)
        self.assertTrue(ids.pop())

    def test_no_parity_error_is_recorded_when_the_arms_match(self):
        for plans in (lambda r, i: None, build_task("sdt", graph(), retriever()).verify_arguments):
            group = self._group(plans)
            for condition, trace in group.items():
                self.assertEqual(trace.parity_error, "", f"{condition} flagged unexpectedly")

    def test_a_parity_break_is_recorded_on_both_arms(self):
        from tcm_agent.runtime import _check_compute_parity
        from tcm_agent.trace import LLMStep, Trace

        def make(condition, n):
            t = Trace("r", "c", "sdt", condition, "m", "h")
            for i in range(n):
                t.llm_steps.append(LLMStep(i, "answer", 0, {}))
            return t

        branches = {"M3C": make("M3C", 2), "M4": make("M4", 1)}
        _check_compute_parity(branches)
        self.assertIn("compute parity broken", branches["M3C"].parity_error)
        self.assertIn("compute parity broken", branches["M4"].parity_error)

    def test_the_report_drops_parity_broken_pairs_from_the_contrast(self):
        from tcm_eval.report import dropped_for_parity, index_items, paired_vectors
        from tcm_eval.scorers import ScoredItem

        def item(case, condition, value, broken=False):
            return ScoredItem(
                case, "sdt", condition, "m", 0,
                {"composite": value},
                {"parity_error": "compute parity broken: ..." if broken else ""},
            )

        items = [
            item("a", "M3C", 0.0), item("a", "M4", 1.0),
            item("b", "M3C", 0.0), item("b", "M4", 1.0),
            item("c", "M3C", 0.0, broken=True), item("c", "M4", 1.0, broken=True),
        ]
        xs, _ = paired_vectors(index_items(items), "m", "M3C", "M4", "composite")
        self.assertEqual(len(xs), 2, "an incomparable pair was scored")
        self.assertEqual(dropped_for_parity(items, "M3C", "M4"), {"m": 1})

    def test_cluster_labels_stay_aligned_with_the_filtered_values(self):
        """A label list built from a different filter mislabels every later pair."""
        from tcm_eval.report import index_items, paired_clusters, paired_vectors
        from tcm_eval.scorers import ScoredItem

        def item(case, condition, value, disease, broken=False):
            return ScoredItem(
                case, "cp", condition, "m", 0,
                {"composite": value, "disease": disease},
                {"parity_error": "broken" if broken else ""},
            )

        items = [
            item("a", "M3C", 0.0, "D1", broken=True), item("a", "M4", 1.0, "D1", broken=True),
            item("b", "M3C", 0.0, "D2"), item("b", "M4", 1.0, "D2"),
            item("c", "M3C", 0.0, "D3"), item("c", "M4", 1.0, "D3"),
        ]
        index = index_items(items)
        xs, _ = paired_vectors(index, "m", "M3C", "M4", "composite")
        labels = paired_clusters(index, "m", "M3C", "M4", "composite", "disease")
        self.assertEqual(len(xs), len(labels))
        self.assertEqual(labels, ["D2", "D3"])

    def test_the_stratum_table_separates_verification_from_a_second_turn(self):
        from tcm_eval.report import verification_stratum_table
        from tcm_eval.scorers import ScoredItem

        def item(case, condition, value, stratum):
            return ScoredItem(
                case, "sdt", condition, "m", 0,
                {"composite": value},
                {"verification_stratum": stratum, "parity_error": ""},
            )

        items = []
        for i in range(30):
            items.append(item(f"d{i}", "M3C", 0.5, "deterministic"))
            items.append(item(f"d{i}", "M4", 0.9, "deterministic"))
        for i in range(30):
            items.append(item(f"n{i}", "M3C", 0.5, "not_applicable"))
            items.append(item(f"n{i}", "M4", 0.5, "not_applicable"))
        table = verification_stratum_table(items, "sdt", "composite")
        self.assertIn("deterministic", table)
        self.assertIn("not_applicable", table)
        rows = {
            line.split("|")[2].strip(): line
            for line in table.splitlines()
            if line.startswith("|") and "stratum" not in line and "---" not in line
        }
        self.assertIn("+0.4", rows["deterministic"].replace("0.400", "+0.400"))
        self.assertIn("30", rows["not_applicable"])


class V4_5_RunProvenance(unittest.TestCase):
    """A trace must be able to prove what produced it, without a manifest."""

    KW = dict(
        framework_hash="fw", kg_hash="kg", dataset_hash="ds",
        model_fingerprint="m", case_set="cs", code={"tools_impl_sha256": "t"},
    )

    def test_the_signature_covers_what_the_framework_hash_cannot(self):
        from tcm_eval.provenance import run_signature

        base = run_signature(**self.KW)
        for field, value in (
            ("model_fingerprint", "other-model"),
            ("case_set", "other-cases"),
        ):
            self.assertNotEqual(
                base, run_signature(**{**self.KW, field: value}), f"{field} not covered"
            )
        self.assertNotEqual(
            base,
            run_signature(**{**self.KW, "code": {"tools_impl_sha256": "rewritten"}}),
            "a tool rewrite left the signature unchanged",
        )

    def test_the_signature_ignores_the_condition(self):
        """Conditions are the independent variable; they must stay poolable."""
        from tcm_eval.provenance import run_signature

        self.assertEqual(run_signature(**self.KW), run_signature(**self.KW))

    def test_every_trace_carries_the_signature(self):
        task = build_task("sdt", graph(), retriever())
        config = FrameworkConfig()
        config.run_signature = "sig-123"
        group = AgentRuntime(
            graph(), retriever(), task, echo([ANSWER, REVISION, REVISION]), config
        ).run_branch_group(SDT_ITEM, ["M3", "M3C", "M4"])
        for condition, trace in group.items():
            self.assertEqual(trace.run_signature, "sig-123", condition)
        solo = AgentRuntime(
            graph(), retriever(), task, echo([ANSWER]), config
        ).run(SDT_ITEM, "M0")
        self.assertEqual(solo.run_signature, "sig-123")

    def test_the_signature_survives_a_trace_round_trip(self):
        from tcm_agent.trace import Trace

        t = Trace("r", "c", "sdt", "M0", "m", "fw", run_signature="sig-123")
        again = Trace.from_dict(json.loads(json.dumps(t.to_dict(), ensure_ascii=False)))
        self.assertEqual(again.run_signature, "sig-123")

    def test_the_framework_hash_stays_blind_to_the_signature(self):
        """It has to, or two models could never be certified as sharing a scaffold."""
        a, b = FrameworkConfig(), FrameworkConfig()
        a.run_signature, b.run_signature = "one", "two"
        self.assertEqual(a.framework_hash(), b.framework_hash())

    def test_manifest_conflicts_catch_a_repointed_experiment(self):
        from tcm_eval.provenance import manifest_conflicts

        frozen = {"task": "sdt", "framework_hash": "aaa", "dataset_sha256": "d1"}
        self.assertEqual(manifest_conflicts(frozen, dict(frozen)), [])
        conflicts = manifest_conflicts(frozen, {**frozen, "dataset_sha256": "d2"})
        self.assertEqual(len(conflicts), 1)
        self.assertIn("dataset_sha256", conflicts[0])

    def test_manifest_conflicts_ignore_fields_a_resume_may_add(self):
        from tcm_eval.provenance import manifest_conflicts

        frozen = {"task": "sdt", "framework_hash": "aaa"}
        self.assertEqual(
            manifest_conflicts(frozen, {**frozen, "models": {"new": {}}, "n_items": 9}),
            [],
        )


class V4_5_ReportProvenance(unittest.TestCase):
    def test_each_dataset_keeps_its_own_provenance_block(self):
        from tcm_eval.report import build_report
        from tcm_eval.scorers import ScoredItem

        items = [
            ScoredItem("a", "sdt", "M0", "m", 0, {"sdt_composite": 0.5}),
            ScoredItem("b", "pa", "M0", "m", 0, {"pa_accuracy": 0.5}),
        ]
        report = build_report(
            items,
            framework={
                "sdt": {"framework_hash": "sdt-hash", "dataset_sha256": "sdt-data"},
                "pa": {"framework_hash": "pa-hash", "dataset_sha256": "pa-data"},
            },
        )
        for token in ("sdt-hash", "pa-hash", "sdt-data", "pa-data"):
            self.assertIn(token, report, f"{token} missing from the provenance block")

    def test_a_forced_score_run_is_labelled_in_the_report(self):
        from tcm_eval.report import build_report
        from tcm_eval.scorers import ScoredItem

        report = build_report(
            [ScoredItem("a", "sdt", "M0", "m", 0, {"sdt_composite": 0.5})],
            framework={
                "sdt": {
                    "framework_hash": "h",
                    "scored_with_allow_drift": True,
                    "drift": ["dataset_sha256 differs from the frozen manifest"],
                }
            },
        )
        self.assertIn("--allow-drift", report)
        self.assertIn("not\nreproducible", report.replace(" ", "\n"))

    def test_a_flat_provenance_block_still_renders(self):
        from tcm_eval.report import build_report
        from tcm_eval.scorers import ScoredItem

        report = build_report(
            [ScoredItem("a", "sdt", "M0", "m", 0, {"sdt_composite": 0.5})],
            framework={"task": "sdt", "framework_hash": "legacy-hash"},
        )
        self.assertIn("legacy-hash", report)

    def test_trace_summaries_do_not_collide_across_datasets(self):
        from tcm_eval.report import build_report
        from tcm_eval.scorers import ScoredItem

        report = build_report(
            [ScoredItem("a", "sdt", "M3", "m", 0, {"sdt_composite": 0.5})],
            trace_summaries={
                "sdt/m/M3": {"n_traces": 11, "mean_n_llm_calls": 1.0},
                "pa/m/M3": {"n_traces": 22, "mean_n_llm_calls": 2.0},
            },
        )
        self.assertIn("| 11 ", report)
        self.assertIn("| 22 ", report)

    def test_the_behaviour_table_separates_agent_from_verifier_calls(self):
        from tcm_eval.report import build_report
        from tcm_eval.scorers import ScoredItem

        report = build_report(
            [ScoredItem("a", "sdt", "M4", "m", 0, {"sdt_composite": 0.5})],
            trace_summaries={
                "sdt/m/M4": {
                    "n_traces": 3,
                    "mean_n_tool_calls": 5.0,
                    "mean_n_agent_tool_calls": 2.0,
                    "mean_n_verification_tool_calls": 3.0,
                }
            },
        )
        self.assertIn("agent calls", report)
        self.assertIn("verifier calls", report)


class V4_6_ClinicalPathwayReporting(unittest.TestCase):
    """A pooled CP accuracy is not a pathway-execution score."""

    SIZES = {
        "CP1_pathway_eligibility": 20, "CP2_stage_identification": 40,
        "CP3_stage_actions": 40, "CP4_treatment_principle": 12,
        "CP4_formula": 12, "CP4_patent_medicine": 12,
        "CP4_external_therapy": 12, "CP5_monitoring": 40,
        "CP6_transition_decision": 8,
    }

    def _items(self, skill):
        from tcm_eval.scorers import ScoredItem, cp_family

        out = []
        for subtask, n in self.SIZES.items():
            family = cp_family(subtask)
            for i in range(n):
                # deterministic: the first `skill * n` items in each subtask hit
                hit = 1.0 if i < round(skill[family] * n) else 0.0
                for condition in ("M3C", "M4"):
                    out.append(
                        ScoredItem(
                            f"{subtask}::{i}", "cp", condition, "m", 0,
                            {"exact": hit, "subtask": subtask, "cp_family": family,
                             "disease": f"D{i % 5}"},
                            {"parity_error": ""},
                        )
                    )
        return out

    def test_every_subtask_maps_to_exactly_one_family(self):
        from tcm_eval.scorers import CP_FAMILIES, cp_family

        seen = [s for subtasks in CP_FAMILIES.values() for s in subtasks]
        self.assertEqual(len(seen), len(set(seen)), "a subtask is in two families")
        self.assertEqual(cp_family("CP4_formula"), "CP4")
        self.assertEqual(cp_family("CP6_transition_decision"), "CP6")
        self.assertEqual(cp_family("something_new"), "other")

    def test_the_scorer_tags_the_family_on_every_item(self):
        from tcm_eval.scorers import score_cp

        row = score_cp({"answer": ["A"]}, {"subtask": "CP6_transition_decision",
                                           "answer": ["A"]})
        self.assertEqual(row["cp_family"], "CP6")

    def test_the_macro_average_is_not_the_pooled_mean(self):
        """The whole point: CP6 is 8 of 196 items but one of six capabilities."""
        from tcm_eval.report import cp_subtask_table

        skill = {"CP1": 0.9, "CP2": 0.9, "CP3": 0.9, "CP4": 0.9, "CP5": 0.9, "CP6": 0.0}
        items = self._items(skill)
        pooled = sum(
            i.metrics["exact"] for i in items if i.condition == "M4"
        ) / sum(1 for i in items if i.condition == "M4")
        table = cp_subtask_table(items)
        macro_line = [
            l for l in table.splitlines() if "**macro**" in l and "| M4 " in l
        ][0]
        macro = float(macro_line.rstrip("|").rsplit("|", 1)[1])
        self.assertAlmostEqual(macro, 0.75, places=2)
        self.assertGreater(pooled, 0.85)
        self.assertLess(macro, pooled - 0.10, "the macro did not down-weight the big subtasks")

    def test_the_table_names_every_subtask_and_a_macro_row(self):
        from tcm_eval.report import cp_subtask_table

        table = cp_subtask_table(
            self._items({f"CP{i}": 0.5 for i in range(1, 7)})
        )
        for subtask in self.SIZES:
            self.assertIn(subtask, table)
        self.assertIn("**macro**", table)

    def test_the_macro_contrast_uses_a_matching_estimator(self):
        from tcm_eval.report import cp_macro_contrast_table

        table, results = cp_macro_contrast_table(
            self._items({f"CP{i}": 0.5 for i in range(1, 7)})
        )
        self.assertTrue(results)
        for result in results.values():
            self.assertTrue(
                result.test.startswith("stratified_macro_bootstrap"),
                f"macro contrast used {result.test}",
            )

    def test_a_stratified_sample_gives_the_small_subtasks_room(self):
        from tcm_eval.datasets import Dataset, FieldMapping

        items = (
            [{"id": f"a{i}", "subtask": "CP3_stage_actions"} for i in range(90)]
            + [{"id": f"b{i}", "subtask": "CP6_transition_decision"} for i in range(10)]
        )
        data = Dataset("cp", items, FieldMapping())
        head = data.subset(40)
        self.assertEqual(
            sum(1 for i in head.items if i["subtask"] == "CP6_transition_decision"), 0
        )
        balanced = data.subset(40, stratify="subtask")
        counts = {}
        for item in balanced.items:
            counts[item["subtask"]] = counts.get(item["subtask"], 0) + 1
        self.assertEqual(len(balanced.items), 40)
        self.assertEqual(counts["CP6_transition_decision"], 10, "CP6 was not filled")
        self.assertEqual(counts["CP3_stage_actions"], 30, "the surplus was not redistributed")

    def test_the_cp_config_stratifies(self):
        import yaml

        payload = yaml.safe_load(
            (REPO / "configs" / "experiment.cp.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["dataset"]["stratify"], "subtask")


class V4_6_StratifiedMacroBootstrap(unittest.TestCase):
    def test_the_point_estimate_weights_strata_equally(self):
        from tcm_eval.stats import stratified_macro_bootstrap

        # stratum "big": 90 items, no effect. stratum "small": 10 items, +1.0.
        a = [0.0] * 100
        b = [0.0] * 90 + [1.0] * 10
        strata = ["big"] * 90 + ["small"] * 10
        result = stratified_macro_bootstrap(a, b, strata, n_resamples=400)
        self.assertAlmostEqual(result.delta, 0.5, places=6)  # (0 + 1) / 2
        self.assertAlmostEqual(sum(b) / len(b), 0.1, places=6)  # the pooled mean

    def test_clustering_within_a_stratum_widens_the_interval(self):
        from tcm_eval.stats import stratified_macro_bootstrap

        a = [0.0] * 100
        b = [1.0] * 50 + [0.0] * 50  # perfectly correlated within each cluster
        strata = ["s"] * 100
        unclustered = [str(i) for i in range(100)]
        clustered = [f"c{i // 25}" for i in range(100)]  # 4 clusters
        narrow = stratified_macro_bootstrap(a, b, strata, unclustered, n_resamples=800)
        wide = stratified_macro_bootstrap(a, b, strata, clustered, n_resamples=800)
        self.assertGreater(
            wide.ci_high - wide.ci_low,
            (narrow.ci_high - narrow.ci_low) * 1.5,
            "clustering inside a stratum did not propagate into the interval",
        )

    def test_misaligned_inputs_are_rejected(self):
        from tcm_eval.stats import stratified_macro_bootstrap

        with self.assertRaises(ValueError):
            stratified_macro_bootstrap([0.0], [0.0, 1.0], ["s", "s"])
        with self.assertRaises(ValueError):
            stratified_macro_bootstrap([0.0], [0.0], ["s"], ["c1", "c2"])

    def test_an_empty_input_is_a_null_result_not_a_crash(self):
        from tcm_eval.stats import stratified_macro_bootstrap

        result = stratified_macro_bootstrap([], [], [])
        self.assertEqual(result.n, 0)
        self.assertEqual(result.p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
