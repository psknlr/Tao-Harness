"""Regressions for the V6 pre-experiment audit.

Three P0s and the P1s behind them. The contamination tests matter most: they
check the *detector*, because a null result from an unvalidated detector is
worth nothing, and "the benchmark is clean" is the claim the whole design rests
on.
"""

import importlib.util
import json
import random
import re
import unittest
from pathlib import Path
from typing import Any, Dict, List

import tcm_tools  # noqa: F401  (registers the tool surface)
from tcm_agent.trace import Trace, read_traces, write_traces
from tcm_kg import load_kg
from tcm_kg.schema import EdgeType, NodeType

REPO = Path(__file__).resolve().parent.parent

_KG: List[Any] = []


def kg():
    if not _KG:
        _KG.append(load_kg())
    return _KG[0]


def builder():
    spec = importlib.util.spec_from_file_location(
        "build_tcm_cp", REPO / "scripts" / "build_tcm_cp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BUILT: List[Dict[str, Any]] = []


def built_items() -> List[Dict[str, Any]]:
    if not _BUILT:
        items, _report = builder().build(kg(), seed=20260829)
        _BUILT.extend(items)
    return _BUILT


class V6_4_SubtypeTraversal(unittest.TestCase):
    """SUBTYPE_HAS_SYNDROME starts at the subtype, not at the disease."""

    def test_no_subtype_syndrome_is_unreachable(self):
        graph = kg()
        missed = 0
        for disease in graph.of_type(NodeType.DISEASE.value):
            via = {
                syn.id
                for _e, sub in graph.neighbours(disease.id, {EdgeType.HAS_SUBTYPE.value})
                for _e2, syn in graph.neighbours(
                    sub.id, {EdgeType.SUBTYPE_HAS_SYNDROME.value}
                )
            }
            got = {r["id"] for r in graph.syndromes_of(disease.id, include_subtypes=True)}
            missed += len(via - got)
        self.assertEqual(missed, 0, "subtype-routed syndromes are still invisible")

    def test_diseases_that_had_no_syndromes_at_all_now_have_them(self):
        graph = kg()
        for name in ("消渴病肾病（糖尿病肾病）", "肠澼（放射性直肠炎）"):
            found = graph.find_by_name(name, [NodeType.DISEASE.value])
            if not found:
                continue
            rows = graph.syndromes_of(found[0].id, include_subtypes=True)
            self.assertTrue(rows, f"{name} still exposes no syndromes")
            self.assertTrue(any(r["via"] == "subtype" for r in rows))

    def test_the_subtype_is_kept_not_flattened(self):
        graph = kg()
        for disease in graph.of_type(NodeType.DISEASE.value):
            for row in graph.syndromes_of(disease.id):
                if row["via"] == "subtype":
                    self.assertIn("subtype", row)
                    self.assertIn("subtype_id", row)
                    return
        self.skipTest("no subtype-routed syndrome in the graph")

    def test_include_subtypes_false_still_means_direct_only(self):
        graph = kg()
        for disease in graph.of_type(NodeType.DISEASE.value):
            rows = graph.syndromes_of(disease.id, include_subtypes=False)
            self.assertTrue(all(r["via"] == "disease" for r in rows))

    def test_provenance_sees_a_subtype_routed_syndrome(self):
        """The other half of the same bug: docs matched only on the disease."""
        graph = kg()
        checked = 0
        for disease in graph.of_type(NodeType.DISEASE.value):
            for row in graph.syndromes_of(disease.id):
                if row["via"] != "subtype":
                    continue
                checked += 1
                self.assertTrue(
                    graph.disease_syndrome_docs(row["id"], disease.id),
                    f"{row['name']} under {disease.name} still looks context-free",
                )
                if checked >= 30:
                    return
        if not checked:
            self.skipTest("no subtype-routed syndrome in the graph")


class V6_2_CP4PrimaryIsDiseaseSpecific(unittest.TestCase):
    """The pathway question cannot be keyed from another disease's guideline."""

    def test_no_primary_cp4_item_uses_cross_disease_gold(self):
        offenders = [
            i["id"]
            for i in built_items()
            if i["subtask"].startswith("CP4_")
            and i.get("treatment_provenance") != "disease_specific"
        ]
        self.assertEqual(offenders[:5], [], f"{len(offenders)} primary CP4 items")

    def test_the_cross_disease_items_survive_as_their_own_subtask(self):
        """異病同治 is real knowledge; dropping the items would lose it."""
        generalisation = [
            i for i in built_items() if i["subtask"] == "CP4G_cross_disease_treatment"
        ]
        self.assertTrue(generalisation, "CP4G was emitted empty")
        self.assertTrue(
            all(i["treatment_provenance"] == "cross_disease_general" for i in generalisation)
        )

    def test_the_cp4g_question_asks_what_it_can_actually_answer(self):
        item = next(
            i for i in built_items() if i["subtask"] == "CP4G_cross_disease_treatment"
        )
        self.assertIn("异病同治", item["question"])
        self.assertNotIn("本病临床路径中，本证推荐", item["question"])

    def test_cp4g_is_excluded_from_the_primary_endpoint(self):
        from tcm_eval.scorers import CP_FAMILIES, cp_family, is_primary_cp

        self.assertFalse(is_primary_cp("CP4G_cross_disease_treatment"))
        self.assertEqual(cp_family("CP4G_cross_disease_treatment"), "CP4G")
        self.assertNotIn("CP4G", CP_FAMILIES)

    def test_the_macro_tables_leave_cp4g_out(self):
        from tcm_eval.report import cp_macro_contrast_table, cp_subtask_table
        from tcm_eval.scorers import ScoredItem, cp_family

        items = []
        for subtask in ("CP4_formula", "CP4G_cross_disease_treatment"):
            for i in range(10):
                for condition, v in (("M3C", 0.0), ("M4", 1.0)):
                    items.append(
                        ScoredItem(
                            f"{subtask}::{i}", "cp", condition, "m", 0,
                            {"exact": v, "subtask": subtask,
                             "cp_family": cp_family(subtask), "disease": f"D{i % 3}"},
                            {"parity_error": "", "branch_group": f"{subtask}::{i}#0"},
                        )
                    )
        table = cp_subtask_table(items)
        self.assertIn("CP4_formula", table)
        self.assertNotIn("CP4G", table)
        _t, results = cp_macro_contrast_table(items)
        for result in results.values():
            self.assertIn("1_families", result.test)

    def test_cp4g_is_still_reported_somewhere(self):
        from tcm_eval.report import cp_secondary_table
        from tcm_eval.scorers import ScoredItem

        items = [
            ScoredItem(
                f"g{i}", "cp", "M3", "m", 0,
                {"exact": 1.0, "subtask": "CP4G_cross_disease_treatment",
                 "cp_family": "CP4G"},
            )
            for i in range(5)
        ]
        self.assertIn("CP4G", cp_secondary_table(items))


class V6_1_ContaminationDetector(unittest.TestCase):
    """Validate the detector before believing any null result it produces."""

    _GRAPH: List[Any] = []

    @classmethod
    def graph_text(cls):
        if not cls._GRAPH:
            from tcm_eval.contamination import GraphText

            cls._GRAPH.append(GraphText.from_kg(kg()))
        return cls._GRAPH[0]

    def _stratum(self, text, cid="probe", answer="x"):
        from tcm_eval.contamination import audit_case

        return audit_case(
            {"id": cid, "clinical_data": text, "options": {"A": answer},
             "answer_letters": ["A"]},
            self.graph_text(),
            "sdt",
        )

    def test_a_verbatim_copy_is_caught(self):
        passage = next(p for p in self.graph_text().passages if len(p) > 60)
        row = self._stratum(passage)
        self.assertEqual(row["stratum"], "likely")
        self.assertGreater(row["ngram_jaccard"], 0.9)

    def test_a_gold_answer_lifted_from_the_graph_is_caught(self):
        syndrome = next(
            n for n in kg().nodes.values()
            if str(n.type) == "Syndrome" and len(n.sentence()) > 20
        )
        row = self._stratum("患者主诉不适。", answer=syndrome.sentence())
        self.assertTrue(row["answer_exact_in_graph"])
        self.assertEqual(row["stratum"], "likely")

    def test_realistic_rewrites_are_caught(self):
        """A leaked record is rewritten between sources, not copied."""
        passage = next(p for p in self.graph_text().passages if len(p) > 80)
        clauses = [c for c in re.split(r"[。；，]", passage) if c]
        rng = random.Random(7)
        variants = {
            "punctuation swapped": passage.replace("，", "；").replace("。", "．"),
            "clauses reordered": "，".join(rng.sample(clauses, len(clauses))) + "。",
            "embedded in a longer case": "患者男性，52岁。" + passage + "余无异常。",
            "half the clauses kept": "，".join(clauses[: max(1, len(clauses) // 2)]) + "。",
        }
        for label, text in variants.items():
            with self.subTest(label):
                self.assertNotEqual(
                    self._stratum(text)["stratum"], "clean", f"{label} was missed"
                )

    def test_unrelated_text_is_clean(self):
        row = self._stratum("The quick brown fox jumps over the lazy dog. " * 8)
        self.assertEqual(row["stratum"], "clean")
        self.assertEqual(row["ngram_jaccard"], 0.0)

    def test_a_short_generic_string_is_not_treated_as_a_leak(self):
        """血瘀证 is in hundreds of guidelines and proves nothing."""
        self.assertIsNone(self.graph_text().contains_exact("血瘀"))

    def test_the_detector_calls_tcm_cp_contaminated(self):
        """CP's gold comes from this graph, so a detector that misses it is broken."""
        from tcm_eval.contamination import audit_dataset

        items = [i for i in built_items() if i["subtask"] == "CP3_stage_actions"][:40]
        for item in items:
            item = dict(item)
            item["answer_letters"] = item["answer"]
        report = audit_dataset(
            [{**i, "answer_letters": i["answer"]} for i in items],
            self.graph_text(),
            "cp",
        )
        self.assertGreater(
            report["strata"]["likely"] / report["n_cases"], 0.5,
            "the detector does not see contamination that is there by construction",
        )

    def test_containment_catches_a_short_passage_inside_a_long_case(self):
        from tcm_eval.contamination import containment, ngrams

        short, long = "舌淡红苔薄白脉细弱", "患者男性五十二岁，" + "舌淡红苔薄白脉细弱" + "，余无异常，随访三月。"
        self.assertEqual(containment(ngrams(short), ngrams(long)), 1.0)
        # ...which Jaccard would have understated
        from tcm_eval.contamination import jaccard

        self.assertLess(jaccard(ngrams(short), ngrams(long)), 0.6)

    def test_normalisation_ignores_width_and_punctuation(self):
        from tcm_eval.contamination import normalise

        self.assertEqual(normalise("询问病史，检查乳房"), normalise("询问病史；检查乳房"))
        self.assertEqual(normalise("ABC１２３"), normalise("ABC123"))


class V6_1_ContaminationReporting(unittest.TestCase):
    def test_the_report_says_so_when_no_audit_was_run(self):
        from tcm_eval.report import contamination_table
        from tcm_eval.scorers import ScoredItem

        items = [ScoredItem("a", "sdt", c, "m", 0, {"sdt_composite": 0.5})
                 for c in ("M1", "M2")]
        table, note = contamination_table(items, "sdt", "sdt_composite")
        self.assertEqual(table, "")
        self.assertIn("No contamination audit", note)

    def test_the_clean_subset_delta_is_computed_separately(self):
        from tcm_eval.report import contamination_table
        from tcm_eval.scorers import ScoredItem

        items = []
        for i in range(20):
            # contaminated cases show a large gain, clean ones none
            stratum = "clean" if i < 10 else "likely"
            gain = 0.0 if stratum == "clean" else 1.0
            for condition, v in (("M1", 0.0), ("M2", gain)):
                items.append(
                    ScoredItem(f"c{i}", "sdt", condition, "m", 0,
                               {"sdt_composite": v, "contamination_stratum": stratum},
                               {"parity_error": ""})
                )
        table, note = contamination_table(items, "sdt", "sdt_composite")
        self.assertIn("clean 10", note)
        row = next(l for l in table.splitlines() if "M1→M2" in l)
        cells = [c.strip() for c in row.split("|")]
        self.assertEqual(float(cells[5]), 0.5)   # all cases
        self.assertEqual(float(cells[7]), 0.0)   # clean only -- the gain was the leak


class V6_5_CrashSafeTraces(unittest.TestCase):
    def setUp(self):
        self.path = Path(
            "/tmp/claude-0/-home-user-Tao-Harness/"
            "3cd2e4a3-0b8c-5ba8-bd7b-9d1ce070b31e/scratchpad/v6_traces.jsonl"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _traces(self, n):
        return [Trace("r", f"c{i}", "sdt", "M0", "m", "fw") for i in range(n)]

    def test_a_write_leaves_no_temp_file(self):
        write_traces(self.path, self._traces(3))
        self.assertEqual(list(self.path.parent.glob(f"{self.path.name}.*.tmp")), [])
        self.assertEqual(len(read_traces(self.path)), 3)

    def test_a_torn_final_line_costs_one_case_not_the_file(self):
        write_traces(self.path, self._traces(5))
        self.path.write_text(self.path.read_text(encoding="utf-8")[:-40], encoding="utf-8")
        self.assertEqual(len(read_traces(self.path)), 4)

    def test_corruption_in_the_middle_raises(self):
        """Skipping it would drop cases from a paid run without saying so."""
        write_traces(self.path, self._traces(5))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        lines[2] = "{not json"
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_traces(self.path)

    def test_a_failed_write_does_not_destroy_the_previous_file(self):
        write_traces(self.path, self._traces(3))

        class Exploding(Trace):
            def to_dict(self):
                raise RuntimeError("disk full")

        with self.assertRaises(RuntimeError):
            write_traces(self.path, [Exploding("r", "c", "sdt", "M0", "m", "fw")])
        self.assertEqual(len(read_traces(self.path)), 3, "the old traces were lost")
        self.assertEqual(list(self.path.parent.glob(f"{self.path.name}.*.tmp")), [])


class V6_5_ProviderParity(unittest.TestCase):
    def test_both_provider_clients_send_the_decode_seed(self):
        """Otherwise samples>1 draws under different rules per provider."""
        import inspect

        from tcm_models.providers import GeminiClient, OpenAICompatClient

        for client in (OpenAICompatClient, GeminiClient):
            source = inspect.getsource(client)
            self.assertIn(
                "decode.seed", source, f"{client.__name__} never sends the seed"
            )

    def test_the_smoke_command_exists_and_is_wired(self):
        from runner.benchmark_runner import cmd_smoke, main  # noqa: F401

        self.assertTrue(callable(cmd_smoke))


class V6_5_TurnMatchedNaming(unittest.TestCase):
    def test_the_controls_are_described_as_turn_matched(self):
        from tcm_eval.report import CONTRASTS

        labels = " ".join(label for _l, _r, label in CONTRASTS)
        self.assertIn("turn-matched", labels)
        self.assertNotIn("compute-matched", labels)

    def test_the_parity_table_reports_a_token_ratio(self):
        from tcm_eval.report import turn_parity_table
        from tcm_eval.scorers import ScoredItem

        items = []
        for condition, calls, tokens in (("M2C", 3, 2000), ("M3", 3, 2470)):
            for i in range(4):
                items.append(
                    ScoredItem(f"c{i}", "sdt", condition, "m", 0, {"exact": 1.0},
                               {"n_llm_calls": calls, "total_tokens": tokens,
                                "parity_error": "", "branch_group": f"c{i}#0"})
                )
        table = turn_parity_table(items)
        self.assertIn("token ratio", table)
        self.assertIn("turn parity", table)


if __name__ == "__main__":
    unittest.main()
