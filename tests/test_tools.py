"""Tool contract: domain isolation, coverage semantics, budgets, determinism."""

import unittest

import tcm_tools  # noqa: F401  (registers the tool surface)
from tcm_kg.schema import Domain
from tcm_tools.base import REGISTRY, Coverage, ToolBudget, ToolContext

from ._fixtures import graph, retriever


def ctx(domain, budget=None):
    return ToolContext(graph(), retriever(), domain, budget or ToolBudget())


class RegistryTests(unittest.TestCase):
    def test_thirteen_tools_are_registered(self):
        self.assertEqual(len(REGISTRY.specs_for(Domain.FULL)), 13)

    def test_no_tool_exposes_a_query_language(self):
        # letting a model write Cypher would make query-language skill a
        # per-model confound in what is meant to be a knowledge comparison
        for spec in REGISTRY.specs_for(Domain.FULL):
            blob = (spec.name + spec.description).lower()
            for banned in ("cypher", "sparql", "select ", "match ("):
                self.assertNotIn(banned, blob, spec.name)

    def test_tool_fingerprint_is_stable(self):
        self.assertEqual(REGISTRY.fingerprint(), REGISTRY.fingerprint())


class DomainIsolationTests(unittest.TestCase):
    def test_medication_tools_are_unreachable_from_the_clinical_domain(self):
        # this is what stops an SDT agent recovering the syndrome by looking at
        # which formula treats it
        for name in (
            "retrieve_medication_knowledge",
            "retrieve_pharmacopeia_entry",
            "retrieve_safety_constraints",
            "check_dose",
        ):
            self.assertNotIn(name, REGISTRY.names_for(Domain.CLINICAL), name)
            result = REGISTRY.call(name, ctx(Domain.CLINICAL), {"name": "x", "herb": "x", "items": ["x"]})
            self.assertFalse(result.ok)
            self.assertIn("not available in domain", result.error or "")

    def test_search_cannot_return_a_forbidden_type(self):
        result = REGISTRY.call(
            "search_tcm_entities",
            ctx(Domain.CLINICAL),
            {"query": "安神定志丸", "entity_types": ["Formula"]},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.coverage, Coverage.NOT_COVERED)

    def test_clinical_search_still_works_for_allowed_types(self):
        result = REGISTRY.call(
            "search_tcm_entities",
            ctx(Domain.CLINICAL),
            {"query": "心悸 善惊易恐 多梦易醒", "entity_types": ["Syndrome"]},
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.data["results"])


class CoverageSemanticsTests(unittest.TestCase):
    def test_dose_is_reported_as_not_covered_not_as_no_problem(self):
        result = REGISTRY.call(
            "check_dose", ctx(Domain.SAFETY), {"items": [{"name": "附子", "dose": "30g"}]}
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.coverage, Coverage.NOT_COVERED)
        finding = result.data["findings"][0]
        self.assertIsNone(finding["dose_range_in_graph"])
        # the graph does hold a genuinely relevant fact, and says so
        self.assertEqual(finding["toxicity_flag"], "有毒")

    def test_combination_refuses_to_rule_on_incompatibility(self):
        # 半夏 and 附子 are a real 十八反 pair, and they co-occur in graph
        # formulas: exactly why co-occurrence must not be read as safety
        result = REGISTRY.call(
            "check_combination", ctx(Domain.SAFETY), {"items": ["半夏", "附子"]}
        )
        self.assertEqual(result.coverage, Coverage.NOT_COVERED)
        pair = result.data["pairs"][0]
        self.assertEqual(pair["incompatibility_verdict"], "not_covered")
        self.assertGreater(pair["n_co_occurrences"], 0)
        self.assertTrue(any("十八反" in c for c in result.caveats))

    def test_empty_is_distinguished_from_not_covered(self):
        # a term the graph has never seen must not read as "no restriction"
        result = REGISTRY.call(
            "retrieve_safety_constraints",
            ctx(Domain.SAFETY),
            {"entity": "完全不存在的药物名称XYZ"},
        )
        self.assertIn(result.coverage, {Coverage.NOT_COVERED, Coverage.EMPTY})
        self.assertTrue(any("不能" in c or "不代表" in c for c in result.caveats))


class DeterministicCheckerTests(unittest.TestCase):
    def test_decoction_requirement_is_grounded_and_attributed(self):
        result = REGISTRY.call(
            "check_decoction_requirement",
            ctx(Domain.SAFETY),
            {"items": ["石膏", "大黄"], "claimed_requirement": "先煎"},
        )
        self.assertEqual(result.coverage, Coverage.SUPPORTED)
        by_item = {f["item"]: f for f in result.data["findings"]}
        self.assertEqual(by_item["石膏"]["claim_verdict"], "supported")
        self.assertEqual(by_item["大黄"]["claim_verdict"], "contradicted_by_graph")

    def test_duplicate_medication_detects_alias_repetition(self):
        result = REGISTRY.call(
            "check_duplicate_medication", ctx(Domain.SAFETY), {"items": ["瓜蒌", "全瓜蒌"]}
        )
        self.assertEqual(len(result.data["alias_duplicates"]), 1)
        self.assertEqual(
            sorted(result.data["alias_duplicates"][0]["listed_as"]), ["全瓜蒌", "瓜蒌"]
        )

    def test_verifier_contradicts_a_mismatched_disease_anchor(self):
        good = REGISTRY.call(
            "verify_tcm_decision",
            ctx(Domain.CLINICAL),
            {"syndrome": "心虚胆怯证", "disease": "心悸（心律失常-室性早搏）"},
        )
        self.assertEqual(good.data["overall"], "supported")
        bad = REGISTRY.call(
            "verify_tcm_decision",
            ctx(Domain.CLINICAL),
            {"syndrome": "心虚胆怯证", "disease": "消渴病肾病（糖尿病肾病）"},
        )
        self.assertEqual(bad.data["overall"], "contradicted")

    def test_verifier_is_silent_rather_than_negative_off_graph(self):
        result = REGISTRY.call(
            "verify_tcm_decision", ctx(Domain.CLINICAL), {"syndrome": "杜撰不存在证"}
        )
        self.assertEqual(result.data["overall"], "not_in_graph")


class BudgetTests(unittest.TestCase):
    def test_global_budget_is_enforced(self):
        context = ctx(Domain.CLINICAL, ToolBudget(max_calls=2, max_calls_per_tool=9))
        for _ in range(2):
            REGISTRY.call("search_tcm_entities", context, {"query": "心悸"})
        blocked = REGISTRY.call("search_tcm_entities", context, {"query": "心悸"})
        self.assertFalse(blocked.ok)
        self.assertIn("budget exhausted", blocked.error)

    def test_per_tool_budget_is_enforced(self):
        context = ctx(Domain.CLINICAL, ToolBudget(max_calls=9, max_calls_per_tool=1))
        REGISTRY.call("search_tcm_entities", context, {"query": "心悸"})
        blocked = REGISTRY.call("search_tcm_entities", context, {"query": "心悸"})
        self.assertIn("per-tool budget", blocked.error)

    def test_every_call_is_recorded_even_when_it_fails(self):
        context = ctx(Domain.CLINICAL)
        REGISTRY.call("no_such_tool", context, {})
        REGISTRY.call("search_tcm_entities", context, {})  # missing required arg
        self.assertEqual(len(context.calls), 2)
        self.assertTrue(all(not c.ok for c in context.calls))

    def test_a_raising_tool_does_not_abort_the_run(self):
        context = ctx(Domain.SAFETY)
        result = REGISTRY.call("check_combination", context, {"items": ["只有一个"]})
        self.assertFalse(result.ok)
        self.assertEqual(len(context.calls), 1)


if __name__ == "__main__":
    unittest.main()
