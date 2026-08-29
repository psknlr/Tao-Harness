"""Hybrid retrieval: score decomposition, determinism, domain scoping."""

import unittest

from tcm_kg.index import KGRetriever, RetrievalParams
from tcm_kg.schema import Domain

from ._fixtures import graph, retriever

CASE = "心悸，善惊易恐，坐卧不安，多梦易醒，恶闻声响，食少纳呆"


class RetrievalTests(unittest.TestCase):
    def test_symptom_text_retrieves_the_right_syndrome(self):
        # the graph has no Symptom entity; this works only because the index is
        # built over per-entity virtual documents that include the protocol's
        # verbatim definition sentence
        hits = retriever().search(
            CASE, domain=Domain.CLINICAL, node_types=["Syndrome"], top_k=5
        )
        self.assertIn("心虚胆怯证", [h.name for h in hits])

    def test_results_are_deterministic(self):
        first = retriever().search(CASE, domain=Domain.CLINICAL, top_k=8)
        second = retriever().search(CASE, domain=Domain.CLINICAL, top_k=8)
        self.assertEqual([h.node_id for h in first], [h.node_id for h in second])
        self.assertEqual([h.score for h in first], [h.score for h in second])

    def test_domain_scoping_hides_treatment_entities(self):
        hits = retriever().search("安神定志丸加减", domain=Domain.CLINICAL, top_k=10)
        self.assertTrue(all(h.node_type != "Formula" for h in hits))
        safety = retriever().search("安神定志丸加减", domain=Domain.SAFETY, top_k=10)
        self.assertIn("Formula", {h.node_type for h in safety})

    def test_anchors_activate_the_graph_and_source_terms(self):
        disease = graph().find_by_name("心悸（心律失常-室性早搏）", ["Disease"])[0]
        anchored = retriever().search(
            CASE, domain=Domain.CLINICAL, node_types=["Syndrome"],
            anchors=[disease.id], top_k=8,
        )
        self.assertTrue(any(h.graph > 0 for h in anchored))
        self.assertTrue(any(h.source > 0 for h in anchored))
        unanchored = retriever().search(
            CASE, domain=Domain.CLINICAL, node_types=["Syndrome"], top_k=8
        )
        self.assertTrue(all(h.graph == 0 and h.source == 0 for h in unanchored))

    def test_score_is_the_declared_weighted_sum(self):
        params = retriever().params
        hit = retriever().search(CASE, domain=Domain.CLINICAL, top_k=1)[0]
        expected = (
            params.alpha_semantic * hit.semantic
            + params.beta_graph * hit.graph
            + params.gamma_source * hit.source
        )
        self.assertAlmostEqual(hit.score, expected, places=9)

    def test_params_fingerprint_changes_with_any_weight(self):
        base = RetrievalParams().fingerprint()
        self.assertNotEqual(RetrievalParams(alpha_semantic=0.5).fingerprint(), base)
        self.assertNotEqual(RetrievalParams(top_k=4).fingerprint(), base)


if __name__ == "__main__":
    unittest.main()
