"""Knowledge graph: ontology conformance, indexing, and derived facts."""

import unittest

from tcm_kg import load_kg
from tcm_kg.normalize import (
    canonical_department,
    canonical_syndrome,
    char_ngrams,
    markers_for_herb_in_sentence,
    split_herb_annotation,
    syndrome_atoms,
)
from tcm_kg.schema import EDGE_SIGNATURES, EdgeType, NodeType, validate_graph


class NormalizeTests(unittest.TestCase):
    def test_enumeration_and_description_are_stripped(self):
        self.assertEqual(canonical_syndrome("（3）痰瘀郁肺证：胸闷，咳嗽"), "痰瘀郁肺证")
        self.assertEqual(canonical_syndrome("1.心虚胆怯证：心悸"), "心虚胆怯证")
        self.assertEqual(canonical_syndrome("③湿热下注证"), "湿热下注证")

    def test_compound_syndromes_split_into_atoms(self):
        self.assertEqual(syndrome_atoms("痰阻血瘀，湿郁化热证"), ["痰阻血瘀证", "湿郁化热证"])
        self.assertEqual(syndrome_atoms("心虚胆怯证"), ["心虚胆怯证"])

    def test_preparation_annotation_is_split_but_disease_gloss_is_kept(self):
        self.assertEqual(split_herb_annotation("生石膏(先煎)"), ("生石膏", ["先煎"]))
        self.assertEqual(split_herb_annotation("麝香（冲服，或白芷代）"), ("麝香", ["冲服"]))
        # a western-medicine gloss is part of the entity name, not an annotation
        name, markers = split_herb_annotation("心悸（心律失常-室性早搏）")
        self.assertEqual(name, "心悸（心律失常-室性早搏）")
        self.assertEqual(markers, [])

    def test_marker_attribution_is_positional(self):
        sentence = "生地、生石膏、地榆炭、生大黄(后下)等"
        self.assertEqual(markers_for_herb_in_sentence(sentence, ["大黄"]), ["后下"])
        # the marker belongs to 大黄, not to every herb in the sentence
        self.assertEqual(markers_for_herb_in_sentence(sentence, ["石膏"]), [])

    def test_department_synonyms_collapse(self):
        self.assertEqual(canonical_department("呼吸科"), "肺病科")
        self.assertEqual(canonical_department("骨伤科"), "骨伤科")

    def test_char_ngrams_cover_unigram_to_trigram(self):
        self.assertEqual(char_ngrams("舌质红"), ["舌", "质", "红", "舌质", "质红", "舌质红"])


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kg = load_kg()

    def test_graph_matches_declared_ontology(self):
        problems = validate_graph(
            {n.id: n.type for n in self.kg.nodes.values()},
            [
                {"type": e.type, "from": e.source, "to": e.target}
                for e in self.kg.edges
            ],
        )
        self.assertEqual(problems, {})

    def test_all_fourteen_node_types_are_populated(self):
        counts = self.kg.type_counts()
        self.assertEqual(len(counts), 14)
        for node_type in NodeType:
            self.assertGreater(counts.get(node_type.value, 0), 0, node_type.value)

    def test_no_symptom_or_pathogenesis_entity_exists(self):
        # the study's central ontology claim: these are runtime variables, not
        # graph entities, and nothing may quietly add them
        names = {t.value for t in NodeType}
        self.assertNotIn("Symptom", names)
        self.assertNotIn("Pathogenesis", names)

    def test_identity_clusters_resolve_aliases(self):
        node = self.kg.find_by_name("全瓜蒌")[0]
        cluster = {n.name for n in self.kg.cluster(node.id)}
        self.assertIn("瓜蒌", cluster)
        self.assertIn("全瓜蒌", cluster)

    def test_provenance_resolves_through_source_docs(self):
        node = self.kg.find_by_name("心虚胆怯证", ["Syndrome"])[0]
        docs = self.kg.documents_for(node.id)
        self.assertTrue(docs)
        self.assertTrue(all(d.type == "DocumentSource" for d in docs))

    def test_preparation_markers_do_not_leak_across_herbs(self):
        gypsum = self.kg.find_by_name("石膏", ["Herb"])[0]
        markers = self.kg.preparation_markers(gypsum.id)
        self.assertIn("先煎", markers)
        self.assertNotIn("后下", markers)

    def test_virtual_document_includes_definition_sentence(self):
        node = self.kg.find_by_name("心虚胆怯证", ["Syndrome"])[0]
        document = self.kg.virtual_document(node.id)
        self.assertIn("善惊易恐", document)

    def test_expand_is_type_filtered(self):
        disease = self.kg.find_by_name("心悸（心律失常-室性早搏）", ["Disease"])[0]
        scores = self.kg.expand([disease.id], hops=1, allowed_types=frozenset({"Syndrome"}))
        self.assertTrue(scores)
        self.assertTrue(
            all(self.kg.node(n).type in {"Syndrome", "Disease"} for n in scores)
        )


if __name__ == "__main__":
    unittest.main()
