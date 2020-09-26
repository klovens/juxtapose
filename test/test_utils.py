import unittest
from netwalk.utils import Vocabulary
from netwalk.utils import load
from netwalk.utils import Similarity

class TestUtils(unittest.TestCase):
    def test_load(self):
        similarity = load("data/similarity_file.csv", sep=",")
        expected = {("gene1", "gene2"): 0.5,
                    ("gene2", "gene3"): 0.7,
                    ("gene1", "gene3"): 0.0}

        self.assertDictEqual(similarity, expected)

    def test_Vocabulary(self):
        genes = ['g0', 'g1', 'g2', 'g3', 'g4']
        id_2_name_map = {0: 'g0', 1: 'g1', 2: 'g2', 3: 'g3', 4: 'g4'}
        name_2_id_map = {'g0': 0, 'g1': 1, 'g2': 2, 'g3': 3, 'g4': 4}
        vocab = Vocabulary(genes)
        self.assertDictEqual(vocab.index, name_2_id_map)
        self.assertDictEqual(vocab.name, id_2_name_map)
        self.assertListEqual(vocab.genes, genes)

    def test_Similarity(self):
        d = {("gene1", "gene2"): 0.5,
             ("gene2", "gene3"): 0.7,
             ("gene1", "gene3"): 0.6,
             ("gene1", "gene4"): 0.3}
        similarity = Similarity(d)
        symmetric_keys = similarity.symmetric_key_set()
        expected_sym_keys = [("gene1", "gene2"), ("gene2", "gene1"), ("gene2", "gene3"), ("gene3", "gene2"),
                             ("gene1", "gene3"), ("gene3", "gene1"), ("gene1", "gene4"), ("gene4", "gene1")]

        assert set(expected_sym_keys) == set(symmetric_keys)
