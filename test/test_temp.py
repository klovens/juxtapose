import unittest
from netwalk.temp import *
from netwalk.utils import load


class TestTemp(unittest.TestCase):
    def test_transform(self):
        edge_list = {("gene1", "gene2"): 0.5,
                     ("gene2", "gene3"): -0.7,
                     ("gene1", "gene3"): 0.0}
        transformed = transform(edge_list)
        transformed_dict = {k: val for k, val in transformed.items()}
        expected = {("gene1", "gene2"): 0.75,
                    ("gene2", "gene3"): 0.15,
                    ("gene1", "gene3"): 0.5}
        transformed_vals = list(transformed_dict.values())
        expected_vals = list(expected.values())
        for a, b in zip(transformed_vals, expected_vals):
            self.assertAlmostEqual(a, b, places=5)

        edge_list = load("data/fake_networks/network_2.csv", sep=",")
        transformed = transform(edge_list)
        expected = {("g1", "g2"): 0.9,
                    ("g2", "g3"): 0.55,
                    ("g2", "g1"): 0.9,
                    ("g3", "g2"): 0.55,
                    ("g2", "g4"): 0.75,
                    ("g4", "g3"): 0.55,
                    ("g3", "g1"): 0.75,
                    ("g1", "g3"): 0.75,
                    ("g1", "g4"): 0.10,
                    ("g4", "g1"): 0.10,
                    ("g4", "g2"): 0.75,
                    ("g3", "g4"): 0.55}

        transformed_dict = {k: val for k, val in transformed.items()}
        transformed_vals = list(transformed_dict.values())
        expected_vals = list(expected.values())
        for a, b in zip(transformed_vals, expected_vals):
            self.assertAlmostEqual(a, b, places=5)

    def test_filter(self):
        edge_list = {("gene1", "gene2"): 0.5,
                     ("gene2", "gene3"): -0.7,
                     ("gene1", "gene3"): 0.0}
        transformed = transform(edge_list)
        transformed_dict = {k: val for k, val in transformed.items()}
        filtered = filter(transformed_dict, exclude=(0.3, 0.6))
        filtered_dict = {k: val for k, val in filtered.items()}

        expected = {("gene1", "gene2"): 0.75,
                    ("gene2", "gene3"): 0.15}

        filtered_vals = list(filtered_dict.values())
        expected_vals = list(expected.values())
        for a, b in zip(filtered_vals, expected_vals):
            self.assertAlmostEqual(a, b, places=5)

        edge_list = {("gene1", "gene2"): 0.5,
                     ("gene2", "gene3"): -0.7,
                     ("gene1", "gene3"): 0.0}

        filtered = filter(edge_list, exclude=(0, 0.7))
        filtered_dict = {k: val for k, val in filtered.items()}

        expected = {("gene1", "gene2"): -0.7,
                    ("gene2", "gene3"): 0.0}

        filtered_vals = list(filtered_dict.values())
        expected_vals = list(expected.values())
        for a, b in zip(filtered_vals, expected_vals):
            self.assertAlmostEqual(a, b, places=5)

    def test_overlay_networks(self):
        original_edge_list_1 = Similarity({("gene1", "gene2"): 0.5,
                                           ("gene2", "gene3"): 0.7,
                                           ("gene1", "gene3"): 0.6,
                                           ("gene1", "gene4"): 0.3,
                                           ("gene2", "gene4"): 0.1,
                                           ("gene3", "gene4"): 0.0,
                                           ("gene1", "gene3"): 0.0})

        original_edge_list_2 = Similarity({("gene1", "gene2"): 0.5,
                                           ("gene2", "gene3"): 0.7,
                                           ("gene1", "gene3"): 0.0,
                                           ("gene1", "gene4"): 0.0,
                                           ("gene2", "gene4"): 0.3,
                                           ("gene4", "gene3"): 0.7,
                                           ("gene1", "gene3"): 0.2})

        edge_list_1 = Similarity({("gene1", "gene2"): 0.5,
                                  ("gene2", "gene3"): 0.7,
                                  ("gene1", "gene3"): 0.6,
                                  ("gene1", "gene4"): 0.3})

        edge_list_2 = Similarity({("gene1", "gene2"): 0.5,
                                  ("gene2", "gene3"): 0.7,
                                  ("gene2", "gene4"): 0.3})

        net_1, net_2 = overlay_networks(net_a_similarity=edge_list_1, net_b_similarity=edge_list_2,
                                        original_net_a=original_edge_list_1, original_net_b=original_edge_list_2)

        assert set(net_1.symmetric_key_set()) == set(net_2.symmetric_key_set())
        net_1_dict = {k: val for k, val in net_1.items()}
        net_2_dict = {k: val for k, val in net_2.items()}
        assert set(net_1_dict.values()) == {0.5, 0.7, 0.6, 0.3, 0.1}
        assert set(net_2_dict.values()) == {0.5, 0.7, 0.3, 0.2, 0.0}

    def test_create_spine(self):
        net1 = Similarity.load("../data/fake_networks/network_1.csv", sep=",")
        net2 = Similarity.load("../data/fake_networks/network_2.csv", sep=",")

        expected_spine = ["g1", "g4"]
        expected_pseudo_spine = ["pseudo_g1", "pseudo_g4"]
        expected_similarity = {("pseudo_g1", "pseudo_g10"): 0.5,
                               ("pseudo_g1", "pseudo_g11"): 0.5,
                               ("pseudo_g1", "pseudo_g4"): -0.8,
                               ("pseudo_g4", "pseudo_g40"): 0.5,
                               ("pseudo_g4", "pseudo_g41"): 0.5}

        spine, pseudo_spine, backbone = create_spine(spine=["g1", "g4"], net_a_tsfmd_similarity=net1,
                                                     net_b_tsfmd_similarity=net2,
                                                     prefix='pseudo_', alpha=2, weight=0.5)
        assert set(spine) == set(expected_spine)
        assert set(pseudo_spine) == set(expected_pseudo_spine)

        self.assertEqual(len(backbone), len(expected_similarity))

        for key, val in backbone.items():
            assert key in expected_similarity.keys()
            assert expected_similarity[key] == val

        expected_spine = ["g1", "g4", "g3"]
        expected_pseudo_spine = ["pseudo_g1", "pseudo_g4", "pseudo_g3"]
        expected_similarity = {("pseudo_g1", "pseudo_g3"): 0.5,
                               ("pseudo_g1", "pseudo_g10"): 0.5,
                               ("pseudo_g1", "pseudo_g11"): 0.5,
                               ("pseudo_g4", "pseudo_g1"): -0.8,
                               ("pseudo_g3", "pseudo_g30"): 0.5,
                               ("pseudo_g3", "pseudo_g31"): 0.5,
                               ("pseudo_g4", "pseudo_g40"): 0.5,
                               ("pseudo_g4", "pseudo_g41"): 0.5}

        spine, pseudo_spine, backbone = create_spine(spine=["g1", "g4", "g3"], net_a_tsfmd_similarity=net1,
                                                     net_b_tsfmd_similarity=net2,
                                                     prefix='pseudo_', alpha=2, weight=0.5)

        assert set(spine) == set(expected_spine)
        assert set(pseudo_spine) == set(expected_pseudo_spine)

        self.assertEqual(len(backbone), len(expected_similarity))

        for key, val in backbone.items():
            assert key in expected_similarity.keys()
            assert expected_similarity[key] == val

    def test_add_anchor(self):
        net1 = Similarity.load("../data/fake_networks/network_1.csv", sep=",")
        net2 = Similarity.load("../data/fake_networks/network_2.csv", sep=",")

        spine_similarity_1 = {("pseudo_g1", "pseudo_g3"): 0.5,
                              ("pseudo_g1", "pseudo_g10"): 0.5,
                              ("pseudo_g1", "pseudo_g11"): 0.5,
                              ("pseudo_g4", "pseudo_g1"): -0.8,
                              ("pseudo_g3", "pseudo_g30"): 0.5,
                              ("pseudo_g3", "pseudo_g31"): 0.5,
                              ("pseudo_g4", "pseudo_g40"): 0.5,
                              ("pseudo_g4", "pseudo_g41"): 0.5}

        spine_similarity_2 = {("pseudo_g1", "pseudo_g10"): 0.5,
                              ("pseudo_g1", "pseudo_g11"): 0.5,
                              ("pseudo_g4", "pseudo_g1"): -0.8,
                              ("pseudo_g4", "pseudo_g40"): 0.5,
                              ("pseudo_g4", "pseudo_g41"): 0.5}

        anchored_net1 = add_anchor(net1, pseudo_similarity=backbone, spine_genes=["g1", "g4", "g3"],
                                   pseudo_spine_genes=["pseudo_g1", "pseudo_g4", "pseudo_g3"],
                                   weight=0.5)

        anchored_net2 = add_anchor(net2, pseudo_similarity=backbone, spine_genes=["g1", "g4"],
                                   pseudo_spine_genes=["pseudo_g1", "pseudo_g4"],
                                   weight=0.7)

        for key, val in anchored_net1.items():
            assert key in spine_similarity_1.keys()
        for key, val in anchored_net2.items():
            assert key in spine_similarity_2.keys()

        result_keys = [anchored_net1[x] for x in anchored_net1.keys()]
        self.assertListEqual(result_keys, net1.update(spine_similarity_1))

        result_keys = [anchored_net2[x] for x in anchored_net2.keys()]
        self.assertListEqual(result_keys, net2.update(spine_similarity_2))
