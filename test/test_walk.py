import unittest
import numpy as np
from netwalk.walk import Walk
from netwalk.walk import PairWalk


class TestWalk(unittest.TestCase):
    def test_from_dict(self):
        d = {(1, 2): 0.7,
             (1, 3): 0.1,
             (2, 4): 0.5,
             (3, 4): 0.3}
        similarity = [[0, 7/8, 1/8, 0],
                      [7/12, 0, 0, 5/12],
                      [1/4, 0, 0, 3/4],
                      [0, 5/8, 3/8, 0]]
        CDF = np.array([[0, 0.875, 1, 1],
                        [7/12, 7/12, 7/12, 1],
                        [0.25, 0.25, 0.25, 1],
                        [0, 0.625, 1, 1]])

        walk = Walk(d)
        self.assertListEqual(list(walk._nodes), [1, 2, 3, 4])
        diff = np.array(similarity) - walk.prob
        self.assertAlmostEqual(np.linalg.norm(diff), 0)
        ids = np.array([walk._ids[node] for node in walk._nodes])
        diff = np.linalg.norm(ids - np.arange(len(walk._nodes)))
        self.assertAlmostEqual(diff, 0)
        diff = np.linalg.norm(walk.cdf - CDF)
        self.assertAlmostEqual(diff, 0)

    def test_generate(self):
        similarity = {(1, 2): 0.5, (1, 3): 0.0}

        walk = Walk(similarity)
        self.assertListEqual(walk.generate(3, 3), [2, 2, 2, 2])

        self.assertListEqual((walk.generate(1, 5)), [0, 1, 0, 1, 0, 1])
        self.assertListEqual((walk.generate(2, 5)), [1, 0, 1, 0, 1, 0])

    def test_make_walks(self):
        similarity = {(1, 2): 0.5, (1, 3): 0.0}
        walk = Walk(similarity)
        dataset = walk.make_walks(walk_per_node=2, walk_length=3)
        expected_dataset = [[0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0],
                            [1, 0, 1, 0], [2, 2, 2, 2], [2, 2, 2, 2]]
        for expected, observed in zip(expected_dataset, dataset['walks']):
            self.assertListEqual(expected, list(observed))
        expected_nodes = [1, 2, 3]
        self.assertListEqual(expected_nodes, list(dataset['nodes']))
        expected_ids = {1: 0, 2: 1, 3: 2}
        self.assertEqual(expected_ids, dataset['ids'])


class TestPairWalk(unittest.TestCase):

    def test_generate(self):
        similarity = {(1, 2): 0.5, (1, 3): 0.0}
        walk = PairWalk(similarity)
        self.assertListEqual(walk.generate(3, 3), [2, 2, 2, 2, 2, 2, 2, 2])
        self.assertListEqual((walk.generate(1, 5)), [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.assertListEqual((walk.generate(2, 5)), [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
