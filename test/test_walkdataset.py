import unittest
from netwalk.walkdataset import WalkDataset
from netwalk.walkdataset import PairWalkDataset
from netwalk.utils import Vocabulary


class TestWalkDataset(unittest.TestCase):
    def setUp(self):
        data = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]]
        self.data = data

    def test_from_csv(self):
        dataset = WalkDataset.from_csv('data/sample_walk_dataset.csv', sep=',')
        self.assertListEqual(dataset.walks, self.data)

    def test__len__(self):
        dataset = WalkDataset(original_walks=[], vocab=Vocabulary([]))
        self.assertEqual(len(dataset), 0)
        dataset = WalkDataset.from_csv('data/sample_walk_dataset.csv', sep=',')
        self.assertEqual(len(dataset), 5)

    def test__getitem__(self):
        dataset = WalkDataset.from_csv('data/sample_walk_dataset.csv', sep=',')
        for i, expected in enumerate(self.data):
            self.assertListEqual(expected, dataset[i])


class TestPairedWalkDataset(unittest.TestCase):
    def setUp(self):
        data = [([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
                ([5, 6, 7, 8, 9], [5, 8, 7, 8, 9])]
        self.data = data

    def test_from_csv(self):
        dataset = PairWalkDataset.from_csv('data/sample_pair_walk_dataset.csv', sep=',')
        self.assertEqual(len(self.data), len(dataset))
        for i, observed_walk in enumerate(dataset):
            (expected_walk_a, expected_walk_b) = self.data[i]
            observed_walk_a, observed_walk_b = observed_walk
            self.assertListEqual(expected_walk_a, observed_walk_a)
            self.assertListEqual(expected_walk_b, observed_walk_b)
