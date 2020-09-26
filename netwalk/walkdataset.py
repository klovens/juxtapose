''' This module contains WalkDataset.
'''
from torch.utils.data import Dataset
from netwalk.utils import Vocabulary


class WalkDataset(Dataset):
    ''' Create a dataset of walks, where each walk is a sequence of genes.

    Args:
        original_walks: a nested list of gene names/IDs.
        vocab: A Vocabulary object including all genes in the original_walks.
    '''
    def __init__(self, original_walks, vocab):
        super(WalkDataset, self).__init__()
        self.vocab = vocab
        self.walks = self._vocab_index(original_walks, vocab)

    @staticmethod
    def _vocab_index(original_walks, vocab):
        ''' Translate walks from original node names to integer indices.
        Args:
            original_walks: The original walks, where each node is represented
                by node name/ID.
            vocab: A Vocabulary object including all genes in the original_walks.
        Returns:
            A translated version of original_walks, where each node is
                represented with an integer index. These indices starts with
                0 to n with no gap, where n is the number of different nodes
                present in at least one of the walks in original_walks.

        '''
        walks = []
        for walk in original_walks:
            walks.append([vocab.index[name] for name in walk])
        return walks

    @classmethod
    def read_csv(cls, address, sep):
        ''' Read dataset from a file.

        Args:
            address: Address of a CSV file containing walks.
            sep: A field delimiter.
        Returns:
            data: A list of walks.
            genes: The set of all genes that appear in at least one walk.
        '''
        data = []
        genes = set()
        with open(address) as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                walk = [gene.strip() for gene in line.split(sep)]
                data.append(walk)
                genes.update(walk)
        return data, genes

    @classmethod
    def from_csv(cls, address, sep):
        ''' Create a WalkDataset from a CSV file.

        Args:
            address: Address of a CSV file containing walks.
            sep: A field delimiter.
        Returns:
            A WalkDataset object.
        '''
        data, genes = cls.read_csv(address, sep)
        vocab = Vocabulary(genes)
        return cls(data, vocab)

    def __getitem__(self, idx):
        return self.walks[idx]

    def __len__(self):
        return len(self.walks)


class PairWalkDataset(WalkDataset):
    ''' Create a dataset of walks, where each walk is a sequence of genes.

    Args:
        original_walks: a nested list of gene names/IDs.
        vocab: A Vocabulary object including all genes in the original_walks.
    '''
    def __init__(self, original_walks, vocab):
        super(PairWalkDataset, self).__init__(original_walks, vocab)


    @classmethod
    def from_csv(cls, address, sep):
        ''' Create a WalkDataset from a CSV file.

        Args:
            address: Address of a CSV file containing walks.
            sep: A field delimiter.
        Returns:
            A WalkDataset object.
        '''
        data = []
        pair_walks, genes = cls.read_csv(address, sep)
        for walk_walk in pair_walks:
            middle = len(walk_walk) // 2
            walk_a, walk_b = walk_walk[:middle], walk_walk[middle:]
            data.append((walk_a, walk_b))
        vocab = Vocabulary(genes)
        return cls(data, vocab)

    @staticmethod
    def _vocab_index(original_walks, vocab):
        ''' Translate walks from original node names to integer indices.
        Args:
            original_walks: The original walks, where each node is represented
                by node name/ID.
            vocab: A Vocabulary object including all genes in the original_walks.
        Returns:
            A translated version of original_walks, where each node is
                represented with an integer index. These indices starts with
                0 to n with no gap, where n is the number of different nodes
                present in at least one of the walks in original_walks.

        '''
        walks = []
        for walk_a, walk_b in original_walks:
            translated_walk_a = [vocab.index[name] for name in walk_a]
            translated_walk_b = [vocab.index[name] for name in walk_b]
            walks.append((translated_walk_a, translated_walk_b))
        return walks
