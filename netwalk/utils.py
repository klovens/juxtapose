''' This module contains utilities classes and functions.

'''
import os.path
import json
import random
import torch
import copy
import numpy as np

class Vocabulary(object):
    '''Create a bijective mapping between gene name/ID and indices.

    Args:
        genes: An array-like containing gene names/IDs.
    '''

    def __init__(self, genes):
        self.genes = genes
        self.index = dict(zip(sorted(genes), range(len(genes))))
        self.name = {idx: gene for gene, idx in self.index.items()}
        self.dim = len(self.genes)

    def to_indices(self, genes):
        return [self.index[gene] for gene in genes]

    def to_names(self, indices):
        return [self.name[i] for i in indices]

    def __len__(self):
        return len(self.genes)


def load_walks(file_dir='.', prefix='pair_walk', sep=','):
    ''' Read dataset from a file.

        Args:
            address: Address of a CSV file containing walks.
            sep: A field delimiter.
        Returns:
            data: A list of walks.
            genes: The set of all genes that appear in at least one walk.
    '''
    walk_address = os.path.join(file_dir, prefix + '_walks.csv')
    walks = np.genfromtxt(walk_address, dtype=np.uint16, delimiter=sep)
    return walks


def dump_walks(walks, out_dir='.', prefix='pair_walk', sep=','):
    # Create walks file
    pass


