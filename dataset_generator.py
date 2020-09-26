''' This module generate walk datasets.
'''
import argparse
import random
import numpy as np
import netwalk.utils as utils
import os.path
from netwalk.walk import WalkGenerator
from similarity import Similarity
import gensim.models
import time
import multiprocessing as mp
import json
import pandas as pd
import seaborn as sns
import copy

def generate_walks(edge_list_address, walk_per_node, walk_length, workers = 4):
    similarity = Similarity(correlation_file_path=edge_list_address, anchors=[],
                            alphas=[], sep=',', prefix='pseudo')
    genes = list(similarity.idx.keys())
    start_time = time.time()
    gen_walk = WalkGenerator(similarity.matrix, genes, walk_length, walk_per_node)
    print("takes {} seconds to create walk object.".format(
        time.time() - start_time))

    num_cpus = workers
    pool = mp.Pool(num_cpus)
    arguments = list(range(len(gen_walk)))
    chunk_size = len(gen_walk) // num_cpus
    walks = pool.map(gen_walk, arguments, chunksize=chunk_size)
    return walks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate datasets')
    parser.add_argument('-c', '--config', metavar='JSON file path',
                        action='store', required=True,
                        help='Path to a config file')
    args = parser.parse_args()
    # read config file
    with open(args.config) as fin:
        params = json.load(fin)
    # make walks and train the network
    for pheno in params['phenotypes']:
        for rep_id in range(params['n_replicates']):
            edge_list_address = os.path.join(params['experiment_name'],
                                             'translated_anchored_{}_{}.csv'.format(pheno,
                                                                str(rep_id)))
            # Create walks
            walks = generate_walks(edge_list_address, params['walk_per_node'],
                                   params['walk_length'], workers=params['n_workers'])

            # Write walks to file
            address = os.path.join(params['experiment_name'],
                                   '{}_{}_walks.csv'.format(pheno, str(rep_id)))
            with open(address, 'w') as fout:
                for w in walks:
                    fout.write('{}\n'.format(','.join([str(s) for s in w])))



