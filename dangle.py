import argparse

from similarity import Similarity
import numpy as np
import pandas as pd
import random
import os
import json
from sklearn.model_selection import train_test_split


def build_backbone(anchors, alphas, weight, edge_percentage):
    dangling = {}
    for anchor, alpha in zip(anchors,alphas):
        pseudo_anchor = 'pseudo_{}'.format(anchor)
        dangles = dangling_structure(pseudo_anchor,
                                     alpha,
                                     weight,
                                     edge_percentage)
        dangles[(anchor, pseudo_anchor)] = weight
        dangling.update(dangles)
    return dangling


def dangling_structure(gene, alpha, weight, edge_percentage):
    num_dangles = alpha
    dangles = ['{}_{:0>3d}'.format(gene, i) for i in range(alpha)]
    sim = {}
    potential_edges = []
    for gene_i in dangles:
        for gene_j in dangles:
            if gene_i == gene_j:
                break
            else:
                potential_edges.append((gene_i, gene_j))
    random.shuffle(potential_edges)
    connected_genes = set()
    for gene_i, gene_j in potential_edges:
        if {gene_i, gene_j} < connected_genes:
            continue
        elif len(connected_genes) < num_dangles:
            connected_genes.add(gene_i)
            connected_genes.add(gene_j)
            sim[(gene_i, gene_j)] = weight

    sim[(gene, dangles[0])] = weight
    for gene_i, gene_j in potential_edges:
        if random.random() < edge_percentage:
            sim[(gene_i, gene_j)] = weight

    return sim


def main(experiment_name, phenotypes, data_directory, anchor_genes,
         num_replicates=1, percent=0.4, num_anchors=50, min_dangle_size=3,
         max_dangle_size=10, test_ratio=0.5):
    assert isinstance(phenotypes, list)
    alphas = random.choices(range(min_dangle_size, max_dangle_size),
                            k=int(num_anchors * test_ratio))
    assert len(alphas) < len(anchor_genes)
    anchor_train_groups = []
    anchor_test_groups = []
    backbones = []
    # Create all backbones
    for rep_id in range(num_replicates):
        random.shuffle(anchor_genes)
        candidates = anchor_genes[:int(num_anchors)]
        genes_of_interest_train, genes_of_interest_test = train_test_split(
            candidates,
            shuffle=True,
            test_size=test_ratio)

        anchor_train_groups.append(genes_of_interest_train)
        anchor_test_groups.append(genes_of_interest_test)
        backbones.append(
            build_backbone(anchors=anchor_train_groups[rep_id], alphas=alphas,
                           weight=1, edge_percentage=percent))
    # Write train anchors to file
    with open(os.path.join(experiment_name, 'train_anchors.csv'), 'w') as fout:
        for gene_group in anchor_train_groups:
            fout.write(','.join(gene_group))
            fout.write("\n")
    # Write test anchors to file
    with open(os.path.join(experiment_name, 'test_anchors.csv'), 'w') as fout:
        for gene_group in anchor_test_groups:
            fout.write(','.join(gene_group))
            fout.write("\n")
    # Adding the backbones and create the similarity object
    for pheno in phenotypes:
        file_name = os.path.join(data_directory, "{}.csv".format(pheno))
        for rep_id in range(num_replicates):
            sim_file_name = "anchored_{}_{}.csv".format(pheno, str(rep_id))
            out_address = os.path.join(experiment_name, sim_file_name)
            similarity = Similarity(file_name,
                                    anchors=anchor_train_groups[rep_id],
                                    alphas=alphas, string_id=True)
            similarity.transform()
            similarity.apply_threshold(lower_cor=0.2, upper_cor=0.8,
                                       value=0)
            similarity.augment(backbones[rep_id])

            similarity.to_csv(out_address)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dangling structures')
    parser.add_argument('-c', '--config', metavar='JSON file path',
                        action='store', required=True,
                        help='Path to a config file')
    args = parser.parse_args()
    config_file_address = args.config
    with open(config_file_address) as fin:
        params = json.load(fin)
    homeostasis_genes = pd.read_csv(params['anchor_file_address'],
                                    dtype=str).iloc[:,0].values
    main(experiment_name=params['experiment_name'],
             phenotypes=params['phenotypes'],
             data_directory=params['data_directory'],
             anchor_genes=homeostasis_genes,
             num_replicates=params['n_replicates'],
             percent=params['percentage'],
             num_anchors=params['n_anchors'],
             min_dangle_size=params['min_dangle_size'],
             max_dangle_size=params['max_dangle_size'],
             test_ratio=params['test_ratio'])

