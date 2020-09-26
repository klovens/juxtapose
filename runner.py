import os
import csv
import json
import argparse
import copy
import warnings
import random
from scipy import stats
from itertools import combinations
from sklearn import linear_model
from scipy.stats.mstats import gmean
from network_stats import alignment_permutation_test 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralBiclustering
from dimensionality_reduction import *
from gensim.models import Word2Vec
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from netwalk.translator import IDCovertor
import logging
from collections import Counter
from multiprocessing import Pool
from scipy.spatial.distance import cdist
logging.basicConfig(level=logging.DEBUG, filemode='w', filename='Experiment.log')


warnings.filterwarnings("ignore", category=DeprecationWarning)


def cdist2(x1, x2, metric):
    if metric == 'angular':
        d = 1 - cdist(x1, x2, metric='cosine')
        return np.clip(a=d, a_min=-1, a_max=1) / np.pi
    else:
        return cdist(x1, x2, metric)

def linear_transform(path_1, path_2, mbd_dim):
    model_1 = Word2Vec.load(path_1)
    model_2 = Word2Vec.load(path_2)
    genes_1 = sorted(model_1.wv.vocab)
    vocab_1_size = len(genes_1)
    genes_2 = sorted(model_2.wv.vocab)
    vocab_2_size = len(genes_2)

    backbone = [g for g in genes_1 if "pseudo" in g]

    # transform with linear regression
    model = linear_model.LinearRegression()
    model.fit(model_1.wv[backbone], model_2.wv[backbone])

    # transform network 1 model
    for g in genes_1:
        model_1.wv[g] = np.array(model.predict([model_1.wv[g]]))
    # transform network 2 model
    for g in genes_2:
        model_2.wv[g] = np.array(model.predict([model_2.wv[g]]))

    dist1 = np.zeros((vocab_1_size, vocab_2_size))
    for i, gene_i in enumerate(genes_1):
        for j, gene_j in enumerate(genes_2):
            dist1[i, j] = cosine_similarity(
                model_1[gene_i].reshape(1, mbd_dim),
                model_2[gene_j].reshape(1, mbd_dim))
    return dist1, genes_1, genes_2


def angular_dist(sim_matrix, genes1, genes2):
    vocab_size1 = len(genes1)
    vocab_size2 = len(genes2)
    dist = np.zeros((vocab_size1, vocab_size2))
    for i, gene_i in enumerate(genes1):
        for j, gene_j in enumerate(genes2):
            dist[i, j] = np.arccos(np.clip(a=sim_matrix[i, j], a_min=-1, a_max=1)) / np.pi
    return dist


def match_dims(sim_matrix):
    numrows, numcols = sim_matrix.shape
    max_sim = np.amax(sim_matrix) + 1
    if numrows > numcols:
        slack = numrows - numcols
        new_cols = np.ones((numrows, slack)) * max_sim
        sim_matrix = np.concatenate((sim_matrix, new_cols), axis=1)
    elif numrows < numcols:
        slack = numcols - numrows
        new_rows = np.ones((slack, numcols)) * max_sim
        sim_matrix = np.concatenate((sim_matrix, new_rows), axis=0)

    return sim_matrix


def compare_anchors(dist_matrix, genes_1, genes_2, train_anchors, convertor, substr='pseudo_'):
    gene_ids_1 = convertor.ints2ids(genes_1)
    gene_ids_2 = convertor.ints2ids(genes_2)
    sub = [substr+x for x in train_anchors]

    for prefix in sub:
        anchor_dist = []
        dangle_1 = [s for s in gene_ids_1 if prefix in s]
        dangle_2 = [s for s in gene_ids_2 if prefix in s]
        # convert dangle ids to ints
        dangle_1 = convertor.ids2ints(dangle_1)
        dangle_2 = convertor.ids2ints(dangle_2)
        idx_1 = []
        idx_2 = []
        for i, item in enumerate(genes_1):
            for d in dangle_1:
                if int(item) == d:
                    idx_1.append(i)
        for i, item in enumerate(genes_2):
            for d in dangle_2:
                if int(item) == d:
                    idx_2.append(i)

        for i,j in zip(idx_1, idx_2):
            anchor_dist.append(dist_matrix[i, j])

        pvalue = 0
        for i in range(0,1000):
            rand_dist = []
            for i in range(0,len(dangle_1)):
                rand_int = random.randint(0, min(len(genes_1)-1, len(genes_2)-1))
                rand_dist.append(dist_matrix[rand_int, rand_int])
            pvalue = pvalue + (sum(anchor_dist) > sum(rand_dist))
    # get the test statistic and p-value
    # statistic, pvalue = stats.ttest_ind(anchor_dist, rand_dist)
        print(pvalue/1000)

    return pvalue


def biclustering(dist, genes_1, genes_2, x_label, y_label, out_file, experiment, id_convertor, n_clusters=3, precent_visualize=0.1):
    model = SpectralBiclustering(n_clusters=n_clusters, n_components=12, n_best=6,
                                 init='random', random_state=1)

    m, n = dist.shape
    assert m == len(genes_1) and n == len(genes_2)
    model.fit(dist)
    rows = [(idx, clust_id) for idx, clust_id in enumerate(model.row_labels_)]
    selected_rows = random.choices(rows, k=int(precent_visualize * len(rows)))
    selected_rows_name = [genes_1[idx] for idx, _ in selected_rows]
    selected_rows_clust_ids = [clust_id for _, clust_id in selected_rows]
    selected_rows_indices = [idx for idx, _ in selected_rows]
    # Slect columns
    cols = [(idx, clust_id) for idx, clust_id in enumerate(model.column_labels_)]
    selected_cols = random.choices(cols, k=int(precent_visualize * len(cols)))
    selected_cols_names = [genes_2[idx] for idx, _ in selected_cols]
    selected_cols_clust_ids = [clust_id for _, clust_id in selected_cols]
    selected_cols_indices = [idx for idx, _ in selected_cols]
    # Selected dist
    selected_dist = dist[selected_rows_indices] [:, selected_cols_indices]
    # Sort rows
    sorted_rows_indices = np.argsort(selected_rows_clust_ids)
    selected_dist = selected_dist[sorted_rows_indices, :]
    selected_row_names = [selected_rows_name[i] for i in sorted_rows_indices]
    #selected_row_names = selected_rows_name[sorted_rows_indices]
    # sort columns
    sorted_cols_indices = np.argsort(selected_cols_clust_ids)
    selected_dist = selected_dist[:, sorted_cols_indices]
    selected_cols_names = [selected_cols_names[i] for i in sorted_cols_indices]

    result = pd.DataFrame(selected_dist, columns=selected_cols_names, index=selected_rows_name)

    ax = sns.heatmap(result, cmap="Greens_r", square=True)
    plt.title("Biclustering Results")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel('{} genes'.format(x_label))
    ax.set_xlabel('{} genes'.format(y_label))
    figure = ax.get_figure()
    figure.savefig(out_file)
    plt.close()

    for bic in range(n_clusters*n_clusters):
        #print(bic)
        r = list(model.rows_[bic])
        rows = [i for (i, b) in zip(genes_1, r) if b]

        c = list(model.columns_[bic])
        columns = [i for (i, b) in zip(genes_2, c) if b]

        rows = id_convertor.ints2ids([int(k) for k in rows])
        columns = id_convertor.ints2ids([int(k) for k in columns])

        cluster_path = os.path.join(experiment, f'{bic}_{x_label}_{y_label}_biclustering.csv')
        with open(cluster_path, 'w') as fout:
            fout.write(','.join(rows))
            fout.write("\n")
            fout.write(','.join(columns))

def get_distance(path_1, path_2, mbd_dim):
    model_1 = Word2Vec.load(path_1)
    model_2 = Word2Vec.load(path_2)
    genes_1 = sorted(model_1.wv.vocab)
    vocab_1_size = len(genes_1)
    genes_2 = sorted(model_2.wv.vocab)
    vocab_2_size = len(genes_2)
    logging.info(f'Read {vocab_1_size} gene from the model in {path_1}')
    logging.info(f'Read {vocab_2_size} gene from the model in {path_2}')
    x1 = np.array([model_1.wv[gene_i] for gene_i in genes_1])
    x2 = np.array([model_2.wv[gene_i] for gene_i in genes_2])
    dist = cdist2(x1, x2, 'cosine')/2
    return dist, genes_1, genes_2


def make_heatmap(dist, image_path):
    df = pd.DataFrame(dist)
    plt.figure()
    ax = sns.heatmap(df, cmap='Greens_r', square=True)
    ax.tick_params(left=False, bottom=False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    figure = ax.get_figure()
    figure.savefig(image_path)
    plt.close()


def read_anchors(anchor_path, non_anchor_path):
    with open(anchor_path, 'r') as f:
        anchors = list(csv.reader(f, delimiter=','))
    with open(non_anchor_path, 'r') as f:
        non_anchors = list(csv.reader(f, delimiter=','))
    return anchors, non_anchors


def train(params):
    for pheno in params['phenotypes']:
        for rep_id in range(params['n_replicates']):
            rep_id = str(rep_id)
            walks_path = os.path.join(params['experiment_name'],
                                      '{}_{}_walks.csv'.format(pheno, rep_id))
            with open(walks_path) as fin:
                walks = list(csv.reader(fin))
            model = Word2Vec(sentences=walks,
                             size=params['embd_dim'],
                             window=params['window'],
                             min_count=params['min_count'],
                             workers=params['n_workers'],
                             iter=params['n_iter'],
                             negative=params['negatives'],
                             alpha=params['alpha'],
                             sg = 1,
                             min_alpha=params['min_alpha'])
            # Write model to file
            model.save(os.path.join(params['experiment_name'],
                                    '{}_{}.model'.format(pheno, rep_id)))


def visualize(params):
    for rep_id in range(params['n_replicates']):
        rep_id = str(rep_id)
        for pheno_1, pheno_2 in combinations(params['phenotypes'], 2):
            path_1 = os.path.join(params['experiment_name'],
                                  '{}_{}.model'.format(pheno_1, rep_id))
            path_2 = os.path.join(params['experiment_name'],
                                  '{}_{}.model'.format(pheno_2, rep_id))
            viz_path = os.path.join(params['experiment_name'],
                                    '{}_vs_{}_{}.pdf'.format(pheno_1, pheno_2,
                                                             rep_id))
            make_heatmap(path_1, path_2,
                         params['embd_dim'],
                         image_path=viz_path)


if __name__ == '__main__':
    anchor_stats = []
    parser = argparse.ArgumentParser(description='Generate datasets')
    parser.add_argument('-c', '--config', metavar='JSON file path',
                        action='store', required=True,
                        help='Path to a config file')
    parser.add_argument('-n', '--no-train', dest='no_train',
                        action='store_true', default=False,
                        help='Skip training and only produce visualizations.')
    args = parser.parse_args()
    # read training config
    with open(args.config) as fin:
        params = json.load(fin)
    logging.info('Read parameters')
    # Train models for all replicates
    if args.no_train is False:
        train(params)
        logging.info('Start training ...')
        
    # Generate visualizations
    train_anchor_path = os.path.join(params['experiment_name'], 'train_anchors.csv')
    test_anchor_path = os.path.join(params['experiment_name'], 'test_anchors.csv')
    
    train_anchors, test_anchors = read_anchors(anchor_path=train_anchor_path, non_anchor_path=test_anchor_path)
    logging.info('Using {} potential anchors for training and {} for testing'.format(len(train_anchors),
                                                                                     len(test_anchors)))
    convertor_file = os.path.join(params['experiment_name'],'IDConvertor.json')
    id_convertor = IDCovertor.load(convertor_file)

    for rep_id in range(params['n_replicates']):
        logging.info(f'Start working on replicate {rep_id}')
        rep_id = str(rep_id)
        for pheno_1, pheno_2 in combinations(params['phenotypes'], 2):
            logging.info(f'Start working on {pheno_1} and {pheno_2}')
            path_1 = os.path.join(params['experiment_name'],
                                  '{}_{}.model'.format(pheno_1, rep_id))
            path_2 = os.path.join(params['experiment_name'],
                                  '{}_{}.model'.format(pheno_2, rep_id))
            viz_path = os.path.join(params['experiment_name'],
                                    '{}_vs_{}_{}.pdf'.format(pheno_1, pheno_2,
                                                             rep_id))

            dist, gene_names_1, gene_names_2 = get_distance(path_1, path_2, params['embd_dim'])
            logging.info('Calculated distance matrix')
            #ang_dist = angular_dist(dist, gene_names_1, gene_names_2)
            ang_dist = dist
            del(dist)
            logging.info('Calculated angular distance matrix')
            #make_heatmap(ang_dist, viz_path)
            #pvalue = alignment_permutation_test(vocab1_length=len(gene_names_1), vocab2_length=len(gene_names_2), distance=ang_dist, actual_score=global_cost)
            #print(pvalue)
            bic_path = os.path.join(params['experiment_name'],
                                    '{}_vs_{}_{}_biclustering.pdf'.format(pheno_1, pheno_2,
                                                                          rep_id))
            biclustering(ang_dist, gene_names_1, gene_names_2, pheno_1, pheno_2, bic_path, params['experiment_name'], id_convertor, n_clusters=5)
            #logging.info('Finsihed biclustering')

            expanded_matrix = match_dims(ang_dist)
            #logging.info('Expanded the angular distance matrix')
            row_ind, col_ind = linear_sum_assignment(expanded_matrix)
            #logging.info('Calculated the Huangarian distance matrix')
            globa_scores = expanded_matrix[row_ind, col_ind]
            global_cost = globa_scores.sum()
            norm = max(len(row_ind), len(col_ind))
            global_cost = global_cost/norm
            print(pheno_1, pheno_2, global_cost)
            #matches = [i for i, j in zip(row_ind, col_ind) if i == j]
            #print(len(matches)/norm)
        #pvalue = compare_anchors(ang_dist, gene_names_1, gene_names_2, train_anchors[int(rep_id)], convertor=id_convertor)
        #anchor_stats.append(pvalue)

    #with open(os.path.join(params['experiment_name'],'stats.csv'),'w') as fout:
    #    csv_out = csv.writer(fout)
    #    csv_out.writerow(['statistic','pval'])
    #    for row in anchor_stats:
    #        csv_out.writerow(row)


