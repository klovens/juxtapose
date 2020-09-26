import os
import csv
import json
import argparse
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import seaborn as sns
from netwalk.translator import ID2NameTranslator, IDCovertor
from scipy.spatial.distance import cdist
from itertools import combinations

def pca_visualization(model, out_file_name):
    x = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(x)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    #pyplot.xlim(-25,  25)
    #pyplot.ylim(-25, 25)
    words = list(model.wv.vocab)
    #for i, word in enumerate(words):
        #pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.savefig(out_file_name)


def tsne_plot(model, out_file_name, perplexity=30, components=2, init='pca', 
              num_iter=500, rand_state=0):
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=perplexity, n_components=components,
                      init=init, n_iter=num_iter, random_state=rand_state)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    print(labels)
    print(labels["pseudo" in labels])
    c=["royalblue" if "pseudo" in x else "orangered" for x in labels]
    pyplot.figure(figsize=(16, 16))
    for i in range(len(x)):
        pyplot.scatter(x[i], y[i], color=c[i], s=30)

    pyplot.savefig(out_file_name)


def tsne_visualize(model, gene, list_names, vocab_length, num_components, out_file_name, converter, translate):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query gene,
    its list of most similar genes, and a list of other genes.
vv    """
    arrays = np.empty((0, vocab_length), dtype='f')
    gene_labels = [gene]
    color_list  = ['red']

    # adds the vector of the query gene
    arrays = np.append(arrays, model.wv.__getitem__([gene]), axis=0)

    # gets list of most similar genes
    close_genes = model.wv.most_similar([gene])

    # adds the vector for each of the closest genes to the array
    for gne_score in close_genes:
        gne_vector = model.wv.__getitem__([gne_score[0]])
        gene_labels.append(gne_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, gne_vector, axis=0)

    # adds the vector for each of the genes from list_names to the array
    for gne in list_names:
        gne_vector = model.wv.__getitem__([gne])
        gene_labels.append(gne)
        color_list.append('green')
        arrays = np.append(arrays, gne_vector, axis=0)

    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=num_components).fit_transform(arrays)
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    gene_labels = ints_to_names(gene_labels, translate, converter)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'genes': gene_labels,
                       'color': color_list})

    fig, _ = pyplot.subplots()
    fig.set_size_inches(9, 9)

    # Basic plot
    sns.set_style("ticks")
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )

    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["genes"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    pyplot.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    pyplot.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)

    pyplot.title('t-SNE visualization for {}'.format(gene.title()))
    pyplot.savefig(out_file_name)

def names_to_ints(names, trans, convertor):
    ids, names =  trans.names2ids(names)
    ints = [str(i) for i in convertor.ids2ints(ids)]
    return ints, ids, names

def ints_to_names(ints, trans, convertor):
    #int to id
    ids = [i for i in convertor.ints2ids(ints)]
    #id to name
    names =  [trans.id2name(x) for x in ids]
    return names

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate datasets')
    parser.add_argument('-c', '--config', metavar='JSON file path',
                        action='store', required=True,
                        help='Path to a config file')
    args = parser.parse_args()
    # read training config
    with open(args.config) as fin:
        params = json.load(fin)

    ensemble_id_name_file = params['vocab']
    convertor_file = os.path.join(params['experiment_name'],'IDConvertor.json')
    trans = ID2NameTranslator(ensemble_id_name_file, sep=',')
    convertor = IDCovertor.load(convertor_file)

    for rep_id in range(params['n_replicates']):
        for pheno in params['phenotypes']:
            path = os.path.join(params['experiment_name'],
                                  '{}_{}.model'.format(pheno, rep_id))
            viz_path = os.path.join(params['experiment_name'],
                                        '{}_{}_tsne.pdf'.format(pheno,
                                                                 rep_id))
            pca_path = os.path.join(params['experiment_name'],
                                        '{}_{}_pca.pdf'.format(pheno,
                                                                 rep_id))
            # load model
            model = Word2Vec.load(path)
            genes = list(model.wv.vocab)
            # make tsne
            #tsne_plot(model, out_file_name=viz_path)
            #pca_visualization(model, out_file_name=pca_path)
            names = params['select_genes']
            ints, ids, names = names_to_ints(names, trans, convertor)
            for ind, g, name in zip(ints, ids, names):
                sim_path = os.path.join(params['experiment_name'],
                                        '{}_{}_most_similar_to_{}.pdf'.format(pheno,
                                                                 rep_id, name))
                rand_path = os.path.join(params['experiment_name'],
                                        '{}_{}_random_compared_to_{}.pdf'.format(pheno,
                                                                 rep_id, g))
                # make a visualization of select genes and their most similar genes
                if ind in model.wv.vocab:
                    negative_ints  = [i[0] for i in model.wv.most_similar(negative=[ind])]
                    tsne_visualize(model, ind, list_names=negative_ints, vocab_length=params['embd_dim'], num_components=2, out_file_name=sim_path, translate=trans,converter=convertor)
                #sampled_ints  = random.choices(genes, k=20) 
                #tsne_visualize(model, ind, list_names=sampled_ints,  vocab_length=params['embd_dim'], num_components=2, out_file_name=rand_path, translate=trans,converter=convertor)
        for pheno_1, pheno_2 in combinations(params['phenotypes'], 2): 
            path_1 = os.path.join(params['experiment_name'],
                                  '{}_{}.model'.format(pheno_1, rep_id))
            path_2 = os.path.join(params['experiment_name'],
                                  '{}_{}.model'.format(pheno_2, rep_id))
            model_1 = Word2Vec.load(path_1)
            model_2 = Word2Vec.load(path_2)
            ints, ids, names = names_to_ints(names, trans, convertor)
            targets_in_model1 = [ (gene_i, name_i) for gene_i, name_i in zip(ints, names) if gene_i in model_1.wv.vocab]
            targets_in_model2 = [ (gene_i, name_i) for gene_i, name_i in zip(ints, names) if gene_i in model_2.wv.vocab]
            x1 = np.array([model_1.wv[gene_i] for gene_i, _ in targets_in_model1])
            x2 = np.array([model_2.wv[gene_i] for gene_i, _ in targets_in_model2])
            # Calculate the distance between elements of x1 and x2 as a distance matrix
            dist = cdist(x1, x2, metric='cosine')/2
            df = pd.DataFrame(dist)

            df.columns = [name_i for _, name_i in targets_in_model2]
            df.index = [name_i for _, name_i in targets_in_model1]
            matrix_path = os.path.join(params['experiment_name'],
                                  '{}_{}_{}_matrix.pdf'.format(pheno_1, pheno_2, rep_id))
            pyplot.figure()
            print(df.index, df.columns)
            print(df.shape)
            ax = sns.heatmap(df, square=True, vmin=0, vmax=1)
            #ax.tick_params(left=False, bottom=False)
            ax.set_yticklabels(list(df.index))
            ax.set_xticklabels(list(df.columns))
            figure = ax.get_figure()
            figure.savefig(matrix_path)
            pyplot.close()
