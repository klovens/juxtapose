''' This module generates walks from a network.

'''
import numpy as np
import gensim.models
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import time
import multiprocessing as mp


EPSILON = 1E-6

class Probability():
    def __init__(self, matrix, gene_names):
        n = matrix.shape[0]
        assert matrix.shape[0] == matrix.shape[1]
        assert len(gene_names) == n
        total_prob = matrix.sum(axis=1).reshape(n, 1)
        corrections = []
        for i, p in enumerate(total_prob):
            if total_prob[i] < EPSILON:
                total_prob[i] = 1
                corrections.append(i)
        self.prob = matrix / total_prob
        for i in corrections:
            self.prob[i, i] = 1
        for i in range(n):
            if abs(np.sum(self.prob[i]) - 1) > EPSILON:
                self.prob[i] /=  (self.prob[i]).sum()
                print((self.prob[i]).sum())
                try:
                    assert abs(np.sum(self.prob[i]) - 1) < EPSILON
                except:
                    print(abs(np.sum(self.prob[i]) - 1))
                    raise
        self.idx = {name:i for i, name in enumerate(gene_names)}

    def __getitem__(self, gene):
        i = self.idx[gene]
        return self.prob[i]



class WalkGenerator(object):
    ''' Create walks using a graph defined by a similarity matrix.

    Args:
        similarity: A dictionary representing the similarity between
            pairs of nodes, where the similarity between nodes u and v
            is represented by similarity((u, v)).
    '''
    def __init__(self, similarity_matrix, genes, walk_length, walk_per_node, fountains=None):
        self.walk_length = walk_length
        self.nodes = np.copy(genes)
        if fountains is None:
            self.fountains = np.copy(self.nodes)
        else:
            self.fountains = np.copy(fountains)
        self.starters = np.repeat(self.fountains, walk_per_node)
        np.random.shuffle(self.starters)
        self.LENGTH = len(self.starters)
        self.prob = Probability(similarity_matrix, self.nodes)



    def __len__(self):
        return self.LENGTH

    def __getitem__(self, i):
        ''' Generate a random walk starting from the i-th gene.

        Args:
            start: Starting point of the random walk.
            length: Length of the random walk.
        '''
        if i >= self.LENGTH:
            raise StopIteration
        current_node = self.starters[i]
        walk = [current_node]
        for _ in range(self.walk_length):
            next_node = np.random.choice(self.nodes, p=self.prob[current_node])
            walk.append(next_node)
            current_node = next_node
        return walk

    def __call__(self, i):
        return self[i]


#if __name__ == '__main__':
    #similarity_matrix = np.random.rand(15000, 15000)
    #genes = np.array(range(15000), dtype=np.uint16)

    #similarity_matrix = np.array([[0.0, 0.0, 0.6, 0.0],
    #                              [0.0, 0.0, 0.3, 0.0],
    #                              [0.6, 0.3, 0.0, 0.0],
    #                              [0.0, 0.0, 0.0, 0.0]])
    #
    #genes = ['1', '2', '3', '4']
    #start_time = time.time()
    #walks = WalkGenerator(similarity_matrix,genes, 50, 100)
    #hours, rem = divmod(time.time() - start_time, 3600)
    #minutes, seconds = divmod(rem, 60)
    #print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    #num_cpus = mp.cpu_count() - 1
    #pool = mp.Pool(num_cpus)
    #arguments = list(range(len(walks)))
    #chunk_size = len(walks) // num_cpus
    #results = pool.map(walks, arguments, chunksize=chunk_size)
    #with open('walks.csv', 'w') as fout:
    #    for w in results:
    #       fout.write('{}\n'.format(','.join([str(x) for x in w])))

    #for w in walks:
    #    print(w)

    # colour_map = "Greens_r"
    # model = gensim.models.Word2Vec(sentences=walks,
    #                        size=5,
    #                        window=2,
    #                        min_count=2,
    #                        workers=3,
    #                        iter=1)
    # wv1 = model.wv
    # vocab_size = len(genes)
    # dist1 = np.zeros((vocab_size, vocab_size))
    # for i, gene_i in enumerate(genes):
    #     for j, gene_j in enumerate(genes):
    #         dist1[i,j] = np.linalg.norm(wv1[gene_i] - wv1[gene_j])
    #
    # df = pd.DataFrame(dist1, columns=genes, index=genes)
    # ax = sns.heatmap(df, cmap=colour_map, square=True)
    # plt.show()
