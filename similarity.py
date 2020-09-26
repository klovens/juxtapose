import random
import numpy as np


class Similarity(object):
    def __init__(self, correlation_file_path, anchors, alphas, sep=',',
                 prefix='pseudo', string_id=False):
        self.real_genes = set()
        with open(correlation_file_path) as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                a, b, _ = line.split(sep)
                a = a.strip()
                b = b.strip()
                if string_id is False:
                    a = int(a)
                    b = int(b)
                self.real_genes.add(a)
                self.real_genes.add(b)
        self.real_genes = list(sorted(self.real_genes))
        assert set(anchors).issubset(self.real_genes)
        self.pseudo_genes = []
        for anchor, alpha in zip(anchors, alphas):
            self.pseudo_genes.append('{}_{}'.format(prefix, anchor))
            for i in range(alpha):
                self.pseudo_genes.append('{}_{}_{:0>3d}'.format(prefix, anchor, i))
        genes = self.real_genes + self.pseudo_genes
        n = len(genes)
        self.matrix = np.zeros((n, n), dtype=np.float32)
        self.idx = {gene: i for i, gene in enumerate(genes)}
        # Assign values to the correlation matrix
        with open(correlation_file_path) as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                a, b, cor = line.split(sep)
                if string_id is False:
                    a = int(a)
                    b = int(b)
                i = self.idx[a]
                j = self.idx[b]
                self.matrix[i,j] = np.float32(cor)
                self.matrix[j,i] = np.float32(cor)

    def average_correlation(self):
        n = len(self.real_genes)
        values = self.matrix[0:n, 0:n][np.nonzero(self.matrix[0:n, 0:n])]
        return np.mean(values)


    def __getitem__(self, item):
        a, b = item
        i = self.idx[a]
        j = self.idx[b]
        return self.matrix[i, j]

    def transform(self, transform=None):
        if transform is None:
            transform = lambda x: 0.5 * x + 0.5
        n = len(self.real_genes) + len(self.pseudo_genes)
        for i in range(n):
            for j in range(n):
                self.matrix[i, j] = transform(self.matrix[i, j])

    def apply_threshold(self, lower_cor, upper_cor, value):
        n = len(self.real_genes) + len(self.pseudo_genes)
        for i in range(n):
            for j in range(n):
                if self.matrix[i, j] > lower_cor and self.matrix[i, j] < upper_cor:
                    self.matrix[i, j] = value

    def to_csv(self, file_name):
        n = len(self.real_genes) + len(self.pseudo_genes)
        genes = self.real_genes + self.pseudo_genes
        with open(file_name, 'w') as f:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        break
                    else:
                        f.write(','.join([genes[i], genes[j], str(self.matrix[i, j])]))
                        f.write("\n")

    def augment(self, dangles):
        genes = self.real_genes + self.pseudo_genes
        for (a, b), w in dangles.items():
            assert a in genes, "gene is missing from similarity matrix."
            assert b in genes, "gene is missing from similarity matrix."
            i = self.idx[a]
            j = self.idx[b]
            self.matrix[i, j] = w
            self.matrix[j, i] = w
