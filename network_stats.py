import numpy as np


def alignment_permutation_test(vocab1_length, vocab2_length, distance, actual_score, num_iteration=1000):
    n = min(vocab1_length, vocab2_length)
    indices = list(range(n))
    scores = []
    for i in range(num_iteration):
        v1 = np.random.choice(indices, size=n)
        v2 = np.random.choice(indices, size=n)
        s = 0
        for i, j in zip(v1, v2):
            s += distance[i, j]
        scores.append(s/n)
    scores = np.array(scores)
    print(scores)
    p = sum(scores >= actual_score) / num_iteration
    return p


