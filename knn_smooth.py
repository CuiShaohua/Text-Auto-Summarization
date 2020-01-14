import numpy as np

window = 3

def knn_smooth(L):

    L = [np.mean(L[i:i+window]) for i in range(len(L)) if i <= window] + [np.mean(L[i-window:i+window]) for i in range(len(L)) if i> window]

    return np.array(L)
