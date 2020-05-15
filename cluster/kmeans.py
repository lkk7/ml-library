import numpy as np
from metrics.metrics import distance


class KMeans:
    """The k-means clustering algorithm.

    Parameters
    ----------
    k : int, default = 3
        The number of clusters
    """
    def __init__(self, k=3):
        self.k = k

    def fit_predict(self, x, n_iter=300):
        """Assign the inputs to clusters.

        Parameters
        ----------
        x : array-like
            Input array.
        n_iter : int, default = 300
            Number of iterations (finishes earlier if convergence is reached)
        """
        index = np.random.permutation(x.shape[0])[:self.k]
        centroids = x[index]
        labels = None
        for i in range(n_iter):
            labels = np.array([distance(x, c) for c in centroids]).argmin(axis=0)
            new_centroids = np.array([x[labels == i].mean(axis=0) for i in range(self.k)])
            if np.all(new_centroids == centroids):
                break
            centroids = new_centroids
        return labels
