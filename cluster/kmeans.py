import numpy as np
from metrics.metrics import distance


class KMeans:
    """The k-means clustering algorithm.

    Parameters
    ----------
    k : int, default = 3
        The number of clusters
    warm_start : bool, default = False
        Whether to reuse the previous result as the initialization or erase it on fit_predict() call.
    """
    def __init__(self, k=3, warm_start=False):
        self.k = k
        self.warm_start = warm_start
        self.trained = False
        self.centroids = None
        self.labels = None

    def fit_predict(self, x, n_iter=300):
        """Assign the inputs to clusters.

        Parameters
        ----------
        x : array-like
            Input array.
        n_iter : int, default = 300
            Number of iterations (finishes earlier if convergence is reached)
        """

        if not (self.warm_start and self.trained):
            index = np.random.permutation(x.shape[0])[:self.k]
            self.centroids = x[index]
            self.trained = True
        for i in range(n_iter):
            self.labels = np.array([distance(x, c) for c in self.centroids]).argmin(axis=0)
            new_centroids = np.array([x[self.labels == i].mean(axis=0) for i in range(self.k)])
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids
        return self.labels
