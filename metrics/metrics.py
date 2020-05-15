import numpy as np


def distance(x_1, x_2, method='euclidean'):
    """Pairwise distance metric between two vectors.
    The vectors are assumed to be rows.

    Parametersz
    ----------
    x_1 : array-like
        One of two vectors to compute distance between.
    x_2 : array-like
        One of two vectors to compute distance between.
    method : str, default = euclidean
        Type of the distance metric.
        'euclidean' for Euclidean (L2) distance, 'manhattan' for Manhattan (L1) distance.
    """
    if method == 'euclidean':
        return np.sqrt(np.sum((x_1 - x_2)**2, axis=1))
    elif method == 'manhattan':
        return np.sum(np.abs((x_1 - x_2)), axis=1)
    else:
        raise ValueError("Wrong 'method' argument. Use 'euclidean' or 'manhattan'.")
