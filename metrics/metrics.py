import numpy as np


def distance(x_1, x_2, method='euclidean'):
    """Pairwise distance metric between two vectors.
    The vectors are assumed to be rows.

    Parameters
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


def mse(pred, y, squared=True):
    """Mean squared error or root mean squared error metric.

    Parameters
    ----------
    pred : array-like
        Array of predictions.
    y : array-like
        Array of true values.
    squared : bool, default = False
        Whether to return mean squared error (MSE) or root mean squared error (RMSE)
    """
    if squared:
        return np.square(np.subtract(pred, y)).mean()
    return np.sqrt(np.square(np.subtract(pred, y)).mean())


def log_loss(probs, y):
    """Logistic loss (cross-entropy loss) metric.

    Parameters
    ----------
    probs : array-like
        Array of probabilities.
    y : array-like
        Array of true labels.
    """
    return np.mean(-y * np.log(probs) - (1 - y) * np.log(1 - probs))


def accuracy_score(pred, y):
    """Classification accuracy score metric.

    Parameters
    ----------
    pred : array-like
        Array of predictions.
    y : array-like
        Array of true values.
    """
    return np.mean(pred == y)
