import numpy as np
from metrics.metrics import distance


class KNNClassifier:
    """The k-nearest neighbors classifier.

    Parameters
    ----------
    k : int, default = 3
        The number of nearest neighbors considered.
    """
    def __init__(self, k=3):
        self.k = k
        self.xs = None
        self.ys = None

    def fit(self, x, y):
        """Use given training data to fit the model.

        Parameters
        ----------
        x : array-like
            Array of training examples.
        y : array-like
            Array of training labels (should be integers)
        """
        self.xs = np.copy(x)
        self.ys = np.copy(y).astype(int)

    def predict(self, x):
        """Predict the class for given input.

        Parameters
        ----------
        x : array-like
            Input array.
        """
        pred = np.empty(x.shape[0])
        for index, row in enumerate(x):
            k_sorted_xs = np.argsort(distance(self.xs, row))[:self.k]
            k_classes = self.ys[k_sorted_xs]
            pred[index] = np.bincount(k_classes).argmax()
        return pred
