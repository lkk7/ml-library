import numpy as np
from utils.math_functions import gaussian

class GaussianNB:
    """Naive Bayes classifier that assumes"""
    def __init__(self):
        self.x = None
        self.y = None
        self.classes = None
        self.coef = []

    def fit(self, x, y):
        """Use given training data to fit the model.

        Parameters
        ----------
        x : array-like
            Array of training examples.
        y : array-like
            Array of training labels.
        """
        self.x = x
        self.y = y
        self.classes = np.unique(y)
        for i, cl in enumerate(self.classes):
            x_class = x[y == cl]
            self.coef.append([(np.mean(feature), np.var(feature)) for feature in x_class.T])

    def predict(self, x):
        """Predict the class for given input.

        Parameters
        ----------
        x : array-like
            Input array.
        """
        probs = np.log(np.tile(np.array([np.mean(self.y == cl) for i, cl in enumerate(self.classes)]), (x.shape[0], 1)))
        print(self.coef)
        for i in range(x.shape[1]):
            for j in range(len(self.classes)):
                probs[:, j] += np.log(gaussian(x[:, i], self.coef[j][i][0], self.coef[j][i][1]) + 1e-300)
        return np.argmax(probs, axis=1)
