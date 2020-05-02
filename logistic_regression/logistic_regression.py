"""Logistic regression module.
It is a binary classification algorithm that outputs the probability that the given input belongs to a class.
The classes should be linearly separable (if they aren't, adding polynomial or interaction terms can be considered).
"""
import numpy as np

from utils.math_functions import sigmoid


class LogisticRegression:
    """The logistic regression binary classifier.

    Parameters
    ----------
    learning_rate : float, default = 0.2
        The learning rate for gradient descent or SGD.
    method : str, default = 'gradient'
        Method of fitting the model.
        'gradient' for gradient descent, 'sgd' for stochastic gradient descent.
    """
    def __init__(self, learning_rate=0.2, method='gradient'):
        self.learning_rate = learning_rate
        self.coef = None
        self.intercept = None
        self.method = method

    def fit(self, x, y, n_iter=1000):
        """Use given training data to fit the model.

        Parameters
        ----------
        x : array-like
            Array of training examples.
        y : array-like
            Array of binary training labels (1 or 0 only).
        n_iter : int, default = 1000
            Number of iterations.
        """
        m, n = x.shape[0], x.shape[1]
        self.coef = np.zeros(n)
        self.intercept = 0
        if self.method == 'gradient':
            for i in range(n_iter):
                dz = sigmoid(np.dot(x, self.coef) + self.intercept) - y
                dw = np.dot(dz, x) / m
                db = np.sum(dz) / m
                self.coef -= self.learning_rate * dw
                self.intercept -= self.learning_rate * db
        elif self.method == 'sgd':
            for i in range(n_iter):
                index = np.random.choice(x.shape[0], 1)
                x_i, y_i = x[index], y[index]
                db = sigmoid(np.dot(x_i, self.coef) + self.intercept) - y_i
                dw = db * x_i
                self.coef -= self.learning_rate * dw.reshape((3,))
                self.intercept -= self.learning_rate * db
        else:
            raise ValueError("Wrong 'method' argument. Use 'gradient', 'sgd' or 'ls'.")

    def predict_proba(self, x):
        """Predict probabilities of class '1' for given input.

        Parameters
        ----------
        x : array-like
            Input array.
        """
        return sigmoid(np.dot(x, self.coef) + self.intercept)

    def predict(self, x):
        """Predict the class for given input.

        Parameters
        ----------
        x : array-like
            Input array.
        """
        return np.round(self.predict_proba(x))
