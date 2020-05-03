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
    reg : str, default = None
        Regularization method.
        For L1 or L2, use 'l1' or 'l2' respectively.
        For elastic net method, use 'elastic'.
        None for no regularization.
    alpha : float, default = 0
        Alpha parameter controlling the 'strength' of regularization.
    l1_ratio : float, default = 0
        Defines the ratio of L1 regularization. Only for elastic regularization option.
        The penalty added to cost is l1_ratio * L1 + 0.5 * (1 - l1_ratio) * L2.
    """
    def __init__(self, learning_rate=0.2, method='gradient', reg=None, alpha=0, l1_ratio=0):
        self.learning_rate = learning_rate
        self.coef = None
        self.intercept = None
        self.method = method
        self.reg = reg
        self.alpha = alpha
        self.l1_ratio = np.clip(l1_ratio, 0, 1)

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
        regularization_term = {'l2': (lambda a, w, l1: w * a),
                               'l1': (lambda a, w, l1: np.sign(w) * a),
                               'elastic': lambda a, w, l1: np.sign(w) * l1 * a + w * (1 - l1) * a,
                               None: (lambda a, w, l1: 0)}[self.reg]
        if self.method == 'gradient':
            for i in range(n_iter):
                dz = sigmoid(np.dot(x, self.coef) + self.intercept) - y
                dw = (np.dot(dz, x) + regularization_term(self.alpha, self.coef, self.l1_ratio)) / m
                db = np.sum(dz) / m
                self.coef -= self.learning_rate * dw
                self.intercept -= self.learning_rate * db
        elif self.method == 'sgd':
            for i in range(n_iter):
                index = np.random.choice(x.shape[0], 1)
                x_i, y_i = x[index], y[index]
                db = sigmoid(np.dot(x_i, self.coef) + self.intercept) - y_i
                dw = db * x_i + regularization_term(self.alpha, self.coef, self.l1_ratio) / m
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
