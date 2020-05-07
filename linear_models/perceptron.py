import numpy as np
from linear_models.logistic_regression import LogisticRegression


class Perceptron(LogisticRegression):
    """A simple (binary classification) perceptron. Uses binary cross-entropy loss for updating weights.
    >>NOTE: it inherits most of the code from logistic regression for simplicity.<<

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
        super().__init__(learning_rate, method, reg, alpha, l1_ratio)

    def predict(self, x):
        """Predict the class for given input.

        Parameters
        ----------
        x : array-like
            Input array.
        """
        return np.heaviside(np.dot(x, self.coef) + self.intercept, 1)
