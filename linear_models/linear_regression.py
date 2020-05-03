"""Linear regression module.
It is a simple regression model that assumes linearity of the true relationship between x and y.
If this relationship is not linear, adding polynomial or interaction terms can be considered.
"""
import numpy as np


class LinearRegression:
    """The linear regression model.

    Parameters
    ----------
    learning_rate : float, default = 0.2
        The learning rate for gradient descent or SGD (if it is used).
    method : str, default = 'gradient'
        Method of fitting the model.
        'gradient' for gradient descent, 'sgd' for stochastic gradient descent, 'neq' for normal equation.
    reg : str, default = None
        Regularization method.
        For L1 or L2, use 'l1' or 'l2' respectively.
        For elastic net method, use 'elastic'.
        None for no regularization.
    alpha : float, default = 0
        Alpha parameter controlling the 'strength' of regularization. Not for normal equation method.
    l1_ratio : float, default = 0
        Defines the ratio of L1 regularization. Only for elastic regularization option.
        The penalty added to cost is l1_ratio * L1 + 0.5 * (1 - l1_ratio) * L2.
    """
    def __init__(self, learning_rate=0.2, method='gradient', reg=None, alpha=0, l1_ratio=0):
        self.learning_rate = learning_rate
        self.coef = None
        self.method = method
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
        m, n = x.shape[0], x.shape[1] + 1
        x_mat = np.column_stack((np.ones((m, 1)), x))   # Insert bias terms in the first column
        self.coef = np.zeros(n)
        regularization_term = {'l2': (lambda a, w, l1: w * a),
                               'l1': (lambda a, w, l1: np.sign(w) * a),
                               'elastic': lambda a, w, l1: np.sign(w) * l1 * a + w * (1 - l1) * a,
                               None: (lambda a, w, l1: w * 0)}[self.reg]
        if self.method == 'gradient':
            for i in range(n_iter):
                prediction = np.dot(x_mat, self.coef)
                reg = regularization_term(self.alpha, self.coef, self.l1_ratio)
                reg[0] = 0
                dw = (np.dot((prediction - y), x_mat) + reg) / m
                self.coef -= self.learning_rate * dw
        elif self.method == 'sgd':
            for i in range(n_iter):
                index = np.random.choice(x_mat.shape[0], 1)
                x_i, y_i = x_mat[index], y[index]
                prediction = np.dot(x_i, self.coef)
                reg = regularization_term(self.alpha, self.coef, self.l1_ratio)
                reg[0] = 0
                dw = np.dot((prediction - y_i), x_i) + reg / m
                self.coef -= self.learning_rate * dw
        elif self.method == 'neq':
            self.coef = np.linalg.pinv(x_mat.T.dot(x_mat)).dot(x_mat.T).dot(y)
        else:
            raise ValueError("Wrong 'method' argument. Use 'gradient', 'sgd' or 'ls'.")

    def predict(self, x):
        """Predict the class for given input.

        Parameters
        ----------
        x : array-like
            Input array.
        """
        return np.dot(np.column_stack((np.ones((x.shape[0], 1)), x)), self.coef)
