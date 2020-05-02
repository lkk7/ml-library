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
        Method of regularization.
        'l1' or 'l2' for L1 or L2 regularization,
        elastic' for using elastic net method, None for no regularization.
    """
    def __init__(self, learning_rate=0.2, method='gradient', reg=None):
        self.learning_rate = learning_rate
        self.coef = None
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
        m, n = x.shape[0], x.shape[1] + 1
        x_mat = np.column_stack((np.ones((m, 1)), x))   # Insert bias terms in the first column
        self.coef = np.zeros(n)
        if self.method == 'gradient':
            for i in range(n_iter):
                pass
        elif self.method == 'sgd':
            for i in range(n_iter):
                pass
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


from utils.dataset_utils import load_regression_data
xs, ys = load_regression_data()
model = LinearRegression(method='neq', reg=None)
model.fit(xs, ys)
print(model.predict(xs[10000:10010]))
print(ys[10000:10010])
