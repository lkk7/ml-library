import numpy as np

from utils.math_functions import reg_grad_dict


class LinearRegression:
    """The linear regression model. Uses MSE cost function.
    Data standardization / normalization can make this model perform better.

    Parameters
    ----------
    learning_rate : float, default = 0.1
        The learning rate for gradient descent or SGD (if it is used).
    method : str, default = 'gradient'
        Method of fitting the model.
        'gradient' for gradient descent, 'sgd' for stochastic gradient descent, 'neq' for normal equation.
    reg : str, default = 'None'
        Regularization method.
        For L1 or L2, use 'l1' or 'l2' respectively.
        For elastic net method, use 'elastic'.
        'None' for no regularization.
    alpha : float, default = 0
        Alpha parameter controlling the 'strength' of regularization. Not for normal equation method.
    l1_ratio : float, default = 0
        Defines the ratio of L1 regularization. Only for elastic regularization option.
        The penalty added to cost is l1_ratio * L1 + 0.5 * (1 - l1_ratio) * L2.
    warm_start : bool, default = False
        Whether to reuse the previous solution as the initialization or erase it on fit() call.
    """
    def __init__(self, learning_rate=0.1, method='gradient', reg='None', alpha=0, l1_ratio=0, warm_start=False):
        self.learning_rate = learning_rate
        self.coef = None
        self.method = method
        self.reg = reg
        self.alpha = alpha
        self.l1_ratio = np.clip(l1_ratio, 0, 1)
        self.warm_start = warm_start
        self.trained = False

    def fit(self, x, y, n_iter=1000):
        """Use given training data to fit the model.

        Parameters
        ----------
        x : array-like
            Array of training examples.
        y : array-like
            Array of training y values.
        n_iter : int, default = 1000
            Number of iterations.
        """
        if len(x.shape) == 1:
            m, n = x.shape[0], 2
        else:
            m, n = x.shape[0], x.shape[1] + 1
        x_mat = np.column_stack((np.ones((m, 1)), x))   # Insert bias terms in the first column
        if not (self.warm_start and self.trained):
            self.coef = np.zeros(n)
        self.trained = True
        regularization_term = reg_grad_dict[self.reg]
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
