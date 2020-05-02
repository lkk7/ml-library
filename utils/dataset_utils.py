"""Module for dataset utilities, such as loading them."""
import numpy as np


def load_bin_class_data():
    """Load the binary classification dataset.
    Returns X and Y.
    """
    data = np.loadtxt('../dataset/bin_class.csv', delimiter=',', skiprows=1)
    return data[:, :-1], data[:, -1]


def load_regression_data():
    """Load the regression dataset.
    Returns X and Y.
    """
    data = np.loadtxt('../dataset/regr.csv', delimiter=',', skiprows=1)
    return data[:, :-1], data[:, -1]
