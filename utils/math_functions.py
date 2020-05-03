"""Module for math functions and utilities."""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Helper dict containing gradient for each regularization in form of lambda functions.
# a – alpha, w – weights, l1 – l1_ratio
reg_grad_dict = {'l2': (lambda a, w, l1: w * a),
                 'l1': (lambda a, w, l1: np.sign(w) * a),
                 'elastic': lambda a, w, l1: np.sign(w) * l1 * a + w * (1 - l1) * a,
                 None: (lambda a, w, l1: w * 0)}
