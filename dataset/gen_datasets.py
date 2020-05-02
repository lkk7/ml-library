"""Module for generating the datasets that can be used to test the algorithms."""
import numpy as np


def gen_bin_class_dataset(rows=20000):
    """Generate the auxiliary dataset for binary classification as 'bin_class.csv'.
       It is a dataset of shape (rows=20000, 4) – a simple one for testing.
       The columns are: x_1, x_2, x_3, y (1 or 0).
    """
    dataset = np.column_stack((np.random.normal(0, 0.05, (rows, 3)), np.zeros(rows)))
    sample = np.random.choice(dataset.shape[0], int(rows / 2), replace=False)
    dataset[sample, 0] *= 2
    dataset[sample, :-1] += 0.2
    dataset[sample, -1] = 1
    np.savetxt('bin_class.csv', dataset, delimiter=',', fmt='%.3f', header='x_1,x_2,x_3,y', comments='')


def gen_regr_dataset(rows=20000):
    """Generate the auxiliary dataset for regression as 'regr.csv'.
       It is a dataset of shape (rows=20000, 3) – a simple one for testing.
       The columns are: x_1, x_2, y.
    """
    x_1 = np.exp(np.linspace(0, 1, rows))
    x_2 = np.linspace(0, 1, rows)
    y = 10 * np.sin(2 * np.linspace(0, 1, rows))
    dataset = np.column_stack((x_1, x_2, y)) + np.random.normal(0, 0.05, (rows, 3))
    np.savetxt('regr.csv', dataset, delimiter=',', fmt='%.3f', header='x_1,x_2,y', comments='')


if __name__ == '__main__':
    gen_bin_class_dataset()
    gen_regr_dataset()
