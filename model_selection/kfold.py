import numpy as np


class KFold:
    """K-fold cross-validating class that splits the data accordingly.

    Parameters
    ----------
    k : int, default = 5
        Number of folds in cross-validation.
    shuffle : bool, default = True
        Whether the data should be shuffled when splitting.
    """
    def __init__(self, k=5, shuffle=True):
        self.k = k
        self.shuffle = shuffle

    def split(self, x):
        """Return the indices of split data.

        Parameters
        ----------
        x : array-like
            Array of data.
        """
        split_data = []
        data_size = len(x)
        all_indices = np.arange(data_size)
        if self.shuffle:
            np.random.shuffle(all_indices)
        fold_size = data_size // self.k
        for i in range(self.k):
            train_indices = all_indices[(i * fold_size):((i + 1) * fold_size)]
            validation_indices = np.delete(all_indices, train_indices)
            split_data.append((train_indices, validation_indices))
        return split_data
