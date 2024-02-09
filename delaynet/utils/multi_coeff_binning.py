"""Multiple Coefficient Binning (MCB) module.

Author: Johann Faouzi <johann.faouzi@gmail.com>
License: BSD-3-Clause

Modified on Thu Sep 14 15:42:20 2023
Author: Massimiliano Zanin

"""

import numpy as np
from numba import njit, prange
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.utils.multiclass import check_classification_targets


@njit()
def _uniform_bins(timestamp_min, timestamp_max, n_timestamps, n_bins):
    bin_edges = np.empty((n_timestamps, n_bins - 1))
    for i in prange(n_timestamps):  # pylint: disable=not-an-iterable
        bin_edges[i] = np.linspace(timestamp_min[i], timestamp_max[i], n_bins + 1)[1:-1]
    return bin_edges


@njit()
def _digitize_1d(x, bins, n_samples, n_timestamps):
    x_digit = np.empty((n_samples, n_timestamps))
    for i in prange(n_timestamps):  # pylint: disable=not-an-iterable
        x_digit[:, i] = np.digitize(x[:, i], bins, right=True)
    return x_digit


@njit()
def _digitize_2d(x, bins, n_samples, n_timestamps):
    x_digit = np.empty((n_samples, n_timestamps))
    for i in prange(n_timestamps):  # pylint: disable=not-an-iterable
        x_digit[:, i] = np.digitize(x[:, i], bins[i], right=True)
    return x_digit


def _digitize(x, bins):
    n_samples, n_timestamps = x.shape
    if bins.ndim == 1:
        x_binned = _digitize_1d(x, bins, n_samples, n_timestamps)
    else:
        x_binned = _digitize_2d(x, bins, n_samples, n_timestamps)
    return x_binned.astype("int64")


class MultipleCoefficientBinning:
    """Bin continuous data into intervals column-wise.

    Uses the approach of the Symbolic Fourier Approximation (SFA) to bin
    continuous data into intervals. The bin edges are computed using different
    strategies: 'uniform', 'quantile', 'normal' or 'entropy'. The alphabet
    can be either the first `n_bins` letters of the Latin alphabet, integers
    or a custom alphabet. [1]_

    Parameters
    ----------
    n_bins : int (default = 4)
        The number of bins to produce. It must be between 2 and
        ``min(n_samples, 26)``.

    strategy : str (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution
        - 'entropy': Bin edges are computed using information gain

    alphabet : None, 'ordinal' or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used if `n_bins` is lower than 27, otherwise the alphabet
        will be defined to [chr(i) for i in range(n_bins)]. If 'ordinal',
        integers are used.

    Attributes
    ----------
    bin_edges : array, shape = (n_bins - 1,) or (n_timestamps, n_bins - 1)
        Bin edges with shape = (n_bins - 1,) if ``strategy='normal'`` or
        (n_timestamps, n_bins - 1) otherwise.

    References
    ----------
    .. [1] P. Schäfer, and M. Högqvist, "SFA: A Symbolic Fourier Approximation
           and Index for Similarity Search in High Dimensional Datasets",
           International Conference on Extending Database Technology,
           15, 516-527 (2012).

    Examples
    --------
    >>> from pyts.approximation import MultipleCoefficientBinning
    >>> X = [[0, 4],
    ...      [2, 7],
    ...      [1, 6],
    ...      [3, 5]]
    >>> transformer = MultipleCoefficientBinning(n_bins=2)
    >>> print(transformer.fit_transform(X))
    [['a' 'a']
     ['b' 'b']
     ['a' 'b']
     ['b' 'a']]

    """

    def __init__(self, n_bins=4, strategy="quantile", alphabet=None):
        self.n_bins = n_bins
        self.strategy = strategy
        self.alphabet = alphabet
        self._n_timestamps_fit = None
        self._alphabet = None
        self.bin_edges = None

    def fit(self, x, y=None):
        """Compute the bin edges for each feature.

        :param x: Data to transform.
        :type x: array-like, shape = (n_samples, n_timestamps)
        :param y: Class labels for each sample. Only used if ``strategy='entropy'``.
        :type y: None or array-like, shape = (n_samples,)
        :return: self
        :rtype: object
        """
        if self.strategy == "entropy":
            if y is None:
                raise ValueError("y cannot be None if strategy='entropy'.")
            x, y = check_X_y(x, y, dtype="float64")
            check_classification_targets(y)
        else:
            x = check_array(x, dtype="float64")
        n_samples, n_timestamps = x.shape
        self._n_timestamps_fit = n_timestamps
        self._alphabet = self._check_params(n_samples)
        self._check_constant(x)
        self.bin_edges = self._compute_bins(
            x, y, n_timestamps, self.n_bins, self.strategy
        )
        return self

    def transform(self, x):
        """Bin the data.

        :param x: Data to transform.
        :type x: array-like, shape = (n_samples, n_timestamps)
        :return: Binned data.
        :rtype: array, shape = (n_samples, n_timestamps)
        """

        check_is_fitted(self, "bin_edges")
        x = check_array(x, dtype="float64")
        self._check_consistent_lengths(x)
        indices = _digitize(x, self.bin_edges)
        if isinstance(self._alphabet, str):
            return indices
        return self._alphabet[indices]

    def _check_params(self, n_samples):
        if not isinstance(self.n_bins, (int, np.integer)):
            raise TypeError("'n_bins' must be an integer.")
        if not 2 <= self.n_bins <= min(n_samples, 26):
            raise ValueError(
                "`n_bins` must be greater than or equal to 2 and lower than "
                f"or equal to min(n_samples, 26) (got {self.n_bins})."
            )
        if self.strategy not in ["uniform", "quantile", "normal", "entropy"]:
            raise ValueError(
                "`strategy` must be either 'uniform', 'quantile',"
                f" 'normal' or 'entropy' (got {self.strategy})."
            )
        if not (
            (self.alphabet is None)
            or (self.alphabet == "ordinal")
            or (isinstance(self.alphabet, (list, tuple, np.ndarray)))
        ):
            raise TypeError(
                "`alphabet` must be None, 'ordinal' or array-like "
                f"with shape (n_bins,) (got {self.alphabet})."
            )
        if self.alphabet is None:
            alphabet = np.array([chr(i) for i in range(97, 97 + self.n_bins)])
        elif self.alphabet == "ordinal":
            alphabet = "ordinal"
        else:
            alphabet = check_array(self.alphabet, ensure_2d=False, dtype=None)
            if alphabet.shape != (self.n_bins,):
                raise ValueError(
                    "If 'alphabet' is array-like, its shape "
                    "must be equal to (n_bins,)."
                )
        return alphabet

    def _check_constant(self, x):
        if np.any(np.max(x, axis=0) - np.min(x, axis=0) == 0):
            raise ValueError("At least one timestamp is constant.")

    def _check_consistent_lengths(self, x):
        if self._n_timestamps_fit != x.shape[1]:
            raise ValueError(
                "The number of timestamps in X must be the same as "
                "the number of timestamps when `fit` was called "
                f"({self._n_timestamps_fit} != {x.shape[1]})."
            )

    def _compute_bins(self, x, y, n_timestamps, n_bins, strategy):
        if strategy == "normal":
            bins_edges = norm.ppf(np.linspace(0, 1, self.n_bins + 1)[1:-1])
        elif strategy == "uniform":
            timestamp_min, timestamp_max = np.min(x, axis=0), np.max(x, axis=0)
            bins_edges = _uniform_bins(
                timestamp_min, timestamp_max, n_timestamps, n_bins
            )
        elif strategy == "quantile":
            bins_edges = np.percentile(
                x, np.linspace(0, 100, self.n_bins + 1)[1:-1], axis=0
            ).T
            if np.any(np.diff(bins_edges, axis=0) == 0):
                raise ValueError(
                    "At least two consecutive quantiles are equal. "
                    "Consider trying with a smaller number of bins or "
                    "removing timestamps with low variation."
                )
        else:
            bins_edges = self._entropy_bins(x, y, n_timestamps, n_bins)
        return bins_edges

    def _entropy_bins(self, x, y, n_timestamps, n_bins):
        bins = np.empty((n_timestamps, n_bins - 1))
        clf = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=n_bins)
        for i in range(n_timestamps):
            clf.fit(x[:, i][:, None], y)
            threshold = clf.tree_.threshold[clf.tree_.children_left != -1]
            if threshold.size < (n_bins - 1):
                raise ValueError(
                    f"The number of bins is too high for timestamp {i}. "
                    "Consider trying with a smaller number of bins or "
                    "removing this timestamp."
                )
            bins[i] = threshold
        return np.sort(bins, axis=1)
