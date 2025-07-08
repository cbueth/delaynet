"""Network pruning methods for delaynet.

This module provides methods for pruning networks by removing weak or
statistically insignificant connections from reconstructed networks.
"""

from typing import Optional, Any
import numpy as np
from statsmodels.stats.multitest import multipletests


def statistical_pruning(
    p_values: np.ndarray,
    alpha: float = 0.05,
    correction: Optional[str] = None,
    **multipletests_kwargs: Any,
) -> np.ndarray:
    """
    Prune network connections by statistical significance, with an optional
    multiple‐comparison correction.

    :param p_values:           Matrix of p-values, shape (n_nodes, n_nodes).
    :type p_values:            numpy.ndarray
    :param alpha:              Per-test significance level (default 0.05).
    :type alpha:               float
    :param correction:         Name of the correction method to use. Must be one
                              of the methods supported by
                              {func}`statsmodels.stats.multitest.multipletests`
                              (e.g. 'bonferroni', 'sidak', 'holm',
                              'fdr_bh', 'fdr_by', 'fdr_tsbh', etc.).
                              If None, no correction is applied.
    :type correction:          str or None
    :param multipletests_kwargs:
                              Additional keyword arguments to pass to
                              multipletests(), such as
                              maxiter, is_sorted, returnsorted.
    :type multipletests_kwargs: dict

    :return:                   Boolean mask indicating which connections are
                               significant after (optional) correction.
    :rtype:                    numpy.ndarray

    Example:
    --------
    >>> mask = statistical_pruning(p_values,
    ...                            alpha=0.05,
    ...                            correction='fdr_bh',
    ...                            maxiter=2)
    """
    flat = p_values.ravel()

    if correction is None:
        mask_flat = flat < alpha
    else:
        # Clamp p-values to avoid numerical issues in correction methods
        # that use log1p(-pvals), such as 'sidak' and 'holm-sidak'
        # When p-values are exactly 1.0, log1p(-1.0) = log(0) causes divide by zero
        flat_clamped = np.clip(flat, 0.0, 1.0 - np.finfo(float).eps)

        # multipletests will handle all supported methods internally
        reject, *_ = multipletests(
            flat_clamped, alpha=alpha, method=correction, **multipletests_kwargs
        )
        mask_flat = reject

    return mask_flat.reshape(p_values.shape)
