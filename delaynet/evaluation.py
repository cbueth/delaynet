"""Evaluation module to compare original and reconstructed networks."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import auc as skauc


def roc_auc_rank_c(
    orig_net: np.ndarray, weight_mat: np.ndarray, rec_net: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """
    Get Receiver Operating Characteristic (roc) curve, Area Under Curve (auc)
    and Spearman rank correlation coefficient (rank_c) for a given reconstructed
    network.

    :param orig_net: Original network
    :type orig_net: np.ndarray( n_nodes, n_nodes )
    :param weight_mat: Weights of the original network
    :type weight_mat: np.ndarray( n_nodes, n_nodes )
    :param rec_net: Reconstructed network
    :type rec_net: np.ndarray( n_nodes, n_nodes )
    :return: ROC curve, Area Under Curve, Spearman rank correlation coefficient
    :rtype: tuple[ np.ndarray( n_nodes, 2 ), float, float ]

    :raises TypeError: If any of the input parameters is not a numpy array.
    :raises ValueError: If any of the input parameters is not a square matrix.
    :raises ValueError: If the input parameters do not have the same number of nodes.
    """
    # Check input
    for mat in [orig_net, weight_mat, rec_net]:
        if not isinstance(mat, np.ndarray):
            raise TypeError(f"{mat} must be a numpy array, but is {type(mat)}.")
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"{mat} must be a square matrix, but has shape {mat.shape}."
            )
    if orig_net.shape != weight_mat.shape or orig_net.shape != rec_net.shape:
        raise ValueError(
            f"orig_net, weight_mat and rec_net must have the same shape, but are "
            f"{orig_net.shape}, {weight_mat.shape} and {rec_net.shape}."
        )

    m_v = np.max(rec_net) + 1
    for k in range(np.size(orig_net, 0)):
        rec_net[k, k] = m_v

    all_pv = []
    all_pv.append(np.min(rec_net))
    all_pv.append(np.max(rec_net))

    for a1 in range(np.size(orig_net, 0)):
        for a2 in range(np.size(orig_net, 0)):
            if a1 == a2:
                continue

            all_pv.append(rec_net[a1, a2])

    all_pv = np.sort(all_pv)
    all_pv2 = []
    for k in range(np.size(all_pv) - 1):
        all_pv2.append(all_pv[k])
        all_pv2.append((all_pv[k] + all_pv[k + 1]) / 2.0)
    all_pv2.append(all_pv[-1])
    all_pv = np.sort(all_pv2)

    roc = [(0.0, 0.0)]

    for thr in range(np.size(all_pv)):
        fpr = np.sum(orig_net[rec_net <= all_pv[thr]] == 0) / np.sum(orig_net == 0)
        tpr = np.sum(orig_net[rec_net <= all_pv[thr]] > 0) / np.sum(orig_net > 0)
        roc.append((fpr, tpr))
    roc = np.array(roc)

    return (
        roc,
        skauc(roc[:, 0], roc[:, 1]),
        np.abs(spearmanr(np.ravel(weight_mat), np.ravel(rec_net)))[0],
    )
