from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes
from typing import Union
import numpy as np
import pandas as pd


def cluster_distance_matrix(X, labels, metric='cosine', label_subset: Union[list, set] = None):
    """
    Compute a C x C matrix where entry (i, j) is the average distance
    between points in cluster i and cluster j.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Data matrix where each row is a point.
    labels : array-like of shape (n,)
        Cluster labels for each row in X. Assumed to be integers or hashable values.
    metric : str or callable, optional (default='euclidean')
        Distance metric accepted by scipy.spatial.distance.cdist.
    label_subset : list or set
        Subset of labels for which to construct cluster distance matrix

    Returns
    -------
    D : ndarray of shape (C, C)
        Cluster-to-cluster average distance matrix.
    cluster_ids : ndarray of shape (C,)
        The unique cluster identifiers corresponding to rows/columns of D.
    """

    X = np.asarray(X)

    if label_subset is None:
        label_subset = np.unique(labels)
    else:
        label_subset = list(label_subset)

    mask = np.isin(labels, label_subset)
    labels = labels[mask]
    X = X[mask]

    C = len(label_subset)

    D = np.zeros((C, C))

    # Precompute index masks for clusters
    cluster_points = [X[labels == cid] for cid in label_subset]

    for i in range(C):
        for j in range(i, C):
            dists = cdist(cluster_points[i], cluster_points[j], metric=metric)
            mean_dist = dists.mean()
            D[i, j] = D[j, i] = mean_dist  # symmetric

    return pd.DataFrame(D, columns=label_subset, index=label_subset)


def align_latent_spaces(matrices):

    assert np.all([mat.shape == matrices[0].shape for mat in matrices[1:]])
    matrices = [mat - mat.mean(0) for mat in matrices]
    aligned_mats = [matrices[0]]
    for mat in matrices[1:]:
        rotation, _ = orthogonal_procrustes(mat, matrices[0])
        rot_mat = mat @ rotation
        aligned_mats.append(rot_mat)
    return aligned_mats


def upper_tri_list_to_matrix(vals):
    vals = np.asarray(vals)
    L = len(vals)

    # solve n*(n+1)/2 = L  (upper triangle including diagonal)
    n = int((np.sqrt(8*L + 1) - 1)//2)

    M = np.zeros((n, n))
    idx = np.triu_indices(n)
    M[idx] = vals
    # mirror strict upper triangle (i<j)
    i, j = np.triu_indices(n, k=1)
    M[j, i] = M[i, j]
    return M
