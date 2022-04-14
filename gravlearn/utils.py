"""Supplementary functions for residual2vec.py."""
import numpy as np
from scipy import sparse
import torch

#
# Homogenize the data format
#
def to_adjacency_matrix(net):
    """Convert to the adjacency matrix in form of sparse.csr_matrix.

    :param net: adjacency matrix
    :type net: np.ndarray or csr_matrix
    :return: adjacency matrix
    :rtype: sparse.csr_matrix
    """
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net
        return sparse.csr_matrix(net)
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net)
    else:
        raise ValueError(
            "Unexpected data type {} for the adjacency matrix".format(type(net))
        )


def angle(x, y):
    x = np.einsum("ij,i->ij", x, 1 / np.linalg.norm(x, axis=1))
    y = np.einsum("ij,i->ij", y, 1 / np.linalg.norm(y, axis=1))
    cos = x @ y.T
    return np.arccos(cos)


def sample_from_exponential(scale, size, max_x=None):
    if max_x is None:
        return np.random.exponential(scale=scale, size=size)

    # Refection sampling
    n_sampled = 0
    angle = np.zeros(size)
    while n_sampled < size:
        sampled_angle = np.random.exponential(scale=scale, size=size - n_sampled)
        sampled_angle = sampled_angle[sampled_angle < np.pi]
        angle[n_sampled : n_sampled + len(sampled_angle)] = sampled_angle
        n_sampled += len(sampled_angle)
    return angle
