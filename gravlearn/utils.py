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
