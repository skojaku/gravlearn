import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import networkx as nx
from numba import njit


def load_network(func):
    def wrapper(binarize=True, symmetrize=False, k_core=None, *args, **kwargs):
        net, labels, node_table = func(*args, **kwargs)
        if symmetrize:
            net = net + net.T
            net.sort_indices()

        if k_core is not None:
            knumbers = k_core_decomposition(net)
            s = knumbers >= k_core
            net = net[s, :][:, s]
            labels = labels[s]
            node_table = node_table[s]

        _, comps = connected_components(csgraph=net, directed=False, return_labels=True)
        ucomps, freq = np.unique(comps, return_counts=True)
        s = comps == ucomps[np.argmax(freq)]
        labels = labels[s]
        net = net[s, :][:, s]
        if binarize:
            net = net + net.T
            net.data = net.data * 0 + 1
        node_table = node_table[s]
        return net, labels, node_table

    return wrapper


@load_network
def load_airport_net():
    # Node attributes
    node_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/node-table-airport.csv"
    )

    # Edge table
    edge_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/edge-table-airport.csv"
    )
    # net = nx.adjacency_matrix(nx.from_pandas_edgelist(edge_table))

    net = sparse.csr_matrix(
        (
            edge_table["weight"].values,
            (edge_table["source"].values, edge_table["target"].values),
        ),
        shape=(node_table.shape[0], node_table.shape[0]),
    )

    s = ~pd.isna(node_table["region"])
    node_table = node_table[s]
    labels = node_table["region"].values
    net = net[s, :][:, s]
    return net, labels, node_table


def load_karate_club_net():
    G = nx.karate_club_graph()
    net = nx.adjacency_matrix(G)
    labels = [G.nodes[i]["club"] for i in G.nodes]
    node_table = pd.DataFrame({"node_id": np.arange(net.shape[0]), "label": labels})
    return net, labels, node_table


def k_core_decomposition(A):
    n_nodes = A.shape[0]
    deg = np.array(A.sum(axis=1)).reshape(-1).astype(int)
    return _k_core_decomposition(A.indptr, A.indices, n_nodes, deg)


@njit(nogil=True)
def _k_core_decomposition(A_indptr, A_indices, n_nodes, deg):

    L = np.zeros(n_nodes)
    k = 1
    n_removed = 0
    mink = 1
    while n_removed < n_nodes:
        removed = True
        while removed:
            removed = False
            mink = n_nodes + 1
            for i in range(n_nodes):
                if L[i] > 0:
                    continue
                if deg[i] <= k:
                    L[i] = k
                    removed = True
                    n_removed += 1
                    for j in A_indices[A_indptr[i] : A_indptr[i + 1]]:
                        if L[j] > 0:
                            continue
                        deg[j] -= 1
                else:
                    mink = np.minimum(mink, deg[i])
        k = mink
    return L
