from scipy import sparse
import numpy as np
from sklearn.decomposition import TruncatedSVD


def fastRP(A, dim, steps, decay_rate):
    """https://arxiv.org/pdf/1908.11512.pdf."""
    R = sparse.random(
        A.shape[0],
        dim,
        density=1 / 3,
        random_state=42,
        data_rvs=lambda x: np.random.choice(
            [-np.sqrt(3), np.sqrt(3)], size=x, replace=True
        ),
    ).toarray()

    S = np.zeros(R.shape)
    denom = np.maximum(np.array(A.sum(axis=1)).reshape(-1), 1e-32)
    normalized_conv_matrix = sparse.diags(1 / denom) @ A.copy()
    if np.isclose(decay_rate, 1):
        for _ in range(steps):
            S += R
            S = normalized_conv_matrix @ S
        S /= np.maximum(1, steps)
    else:
        for _ in range(steps):
            S += R
            S = decay_rate * normalized_conv_matrix @ S
        S *= 1 - decay_rate
    return S


def NormalizedLaplacianEmbedding(A, dim):
    # calculate the normalized laplacian
    indeg = np.array(A.sum(axis=0)).reshape(-1)
    outdeg = np.array(A.sum(axis=1)).reshape(-1)

    indeg_inv = np.sqrt(1 / np.maximum(1, indeg))
    outdeg_inv = np.sqrt(1 / np.maximum(1, outdeg))

    L = sparse.diags(outdeg_inv) @ A @ sparse.diags(indeg_inv)
    svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
    return svd.fit_transform(L)
