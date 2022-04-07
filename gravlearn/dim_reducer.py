"""Dimensionality reduction using the soft max"""
# %%
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import torch
from torch import FloatTensor
import numpy as np
from torch.nn.utils.parametrizations import orthogonal
import geoopt


class ClassificationDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        assert X.shape[0] == Y.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx]


class SoftMaxClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, n_components=2):
        super(SoftMaxClassifier, self).__init__()

        self.encoder = orthogonal(nn.Linear(input_dim, n_components))
        self.decoder = nn.Linear(n_components, n_classes)
        self.softmax = nn.Softmax(dim=1)

        self.encoder.weight.required_grad = True
        self.decoder.weight.required_grad = True

    def forward(self, data, normalize=True):
        score = self.decoder(self.encoder(data))

        if normalize:
            score = self.softmax(score)
        return score


class SoftMaxReducer:
    def __init__(self):
        pass

    def fit(self, X, Y, batch_size=32, epochs=5, min_iter=100):

        Y = np.unique(Y, return_inverse=True)[1]
        n_classes = len(set(Y))
        self.model = SoftMaxClassifier(input_dim=X.shape[1], n_classes=n_classes)
        dataset = ClassificationDataset(
            torch.from_numpy(X).to(torch.float), torch.from_numpy(Y).to(torch.long)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        min_epochs = min_iter / (X.shape[0] / batch_size)
        epochs = np.minimum(epochs, np.ceil(min_epochs).astype(int))

        optim = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.003
        )
        pbar = tqdm(total=len(dataloader) * epochs)
        celoss = nn.CrossEntropyLoss()
        for ep in range(epochs):
            for it, (x, y) in enumerate(dataloader):
                # for iword, owords, nwords in pbar:
                for param in self.model.parameters():  # clear out the gradient
                    param.grad = None

                score = self.model.forward(x, normalize=False)
                loss = celoss(score, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optim.step()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        self.proj = self.model.encoder.weight.clone().detach().numpy().T
        self.proj = np.einsum(
            "ij,i->ij", self.proj, 1 / np.linalg.norm(self.proj, axis=1)
        )
        return self

    def transform(self, data):
        return data @ self.proj
        # return (
        #    self.model.encoder(torch.from_numpy(data).to(torch.float)).detach().numpy()
        # )


class SphericalPCA:
    """https://arxiv.org/pdf/1903.06877.pdf"""

    def __init__(self, mu=1, lam=3, iterations=30):
        super(SphericalPCA, self).__init__()
        self.iterations = iterations
        self.mu = mu
        self.lam = lam

    def fit_transform(self, X, dim):
        X = X.T  # swap
        n_rows, n_cols = X.shape

        U = np.random.randn(n_rows, dim)
        V = np.random.randn(dim, n_cols)

        for it in range(self.iterations):
            M = 2 * (X - U @ V) @ V.T + self.mu * U
            Y, s, Z = np.linalg.svd(M, full_matrices=False)
            U = Y @ Z

            Q = 2 * U.T @ X + (self.lam - 2) * V
            V = np.einsum("ij,j->ij", Q, 1 / np.linalg.norm(Q, axis=0))

        V, U = U, V.T  # swap back
        self.V = V
        return U

    def transform(self, X):
        return X @ self.V


#
# X = np.random.randn(100, 10)
# X = np.einsum("ij,i->ij", X, 1 / np.linalg.norm(X, axis=1))
#
# U, V = SphericalPCA(iterations=1000, mu=0, lam=0).fit_transform(X, 2)
# U.shape
## %%
#
