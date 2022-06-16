import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch import FloatTensor
from . import utils

from abc import ABC, abstractmethod


class EmbeddingModel(nn.Module):
    def forward(self, data, layer="center"):

        if layer == "center":
            x = self.forward_i(data)
        elif layer == "context":
            x = self.forward_o(data)

        if self.training is False:
            if next(self.parameters()).is_cuda:
                return x.detach().cpu().numpy()
            else:
                return x.detach().numpy()
        else:
            return x

    @abstractmethod
    def forward_i(self, data):
        pass

    @abstractmethod
    def forward_o(self, data):
        pass


class Word2Vec(EmbeddingModel):
    def __init__(self, vocab_size, dim, normalize=False):
        super(Word2Vec, self).__init__()
        # Layers
        self.ivectors = torch.nn.Embedding(vocab_size, dim, dtype=torch.float)
        self.ovectors = torch.nn.Embedding(vocab_size, dim, dtype=torch.float)
        self.scale = nn.Parameter(torch.Tensor([1]), requires_grad=False)

        # Parameters
        self.normalize = normalize
        self.training = True

    def forward_i(self, data):
        return self.ivectors(data)

    def forward_o(self, data):
        return self.ovectors(data)


class PrecompressedWord2Vec(Word2Vec):
    def __init__(self, vocab_size, dim, base_emb, normalize=False):
        super(Word2Vec, self).__init__()

        dim_base = base_emb.shape[1]

        self.base_vectors = torch.nn.Embedding(vocab_size, dim_base, dtype=torch.float)
        self.base_vectors.weight = nn.Parameter(
            FloatTensor(base_emb), requires_grad=False
        )

        # Layers
        self.ivectors = torch.nn.Linear(dim_base, dim, dtype=torch.float)
        self.ovectors = torch.nn.Linear(dim_base, dim, dtype=torch.float)
        self.scale = nn.Parameter(torch.Tensor([1]), requires_grad=False)

        # Parameters
        self.normalize = normalize
        self.training = True

    def forward_i(self, data):
        x = self.base_vectors(data)
        return self.ivectors(x)

    def forward_o(self, data):
        x = self.base_vectors(data)
        return self.ovectors(x)


class Bag2Vec(EmbeddingModel):
    def __init__(self, vocab_size, dim, weights=None, normalize=False):
        super(Bag2Vec, self).__init__()

        # Layers
        self.ivectors = torch.nn.EmbeddingBag(
            vocab_size,
            dim,
            mode="sum",
            include_last_offset=True,
            dtype=torch.float,
        )
        self.ovectors = torch.nn.EmbeddingBag(
            vocab_size,
            dim,
            mode="sum",
            include_last_offset=True,
            dtype=torch.float,
        )
        self.scale = nn.Parameter(torch.Tensor([1]), requires_grad=False)

        # Parameters
        self.normalize = normalize
        self.training = True

        if weights is None:
            weights = np.ones(vocab_size)
        self.weights = weights

    def forward_i(self, data):
        return self._bag2vec(data, return_context_vector=False)

    def forward_o(self, data):
        return self._bag2vec(data, return_context_vector=True)

    def _bag2vec(self, data, return_context_vector=False):
        A = utils.to_adjacency_matrix(data).astype(float)
        A.data *= self.weights[A.indices]  # Apply weights

        # Row normalize
        rowsum = np.array(A.sum(axis=1)).reshape(-1)
        A = sparse.diags(1 / np.maximum(1e-15, rowsum)) @ A

        offsets, indices, weight = (
            torch.from_numpy(A.indptr).to(torch.long),
            torch.from_numpy(A.indices).to(torch.long),
            torch.from_numpy(A.data).to(torch.float),
        )

        if self.ivectors.weight.is_cuda:
            device = self.ovectors.weight.get_device()
            indices = indices.cuda(device)
            offsets = offsets.cuda(device)
            weight = weight.cuda(device)

        if return_context_vector:
            x = self.ovectors(
                input=indices,
                offsets=offsets,
                per_sample_weights=weight,
            )
        else:
            x = self.ivectors(
                input=indices,
                offsets=offsets,
                per_sample_weights=weight,
            )
        return x


class GravLearnModel(EmbeddingModel):
    def __init__(
        self,
        vocab_size,
        dim,
        base_emb,
        weights=None,
        normalize=False,
    ):
        super(GravLearnModel, self).__init__()

        # Layers
        base_emb_dim = base_emb.shape[1]
        self.reducer = torch.nn.EmbeddingBag(
            vocab_size,
            base_emb_dim,
            mode="sum",
            include_last_offset=True,
            dtype=torch.float,
        )
        self.middle_layer = nn.Linear(base_emb_dim, base_emb_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.scale = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.output_layer = torch.nn.Linear(
            base_emb_dim, dim, dtype=torch.float, bias=True
        )

        # Parameters
        self.normalize = normalize
        self.training = True

        if weights is None:
            weights = np.ones(base_emb.shape[0])
        self.weights = weights

        # Initialize Layers
        self.reducer.weight = nn.Parameter(FloatTensor(base_emb), requires_grad=False)

    def forward_i(self, data):

        x = self._input_to_vec(data)
        # x = F.normalize(x, p=2, dim=1)  # to prevent the centering
        x = self.middle_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = F.normalize(x, p=2, dim=1) if self.normalize else x

        if self.training is False:
            if self.output_layer.weight.is_cuda:
                return x.detach().cpu().numpy()
            else:
                return x.detach().numpy()
        else:
            return x

    def forward_o(self, data):
        return self.forward_i(data)

    def _input_to_vec(self, data):
        A = utils.to_adjacency_matrix(data).astype(float)
        A.data *= self.weights[A.indices]  # Apply weights

        # Row normalize
        rowsum = np.array(A.sum(axis=1)).reshape(-1)
        A = sparse.diags(1 / np.maximum(1e-15, rowsum)) @ A

        offsets, indices, weight = (
            torch.from_numpy(A.indptr).to(torch.long),
            torch.from_numpy(A.indices).to(torch.long),
            torch.from_numpy(A.data).to(torch.float),
        )

        if self.output_layer.weight.is_cuda:
            device = self.output_layer.weight.get_device()
            indices = indices.cuda(device)
            offsets = offsets.cuda(device)
            weight = weight.cuda(device)

        x = self.reducer(
            input=indices,
            offsets=offsets,
            per_sample_weights=weight,
        )
        return x
