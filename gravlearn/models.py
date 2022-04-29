import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch import FloatTensor
from .fastRP import fastRP
from . import utils


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, dim, normalize=False):
        super(Word2Vec, self).__init__()
        # Layers
        self.ivectors = torch.nn.Embedding(vocab_size, dim, dtype=torch.float)
        self.scale = nn.Parameter(torch.Tensor([1]), requires_grad=False)

        # Parameters
        self.normalize = normalize
        self.training = True

    def forward(self, data):
        x = self.ivectors(data)
        if self.training is False:
            if self.ivectors.weight.is_cuda:
                return x.detach().cpu().numpy()
            else:
                return x.detach().numpy()
        else:
            return x


# class GravLearnModel(nn.Module):
#    def __init__(self, vocab_size, dim, normalize=False):
#        super(GravLearnModel, self).__init__()
#
#        # Layers
#        self.reducer = torch.nn.Embedding(vocab_size, dim, dtype=torch.float)
#        self.relu = nn.LeakyReLU()
#        self.dropout = nn.Dropout(p=0.2)
#        self.scale = nn.Parameter(torch.Tensor([1]), requires_grad=True)
#        self.output_layer = torch.nn.Linear(dim, dim, dtype=torch.float, bias=True)
#
#        # Parameters
#        self.normalize = normalize
#        self.training = True
#
#    def forward(self, data):
#
#        x = self.reducer(data)
#        x = self.dropout(x)
#        x = self.relu(x)
#        x = self.output_layer(x)
#        x = F.normalize(x, p=2, dim=1) if self.normalize else x
#
#        if self.training is False:
#            if self.output_layer.weight.is_cuda:
#                return x.detach().cpu().numpy()
#            else:
#                return x.detach().numpy()
#        else:
#            return x


class GravLearnModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim,
        base_emb,
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
        self.middle_layer = nn.Linear(base_emb_dim, dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.scale = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.output_layer = torch.nn.Linear(dim, dim, dtype=torch.float, bias=True)

        # Parameters
        self.normalize = normalize
        self.training = True

        # Initialize Layers
        self.reducer.weight = nn.Parameter(FloatTensor(base_emb), requires_grad=False)

    def forward(self, data):

        x = self._input_to_vec(data)
        x = F.normalize(x, p=2, dim=1)  # to prevent the centering
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

    def _input_to_vec(self, data):
        A = utils.to_adjacency_matrix(data).astype(float)
        deg = np.array(A.sum(axis=1)).reshape(-1)
        A = sparse.diags(1 / np.maximum(1e-32, deg)) @ A
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
