import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch import FloatTensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_sampler import QuadletDataset, TripletDataset
from .word2vec_loss import (
    Ref2VecQuadletLoss,
    Ref2VecQuadletAngularLoss,
    Ref2VecTripletLoss,
)


def train_ref2vec(
    seqs,
    refs_for_seq,
    window_length,
    dim,
    conv_matrix=None,
    conv_steps=5,
    conv_decay_rate=1.0,
    conv_layer_size=512,
    batch_size=256,
    cuda=True,
    buffer_size=100000,
    epochs=1,
    learn_joint_prob=True,
    checkpoint=5000,
    outputfile=None,
    weights=None,
    normalize=False,
):
    n_elements = refs_for_seq.shape[1]

    # Set dataset
    if learn_joint_prob:  # Learning the joint probability distribution
        dataset = QuadletDataset(
            seqs=seqs,
            window_length=window_length,
            padding_id=n_elements,
            buffer_size=buffer_size,
            epochs=epochs,
        )
    else:
        dataset = TripletDataset(
            seqs=seqs,
            window_length=window_length,
            padding_id=n_elements,
            buffer_size=buffer_size,
            epochs=epochs,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # Set up the embedding model
    model = ref2vec(
        vocab_size=n_elements,
        dim=dim,
        learn_joint_prob=learn_joint_prob,
        weights=weights,
        conv_matrix=conv_matrix,
        conv_steps=conv_steps,
        conv_decay_rate=conv_decay_rate,
        conv_layer_size=conv_layer_size,
        normalize=normalize,
    )
    model.train()
    if cuda:
        model = model.cuda()

    # Set up the loss functions
    if learn_joint_prob:  # Learning the joint probability distribution
        neg_sampling = Ref2VecQuadletAngularLoss(embedding=model, refs=refs_for_seq)
    else:
        neg_sampling = Ref2VecTripletLoss(embedding=model, refs=refs_for_seq)

    # Training
    focal_params = filter(lambda p: p.requires_grad, model.parameters())
    optim = Adam(focal_params, lr=0.003)

    pbar = tqdm(enumerate(dataloader), miniters=100, total=len(dataloader))
    for it, data_samples in pbar:
        # for iword, owords, nwords in pbar:
        focal_params = filter(lambda p: p.requires_grad, model.parameters())
        for param in focal_params:  # clear out the gradient
            param.grad = None
        loss = neg_sampling(*data_samples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(focal_params, 1)
        optim.step()
        pbar.set_postfix(loss=loss.item())

        if (it + 1) % checkpoint == 0:
            if outputfile is not None:
                torch.save(model.state_dict(), outputfile)
                pbar.set_description("Saving model at iteration {}".format(it + 1))

    if outputfile is not None:
        torch.save(model.state_dict(), outputfile)
    model.eval()
    return model


def load_ref2vec(filename):
    weights = torch.load(filename)
    vocab_size = weights["reducer.weight"].shape[0]
    conv_layer_size = weights["reducer.weight"].shape[1]
    dim = weights["ivectors.weight"].shape[1]
    model = ref2vec(vocab_size=vocab_size, dim=dim, conv_layer_size=conv_layer_size)
    model.load_state_dict(weights)
    model.eval()
    return model


class ref2vec(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim,
        conv_matrix=None,
        learn_joint_prob=True,
        weights=None,
        conv_layer_size=256,
        conv_steps=10,
        conv_decay_rate=0.9,
        normalize=True,
    ):
        super(ref2vec, self).__init__()

        # Layers
        self.reducer = torch.nn.EmbeddingBag(
            vocab_size,
            conv_layer_size,
            mode="sum",
            include_last_offset=True,
            dtype=torch.float,
        )
        self.linear_middle = nn.Linear(conv_layer_size, dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.radius = torch.nn.Linear(1, 1, dtype=torch.float, bias=False)
        self.ivectors = torch.nn.Linear(dim, dim, dtype=torch.float, bias=True)
        self.ovectors = torch.nn.Linear(dim, dim, dtype=torch.float, bias=False)

        # Parameters
        self.learn_joint_prob = learn_joint_prob
        self.normalize = normalize
        self.training = True
        self.weights = weights

        # Initialize Layers
        if conv_matrix is not None:
            assert vocab_size == conv_matrix.shape[0]

            RP = fastRP(
                conv_matrix, conv_layer_size, conv_steps, conv_decay_rate
            )  # random projection
            self.reducer.weight = nn.Parameter(FloatTensor(RP), requires_grad=False)

    def forward(self, data, inner=True):

        x = self.data2vec(data)
        x = F.normalize(x, p=2, dim=1)  # to prevent the centering
        x = self.linear_middle(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.ivectors(x) if (inner or self.learn_joint_prob) else self.ovectors(x)
        x = self.radius.weight * F.normalize(x, p=2, dim=1) if self.normalize else x

        if self.training is False:
            if self.ivectors.weight.is_cuda:
                return x.detach().cpu().numpy()
            else:
                return x.detach().numpy()
        else:
            return x

    def data2vec(self, data):
        A = to_adjacency_matrix(data).astype(float)
        if self.weights is not None:
            A.data = A.data * self.weights[A.indices]
        deg = np.array(A.sum(axis=1)).reshape(-1)
        A = sparse.diags(1 / np.maximum(1e-32, deg)) @ A
        offsets, indices, weight = (
            torch.from_numpy(A.indptr).to(torch.long),
            torch.from_numpy(A.indices).to(torch.long),
            torch.from_numpy(A.data).to(torch.float),
        )
        if self.ivectors.weight.is_cuda:
            indices = indices.cuda()
            offsets = offsets.cuda()
            weight = weight.cuda()

        x = self.reducer(
            input=indices,
            offsets=offsets,
            per_sample_weights=weight,
        )
        return x


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


# class ref2vec(nn.Module):
#    def __init__(
#        self,
#        vocab_size,
#        dim,
#        learn_joint_prob=True,
#        weights=None,
#        normalize=True,
#    ):
#        super(ref2vec, self).__init__()
#        self.normalize = normalize
#        self.radius = torch.nn.Linear(1, 1, dtype=torch.float, bias=True)
#
#        self.dropout = nn.Dropout(p=0.2)
#        self.relu = nn.LeakyReLU()
#        self.linear_output = nn.Linear(dim, dim)
#
#        # Initialize the embedding vectors
#        self.ivectors = torch.nn.EmbeddingBag(
#            vocab_size,
#            dim,
#            mode="sum",
#            include_last_offset=True,
#            dtype=torch.float,
#        )
#
#        # Intiialize the otu-vector if learn_joint_prob is False
#        self.learn_joint_prob = learn_joint_prob
#        if self.learn_joint_prob == False:
#            self.ovectors = torch.nn.EmbeddingBag(
#                vocab_size,
#                dim,
#                mode="sum",
#                include_last_offset=True,
#                dtype=torch.float,
#            )
#
#        # Set the training mode on
#        self.training = True
#        self.weights = weights
#
#    def forward(self, data, inner=True):
#        A = to_adjacency_matrix(data).astype(float)
#
#        if self.weights is not None:
#            A.data = A.data * self.weights[A.indices]
#        deg = np.array(A.sum(axis=1)).reshape(-1)
#        A = sparse.diags(1 / np.maximum(1e-32, deg)) @ A
#
#        offsets = torch.from_numpy(A.indptr).to(torch.long)
#        indices = torch.from_numpy(A.indices).to(torch.long)
#        weight = torch.from_numpy(A.data).to(torch.float)
#
#        if self.ivectors.weight.is_cuda:
#            indices = indices.cuda()
#            offsets = offsets.cuda()
#            weight = weight.cuda()
#
#        if inner or self.learn_joint_prob:
#            x = self.ivectors(
#                input=indices,
#                offsets=offsets,
#                per_sample_weights=weight,
#            )
#        else:
#            x = self.ovectors(
#                input=indices,
#                offsets=offsets,
#                per_sample_weights=weight,
#            )
#
#        x = F.normalize(x, p=2, dim=1)
#        x = self.dropout(x)
#        x = self.relu(x)
#        x = self.linear_output(x)
#
#        if self.normalize:
#            x = self.radius.weight * F.normalize(x, p=2, dim=1)
#
#        if self.training is False:
#            if self.ivectors.weight.is_cuda:
#                return x.detach().cpu().numpy()
#            else:
#                return x.detach().numpy()
#        else:
#            return x
#
#    def load(self, filename):
#        state_dict = torch.load(filename)
#        for k, v in state_dict.items():
#            if k == "ivectors.weight":
#                self.ivectors.weight = nn.Parameter(v, requires_grad=True)
#            if k == "ovectors.weight":
#                self.ovectors.weight = nn.Parameter(v, requires_grad=True)
#            if k == "reducer.weight":
#                self.reducer.weight = nn.Parameter(v, requires_grad=False)
#            if k == "radius.weight":
#                self.radius.weight = nn.Parameter(v, requires_grad=True)
#            if k == "linear_output.weight":
#                self.linear_output.weight = nn.Parameter(v, requires_grad=True)
#        self.eval()
#        return self
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
        ValueError("Unexpected data type {} for the adjacency matrix".format(type(net)))
