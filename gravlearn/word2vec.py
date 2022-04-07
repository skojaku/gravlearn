"""Word2Vec"""
import geoopt
import torch.nn as nn
from torch import FloatTensor, LongTensor


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=padding_idx
        )
        self.ovectors = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=padding_idx
        )
        self.ivectors.weight = geoopt.ManifoldParameter(
            FloatTensor(self.vocab_size + 1, self.embedding_size).uniform_(
                -0.5 / self.embedding_size, 0.5 / self.embedding_size
            ),
            manifold=geoopt.Sphere(),
        )
        self.ovectors.weight = geoopt.ManifoldParameter(
            FloatTensor(self.vocab_size + 1, self.embedding_size).uniform_(
                -0.5 / self.embedding_size, 0.5 / self.embedding_size
            ),
            manifold=geoopt.Sphere(),
        )
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LongTensor(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LongTensor(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)
