"""Quadlet Loss."""
import torch.nn as nn
from .metrics import DistanceMetrics


class QuadletLoss(nn.Module):
    def __init__(self, embedding, dist_metric=DistanceMetrics.COSINE):
        super(QuadletLoss, self).__init__()
        self.embedding = embedding
        self.weights = None
        self.dist_func = dist_metric
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, iword, oword, inword, onword):
        ivectors = self.embedding.forward(iword)
        ovectors = self.embedding.forward(oword)
        invectors = self.embedding.forward(inword)
        onvectors = self.embedding.forward(onword)

        oloss = self.logsigmoid(
            -self.embedding.concentration * self.dist_func(ivectors, ovectors)
        )
        nloss = self.logsigmoid(
            -self.embedding.concentration * self.dist_func(invectors, onvectors).neg()
        )
        return -(oloss + nloss).mean()
