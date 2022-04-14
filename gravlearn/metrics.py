"""Metrics"""
import torch
from enum import Enum
import torch.nn.functional as F


class DistanceMetrics(Enum):
    """
    The metric for the loasses
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    ANGULAR = lambda x, y: torch.arccos((1 - 1e-2) * F.cosine_similarity(x, y))
    DOTSIM = lambda x, y: -(x * y).sum(dim=1)

    def is_scale_invariant(dist_metric):
        return torch.isclose(
            dist_metric(torch.ones(1, 2), torch.ones(1, 2)),
            dist_metric(torch.ones(1, 2), 2 * torch.ones(1, 2)),
        )
