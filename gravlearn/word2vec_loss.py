"""Embedding module for Word2Vec Loss."""
from re import I
import torch.nn as nn
import torch
import numpy as np


class Word2VecTripletLoss(nn.Module):
    def __init__(self, embedding):
        super(Word2VecTripletLoss, self).__init__()
        self.embedding = embedding
        self.weights = None

    def forward(self, iword, oword, nword):
        ivectors = self.embedding.forward(iword, inner=True)
        ovectors = self.embedding.forward(oword, inner=False)
        nvectors = self.embedding.forward(nword, inner=False)
        oloss = (ovectors * ivectors).sum().sigmoid().clamp(1e-12, 1).log()
        nloss = (-nvectors * ivectors).sum().sigmoid().clamp(1e-12, 1).log()
        return -(oloss + nloss)


class Word2VecQuadletLoss(nn.Module):
    def __init__(self, embedding):
        super(Word2VecQuadletLoss, self).__init__()
        self.embedding = embedding
        self.weights = None

    def forward(self, iword, oword, inword, onword):
        ivectors = self.embedding.forward(iword, inner=True)
        ovectors = self.embedding.forward(oword, inner=False)
        invectors = self.embedding.forward(inword, inner=True)
        onvectors = self.embedding.forward(onword, inner=False)

        oloss = (ovectors * ivectors).sum().sigmoid().clamp(1e-12, 1).log()
        nloss = (-invectors * onvectors).sum().sigmoid().clamp(1e-12, 1).log()
        return -(oloss + nloss)


class Ref2VecTripletLoss(nn.Module):
    def __init__(self, embedding, refs):
        super(Ref2VecTripletLoss, self).__init__()
        self.embedding = embedding
        self.weights = None
        self.refs = refs
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, iword, oword, nword):
        ivectors = self.embedding.forward(self.refs[iword, :], inner=True)
        ovectors = self.embedding.forward(self.refs[oword, :], inner=False)
        nvectors = self.embedding.forward(self.refs[nword, :], inner=False)
        oloss = self.logsigmoid((ovectors * ivectors).sum(dim=1))
        nloss = self.logsigmoid((-nvectors * ivectors).sum(dim=1))
        return -(oloss + nloss).mean()


class Ref2VecTripletAngularLoss(nn.Module):
    def __init__(self, embedding, refs):
        super(Ref2VecTripletLoss, self).__init__()
        self.embedding = embedding
        self.weights = None
        self.refs = refs
        self.logsigmoid = nn.LogSigmoid()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, iword, oword, nword):
        ivectors = self.embedding.forward(self.refs[iword, :], inner=True)
        ovectors = self.embedding.forward(self.refs[oword, :], inner=False)
        nvectors = self.embedding.forward(self.refs[nword, :], inner=False)
        if self.embedding.normalize:
            vec_norm = torch.pow(self.embedding.radius.weight, 2)
            pos_angle = torch.acos(
                self.margin * (ivectors * ovectors).sum(dim=1) / vec_norm
            )
            neg_angle = torch.acos(
                self.margin * (ivectors * nvectors).sum(dim=1) / vec_norm
            )
            pos_rad = self.embedding.radius.weight
            neg_rad = self.embedding.radius.weight
        else:
            pos_angle = torch.acos(self.margin * self.cos(ivectors, ovectors))
            neg_angle = torch.acos(self.margin * self.cos(ivectors, nvectors))
            pos_rad = ivectors.norm(dim=1) * ovectors.norm(dim=1)
            neg_rad = ivectors.norm(dim=1) * nvectors.norm(dim=1)

        oloss = self.logsigmoid(-pos_angle * pos_rad / np.pi)
        nloss = self.logsigmoid(neg_angle * neg_rad / np.pi)
        return -(oloss + nloss).mean()


class Ref2VecQuadletLoss(nn.Module):
    def __init__(self, embedding, refs):
        super(Ref2VecQuadletLoss, self).__init__()
        self.embedding = embedding
        self.weights = None
        self.refs = refs
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, iword, oword, inword, onword):
        ivectors = self.embedding.forward(self.refs[iword, :], inner=True)
        ovectors = self.embedding.forward(self.refs[oword, :], inner=False)
        invectors = self.embedding.forward(self.refs[inword, :], inner=True)
        onvectors = self.embedding.forward(self.refs[onword, :], inner=False)

        oloss = self.logsigmoid((ovectors * ivectors).sum(dim=1))
        nloss = self.logsigmoid((-invectors * onvectors).sum(dim=1))
        return -(oloss + nloss).mean()


class Ref2VecQuadletAngularLoss(nn.Module):
    def __init__(self, embedding, refs):
        super(Ref2VecQuadletAngularLoss, self).__init__()
        self.embedding = embedding
        self.weights = None
        self.refs = refs
        self.logsigmoid = nn.LogSigmoid()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.margin = 1 - 5e-2  # margin to prevent arccos(-1, or 1)

    def forward(self, iword, oword, inword, onword):
        ivectors = self.embedding.forward(self.refs[iword, :], inner=True)
        ovectors = self.embedding.forward(self.refs[oword, :], inner=False)
        invectors = self.embedding.forward(self.refs[inword, :], inner=True)
        onvectors = self.embedding.forward(self.refs[onword, :], inner=False)

        if self.embedding.normalize:
            vec_norm = torch.pow(self.embedding.radius.weight, 2)
            pos_angle = torch.acos(
                self.margin * (ivectors * ovectors).sum(dim=1) / vec_norm
            )
            neg_angle = torch.acos(
                self.margin * (invectors * onvectors).sum(dim=1) / vec_norm
            )
            pos_rad = self.embedding.radius.weight
            neg_rad = self.embedding.radius.weight
        else:
            pos_angle = torch.acos(self.margin * self.cos(ivectors, ovectors))
            neg_angle = torch.acos(self.margin * self.cos(invectors, onvectors))
            pos_rad = ivectors.norm(dim=1) * ovectors.norm(dim=1)
            neg_rad = invectors.norm(dim=1) * onvectors.norm(dim=1)

        oloss = self.logsigmoid(-pos_angle * pos_rad / np.pi)
        nloss = self.logsigmoid(neg_angle * neg_rad / np.pi)
        return -(oloss + nloss).mean()
