import unittest
import networkx as nx
import numpy as np
import gravlearn
import torch


class TestGravLearn(unittest.TestCase):
    def setUp(self):
        G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(G)
        self.labels = [G.nodes[i]["club"] for i in G.nodes]

        sampler = gravlearn.RandomWalkSampler(self.A, walk_length=40, p=1, q=1)
        self.seqs = [
            sampler.sampling(i) for _ in range(10) for i in range(self.A.shape[0])
        ]

    def test_grav_learn(self):
        device = "cpu"
        model = gravlearn.GravLearnModel(self.A.shape[0], 32)
        dist_metric = gravlearn.DistanceMetrics.COSINE
        model = gravlearn.train(
            model, self.seqs, window_length=5, dist_metric=dist_metric, device=device
        )
        emb = model.forward(torch.from_numpy(np.arange(self.A.shape[0])).to(device))

    def test_grav_learn_set(self):
        device = "cpu"
        model = gravlearn.GravLearnSetModel(self.A.shape[0], 32)
        dist_metric = gravlearn.DistanceMetrics.COSINE
        model = gravlearn.train(
            model,
            self.seqs,
            window_length=5,
            dist_metric=dist_metric,
            device=device,
            bags=self.A,
        )
        emb = model.forward(self.A)


if __name__ == "__main__":
    unittest.main()
