[![Unit Test & Deploy](https://github.com/skojaku/gravlearn/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/gravlearn/actions/workflows/main.yml)

```python
import networkx as nx
import gravlearn
import torch

# Load data
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
labels = [G.nodes[i]["club"] for i in G.nodes]

# Generate the sequence for demo
sampler = gravlearn.RandomWalkSampler(A, walk_length=40, p=1, q=1)
walks = [sampler.sampling(i) for _ in range(10) for i in range(A.shape[0])]

# Training
model = gravlearn.GravLearnSetModel(A.shape[0], 32) # Embedding based on set
#model = gravlearn.GravLearnModel(A.shape[0], 32) # Embedding based on one-hot vector

dist_metric = gravlearn.DistanceMetrics.COSINE
model = gravlearn.train(model, walks, device = device, bags =A ,window_length=5, dist_metric=dist_metric)
#model = gravlearn.train(model, walks, device = device, window_length=5, dist_metric=dist_metric)

# Embedding
emb = model.forward(A)
# emb = model.forward(torch.arange(A.shape[0]))
```
