"""Test by visual inspection of the generated embeddings."""
# %%
%load_ext autoreload
%autoreload 2
%matplotlib ipympl
import numpy as np
import pandas as pd
from ref2vec import dim_reducer
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import networkx as nx
import ref2vec as gn
import torch
import residual2vec
# %%
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)

labels = [G.nodes[i]["club"] for i in G.nodes]
sampler = gn.RandomWalkSampler(A, walk_length=40, p=1, q=1)
walks = [sampler.sampling(i) for _ in range(10) for i in range(A.shape[0])]

model = gn.train_ref2vec(
    seqs=walks,
    refs_for_seq=A,
    conv_matrix=A,
    conv_decay_rate=1.0,
    window_length=5,
    epochs = 1,
    dim=64,
    batch_size = 256,
    learn_joint_prob=True,
    checkpoint=100,
    outputfile="test.pth",
    weights=None,  # 1/np.array(A.sum(axis = 1)).reshape(-1),
    cuda=True,
    normalize=True,
)

# %%
model.eval()
emb = model.forward(A)

# %%
model

# %% 3d plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
reducer = dim_reducer.SphericalPCA()
xyz = reducer.fit_transform(emb, dim=3)

# creating figure
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# creating the plot
cmap = sns.color_palette().as_hex()
plot_geeks = ax.scatter(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    color=[cmap[i] for i in np.unique(labels, return_inverse=True)[1]],
)
plt.show()
# %%
model.radius.weight
# %%

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis(n_components=1)
clf = clf.fit(emb, labels)
clf.score(emb, labels)


# reducer.model.encoder.weight.data.numpy().shape
# logging.getLogger("urllib3").setLevel(level=logging.DEBUG)
# %%
# Node attributes
node_table = pd.read_csv(
    "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/node-table-airport.csv"
)

# Edge table
edge_table = pd.read_csv(
    "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/edge-table-airport.csv"
)
# net = nx.adjacency_matrix(nx.from_pandas_edgelist(edge_table))

net = sparse.csr_matrix(
    (
        edge_table["weight"].values,
        (edge_table["source"].values, edge_table["target"].values),
    ),
    shape=(node_table.shape[0], node_table.shape[0]),
)

s = ~pd.isna(node_table["region"])
labels = node_table[s]["region"].values
net = net[s, :][:, s]


# %%
sampler = gn.RandomWalkSampler(net, walk_length=40, p=1, q=1)
walks = [sampler.sampling(i) for _ in range(1) for i in range(net.shape[0])]

model = gn.train_ref2vec(
    seqs=walks,
    refs_for_seq=net,
    conv_matrix=net,
    conv_decay_rate=1.0,
    conv_steps=5,
    conv_layer_size=512,
    window_length=5,
    dim=64,
    learn_joint_prob=True,
    batch_size=256,
    checkpoint=5000,
    outputfile="test.pth",
    weights=None,
    cuda=True,
    normalize=True
    # weights=1/np.array(net.sum(axis = 1)).reshape(-1),
)

# %%
model.eval()
emb = model.forward(net)
# %%
model.radius.weight
# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(7, 5))
clf = LinearDiscriminantAnalysis(n_components=2, shrinkage = 0.9, solver = "eigen")
clf = clf.fit(emb, labels)
xy = clf.transform(emb)
sns.scatterplot(
    x=xy[:, 0],
    y=xy[:, 1],
    hue=labels,
    size=np.array(net.sum(axis=1)).reshape(-1) / 10,
    sizes=(1, 50),
    ax=ax,
)
sns.despine()
clf.score(emb, labels)
# %%
np.linalg.norm(emb, axis = 1)
# %% 3D plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
reducer = dim_reducer.SphericalPCA(lam = 3, mu = 0)
xyz = reducer.fit_transform(emb, dim=3)

# creating figure
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# creating the plot
cmap = sns.color_palette().as_hex()
plot_geeks = ax.scatter(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    color=[cmap[i] for i in np.unique(labels, return_inverse=True)[1]],
    s =np.array(net.sum(axis=1)).reshape(-1) / 10,
)
plt.show()

# %% 2D plot

reducer = dim_reducer.SphericalPCA(lam = 3, mu = 0)
xy = reducer.fit_transform(emb, dim=2)

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    x=xy[:, 0],
    y=xy[:, 1],
    hue=labels,
    size=np.array(net.sum(axis=1)).reshape(-1) / 10,
    sizes=(1, 50),
    ax=ax,
)
sns.despine()
# %%

# %%
