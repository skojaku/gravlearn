# %%
%load_ext autoreload
%autoreload 2
import numpy as np
import pandas as pd
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import gravlearn

device = "cuda:1"
# %%

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)

labels = [G.nodes[i]["club"] for i in G.nodes]
A = sparse.csr_matrix(sparse.triu(A))
sampler = gravlearn.RandomWalkSampler(A, walk_length=40, p=1, q=1)
walks = [sampler.sampling(i) for _ in range(50) for i in range(A.shape[0])]
base_emb = gravlearn.NormalizedLaplacianEmbedding(A, dim = 6)
model = gravlearn.Word2Vec(A.shape[0], dim=32, normalize=False)
#model = gravlearn.PrecompressedWord2Vec(A.shape[0], dim=32, base_emb = base_emb, normalize=False)
#model = gravlearn.GravLearnModel(A.shape[0], dim=32, base_emb=base_emb, normalize=False)
dist_metric = gravlearn.DistanceMetrics.DOTSIM
gravlearn.train(
    model,
    walks,
    device=device,
    window_length=10,
    dist_metric=dist_metric,
    batch_size=128,
    train_by_triplet=True,
    context_window_type = "suc",
    epochs = 10
)

# %%
emb = model.forward(torch.arange(A.shape[0]).to(device))

# %%
clf = LinearDiscriminantAnalysis(n_components=1)
clf = clf.fit(emb, labels)
xy = PCA(n_components=2).fit_transform(emb)

sns.scatterplot(
    x=xy[:, 0],
    y=xy[:, 1],
    hue=labels,
    size=np.array(A.sum(axis=1)).reshape(-1) / 10,
    sizes=(1, 50),
)
sns.despine()
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
net = net + net.T

# %%
sampler = gravlearn.RandomWalkSampler(net, walk_length=40, p=1, q=1)
walks = [sampler.sampling(i) for _ in range(1) for i in range(net.shape[0])]
base_emb = gravlearn.NormalizedLaplacianEmbedding(net, dim = 64)
# %%


# %%
model = gravlearn.Word2Vec(net.shape[0], dim=32, normalize=False)
#model = gravlearn.PrecompressedWord2Vec(net.shape[0], dim=32, base_emb = base_emb, normalize=False)
#model = gravlearn.Bag2Vec(net.shape[0], dim=32, normalize=False)
#model = gravlearn.GravLearnModel(net.shape[0], dim=32, base_emb=base_emb, weights = weights)
#model = gravlearn.GravLearnModel(net.shape[0], dim=32, base_emb=base_emb, weights = weights)
dist_metric = gravlearn.DistanceMetrics.DOTSIM
gravlearn.train(
    model,
    walks,
    device=device,
    #bags=net,
    window_length=5,
    dist_metric=dist_metric,
    batch_size=1024,
    share_center=True,
    epochs = 1,
    train_by_triplet=True,
    context_window_type = "double"
)


# %%
model.eval()
#emb = model.forward(net)
emb = model.forward(torch.arange(net.shape[0]).to(device))

# %%
base_emb = gravlearn.fastRP(net, 64, 10, decay_rate = 1)
base_emb = np.einsum("ij,i->ij", base_emb, 1 / np.linalg.norm(base_emb, axis=1))
emb = base_emb.copy()
# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(7, 5))
clf = LinearDiscriminantAnalysis(n_components=2)
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
model.scale
# %%
np.linalg.norm(emb, axis=1)
# %% 3D plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

reducer = dim_reducer.SphericalPCA(lam=3, mu=0)
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
    s=np.array(net.sum(axis=1)).reshape(-1) / 10,
)
plt.show()

# %% 2D plot

reducer = dim_reducer.SphericalPCA(lam=3, mu=0)
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
