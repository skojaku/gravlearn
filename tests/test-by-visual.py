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

# %%
device = "cuda:1"

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)

labels = [G.nodes[i]["club"] for i in G.nodes]
sampler = gravlearn.RandomWalkSampler(A, walk_length=40, p=1, q=1)
walks = [sampler.sampling(i) for _ in range(1) for i in range(A.shape[0])]
base_emb = gravlearn.fastRP(A, 256, 10, 1)
model = gravlearn.GravLearnModel(A.shape[0], dim=32, base_emb = base_emb, normalize=False)
dist_metric = gravlearn.DistanceMetrics.COSINE
gravlearn.train(model, walks, device = device, bags =A ,window_length=5, dist_metric=dist_metric, batch_size=1024)

emb = model.forward(A)

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


# %%
sampler = gravlearn.RandomWalkSampler(net, walk_length=40, p=1, q=1)
walks = [sampler.sampling(i) for _ in range(1) for i in range(net.shape[0])]

base_emb = gravlearn.fastRP(net, 256, 10, 1)
model = gravlearn.GravLearnModel(net.shape[0], dim=32, base_emb = base_emb, normalize=False)
dist_metric = gravlearn.DistanceMetrics.EUCLIDEAN
gravlearn.train(model, walks, device = device, bags = net ,window_length=5, dist_metric=dist_metric, batch_size=1024)


# %%
model.eval()
emb = model.forward(net)
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
