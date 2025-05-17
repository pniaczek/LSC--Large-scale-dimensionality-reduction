import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import umap

COLORS = plt.get_cmap("tab10")
FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def run_umap(X: np.ndarray,
             n_components: int = 2,
             n_neighbors: int = 15,
             min_dist: float = 0.1,
             metric: str = 'euclidean',
             random_state: int = 42) -> np.ndarray:
    """
    Perform UMAP dimensionality reduction on full dataset.

    Returns:
        np.ndarray: Transformed data of shape (N, n_components)
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    return reducer.fit_transform(X)

def plot_low_dim_graph_umap(X, y, n_neighbors=15, subset_size=1000, save_path="plots/low_dim_umap.png"):
    X = X[:subset_size]
    y = y[:subset_size]

    rng = np.random.RandomState(42)
    X_random = rng.normal(0, 0.01, size=(subset_size, 2))

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_random)
    knn_graph = nn.kneighbors_graph(X_random, mode='connectivity')
    knn_graph.setdiag(0)
    knn_graph.eliminate_zeros()
    G = nx.from_scipy_sparse_array(knn_graph)

    pos = {i: X_random[i] for i in range(subset_size)}
    node_colors = [COLORS(y[i] % 10) for i in range(subset_size)]

    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_color=node_colors, node_size=10, edge_color='gray', alpha=0.5)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=FASHION_CLASSES[i], markerfacecolor=COLORS(i), markersize=6) for i in range(10)]
    plt.legend(handles=handles, title="Class")
    plt.title(f"Low-dimensional graph BEFORE optimization (subset of {subset_size})")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def plot_final_embedding_umap(X_2d, y, subset_size=10000, title="Final UMAP embedding", save_path="plots/umap_final.png"):
    X_2d = X_2d[:subset_size]
    y = y[:subset_size]

    plt.figure(figsize=(8, 8))
    for i in range(10):
        mask = (y == i)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, color=COLORS(i), label=FASHION_CLASSES[i])
    plt.legend(title="Class")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

