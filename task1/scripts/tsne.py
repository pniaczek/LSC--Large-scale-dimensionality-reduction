import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from sklearn.manifold import _t_sne
import types

COLORS = plt.get_cmap("tab10")
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def run_tsne(X: np.ndarray, n_components: int = 2, perplexity: float = 30.0,
             learning_rate: float = 200.0, n_iter: int = 1000,
             random_state: int = 42) -> np.ndarray:
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                init='pca',
                random_state=random_state,
                verbose=1)
    return tsne.fit_transform(X)

def plot_final_embedding_tsne(X_2d, y, subset_size=10000, title="Final t-SNE embedding",
                    save_path="plots/tsne_final.png"):
    X_2d = X_2d[:subset_size]
    y = y[:subset_size]
    plt.figure(figsize=(8, 8))
    for i in range(10):
        mask = (y == i)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, color=COLORS(i), label=FASHION_CLASSES[i])
    plt.legend(title="Class")
    plt.title(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def plot_low_dim_graph_tsne(X, y, n_neighbors=15, subset_size=1000, save_path="plots/tsne_low_dim.png"):
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
    plt.title(f"Low-dimensional graph BEFORE t-SNE (subset of {subset_size})")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()




def animate_tsne(X, y, n_iter=500, perplexity=30.0, learning_rate=200.0, interval=100, save_path='plots/tsne_animation.gif'):
    positions = []

    original_gradient_descent = _t_sne._gradient_descent

    def _gradient_descent(objective, p0, it, max_iter, n_iter_check=1,
                        n_iter_without_progress=30, momentum=0.5,
                        learning_rate=1000.0, min_gain=0.01,
                        min_grad_norm=1e-7, min_error_diff=1e-7,
                        verbose=0, args=None, kwargs=None):

        n_iter = max_iter

        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(float).max
        best_error = np.finfo(float).max

        best_iter = 0
        
        for i in range(it, n_iter):
            positions.append(p.copy())
            new_error, grad = objective(p, *args)
            error_diff = np.abs(new_error - error)
            error = new_error
            grad_norm = np.linalg.norm(grad)

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                break
            if min_grad_norm >= grad_norm:
                break
            if min_error_diff >= error_diff:
                break

            inc = update * grad >= 0.0
            dec = np.invert(inc)
            gains[inc] += 0.05
            gains[dec] *= 0.95
            gains = np.clip(gains, min_gain, np.inf)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update

        return p, error, i

    _t_sne._gradient_descent = _gradient_descent

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
                n_iter=n_iter, init='random', random_state=42, verbose=1)
    X_embedded = tsne.fit_transform(X)

    _t_sne._gradient_descent = original_gradient_descent

    positions = np.array(positions).reshape(-1, X.shape[0], 2)

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(i):
        ax.clear()
        for j in range(10):
            mask = (y == j)
            ax.scatter(positions[i][mask, 0], positions[i][mask, 1], s=5, label=str(j))
        ax.legend(loc='best', fontsize=6)
        ax.set_title(f't-SNE Iteration {i+1}')
        ax.set_xlim(np.min(positions[:, :, 0]), np.max(positions[:, :, 0]))
        ax.set_ylim(np.min(positions[:, :, 1]), np.max(positions[:, :, 1]))

    ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=interval)
    ani.save(save_path, writer='pillow')
    plt.close()
