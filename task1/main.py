from scripts.load_data import load_fashion_mnist
import numpy as np
from scripts.umap import run_umap, plot_final_embedding_umap, plot_low_dim_graph_umap
from scripts.tsne import run_tsne, plot_low_dim_graph_tsne, plot_final_embedding_tsne, animate_tsne



if __name__ == "__main__":

    #X, y = load_fashion_mnist(save=True)
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    # tsne
    X_subset = X[:1000]
    y_subset = y[:1000]

    X_tsne = run_tsne(X)
    plot_low_dim_graph_tsne(X_subset, y_subset)
    plot_final_embedding_tsne(X_tsne, y)
    animate_tsne(X_subset,y_subset)

    # umap
    X_umap = run_umap(X)
    plot_low_dim_graph_umap(X_subset, y_subset)
    plot_final_embedding_umap(X_umap, y, subset_size=10000)

