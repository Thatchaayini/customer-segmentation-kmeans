import os

import matplotlib
matplotlib.use("Agg")  # save charts as image files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from preprocess import preprocess
from segmentation import segment_customers


def reduce_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return coords, pca.explained_variance_ratio_


def reduce_tsne(X_scaled, n_components=2, perplexity=30):
    # perplexity must be smaller than the number of samples
    perplexity = min(perplexity, max(1, len(X_scaled) - 1))
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init="pca",
        random_state=42,
    )
    return tsne.fit_transform(X_scaled)


def plot_clusters(coords, labels, title, xlabel, ylabel, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=labels, cmap="viridis", s=40, alpha=0.8
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(*scatter.legend_elements(), title="Cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def visualize_segments(n_clusters=3, out_dir="images"):
    df, X_scaled, scaler = preprocess()
    result, model, k = segment_customers(df, X_scaled, n_clusters)
    labels = result["Cluster"].to_numpy()

    pca_coords, variance = reduce_pca(X_scaled)
    plot_clusters(
        pca_coords, labels,
        f"Customer Segments (PCA, K={k})",
        f"PC1 ({variance[0]:.0%} variance)",
        f"PC2 ({variance[1]:.0%} variance)",
        f"{out_dir}/pca_clusters.png",
    )

    tsne_coords = reduce_tsne(X_scaled)
    plot_clusters(
        tsne_coords, labels,
        f"Customer Segments (t-SNE, K={k})",
        "t-SNE 1", "t-SNE 2",
        f"{out_dir}/tsne_clusters.png",
    )
    return variance


if __name__ == "__main__":
    variance = visualize_segments()
    print(f"PCA explained variance: PC1 = {variance[0]:.1%}, PC2 = {variance[1]:.1%}")
    print("Saved: images/pca_clusters.png")
    print("Saved: images/tsne_clusters.png")