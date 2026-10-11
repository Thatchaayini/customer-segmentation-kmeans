import os

import matplotlib
matplotlib.use("Agg")  # save charts as image files, no window needed
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES, DEFAULT_K, LINKAGE
from evaluation import evaluate_clustering
from preprocess import preprocess


def hierarchical_clustering(X_scaled, n_clusters=DEFAULT_K, linkage_method=LINKAGE):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    return model.fit_predict(X_scaled)


def dbscan_clustering(X_scaled, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    """Label -1 means noise (outlier)."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)


def evaluate_ignoring_noise(X_scaled, labels):
    """Score the clusters, leaving out DBSCAN noise points (label -1)."""
    X = X_scaled.to_numpy() if hasattr(X_scaled, "to_numpy") else X_scaled
    mask = labels != -1
    n_noise = int((~mask).sum())
    try:
        metrics = evaluate_clustering(X[mask], labels[mask])
    except ValueError:
        metrics = None  # fewer than 2 clusters found
    return metrics, n_noise


def save_dendrogram(X_scaled, path="images/dendrogram.png", method=LINKAGE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Z = linkage(X_scaled, method=method)
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, truncate_mode="lastp", p=20, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Cluster size (or customer index)")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df, X_scaled, scaler = preprocess()

    hier_labels = hierarchical_clustering(X_scaled)
    print("Hierarchical:", evaluate_ignoring_noise(X_scaled, hier_labels)[0])

    db_labels = dbscan_clustering(X_scaled)
    n_clusters = len(set(db_labels) - {-1})
    metrics, n_noise = evaluate_ignoring_noise(X_scaled, db_labels)
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points, {metrics}")

    save_dendrogram(X_scaled)
    print("Saved: images/dendrogram.png")