import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from advanced_clustering import (
    dbscan_clustering,
    evaluate_ignoring_noise,
    hierarchical_clustering,
    save_dendrogram,
)


def blobs():
    X, _ = make_blobs(n_samples=90, centers=3, cluster_std=0.5, random_state=0)
    return pd.DataFrame(X, columns=["a", "b"])


def test_hierarchical_returns_requested_clusters():
    labels = hierarchical_clustering(blobs(), n_clusters=3)
    assert len(set(labels)) == 3


def test_dbscan_marks_far_point_as_noise():
    X = blobs()
    X.loc[len(X)] = [100, 100]
    labels = dbscan_clustering(X, eps=1.0, min_samples=5)
    assert labels[-1] == -1


def test_evaluate_ignoring_noise_counts_noise():
    X = blobs()
    labels = np.zeros(len(X), dtype=int)
    labels[:45] = 1
    labels[-3:] = -1
    metrics, n_noise = evaluate_ignoring_noise(X, labels)
    assert n_noise == 3
    assert metrics is not None


def test_single_cluster_gives_no_metrics():
    X = blobs()
    labels = np.zeros(len(X), dtype=int)
    metrics, n_noise = evaluate_ignoring_noise(X, labels)
    assert metrics is None
    assert n_noise == 0


def test_dendrogram_file_is_saved(tmp_path):
    path = tmp_path / "dendro.png"
    save_dendrogram(blobs(), path=str(path))
    assert path.exists() and path.stat().st_size > 0