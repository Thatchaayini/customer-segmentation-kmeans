import pandas as pd
from sklearn.datasets import make_blobs

from fast_tuning import tune_kmeans_fast
from tuning import tune_kmeans

GRID = {"n_clusters": [2, 3, 4], "init": ["k-means++"], "n_init": [5], "max_iter": [100]}


def blobs():
    X, _ = make_blobs(n_samples=90, centers=3, cluster_std=0.5, random_state=0)
    return pd.DataFrame(X, columns=["a", "b"])


def test_fast_matches_serial():
    X = blobs()
    fast = tune_kmeans_fast(X, GRID, n_jobs=1)
    slow = tune_kmeans(X, GRID)
    assert len(fast) == len(slow)
    assert fast["silhouette"].max() == slow["silhouette"].max()


def test_cached_call_returns_same_result():
    X = blobs()
    first = tune_kmeans_fast(X, GRID, n_jobs=1)
    second = tune_kmeans_fast(X, GRID, n_jobs=1)
    assert first.equals(second)