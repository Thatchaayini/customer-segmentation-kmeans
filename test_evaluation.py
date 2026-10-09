import pandas as pd
import pytest

from evaluation import evaluate_clustering, evaluate_k_range


def make_data():
    return pd.DataFrame({
        "a": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
        "b": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
    })


def test_metrics_returned_for_good_clustering():
    X = make_data()
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    metrics = evaluate_clustering(X, labels)
    assert set(metrics) == {"silhouette", "calinski_harabasz", "davies_bouldin"}
    assert metrics["silhouette"] > 0.9


def test_single_cluster_raises_error():
    X = make_data()
    with pytest.raises(ValueError):
        evaluate_clustering(X, [0] * 9)


def test_k_range_returns_one_row_per_k():
    X = make_data()
    results = evaluate_k_range(X, k_values=range(2, 5))
    assert list(results.index) == [2, 3, 4]
    assert "inertia" in results.columns


def test_best_k_is_three_for_three_blobs():
    X = make_data()
    results = evaluate_k_range(X, k_values=range(2, 6))
    assert results["silhouette"].idxmax() == 3