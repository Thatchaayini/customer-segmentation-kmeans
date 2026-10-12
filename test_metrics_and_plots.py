import numpy as np
import pandas as pd
import pytest

from evaluation import evaluate_clustering, evaluate_k_range
from visualization import plot_clusters, reduce_pca, visualize_segments


def blobs():
    return pd.DataFrame({
        "a": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
        "b": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
    })


def random_data():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(40, 5)), columns=list("abcde"))


# ---------- evaluation metrics ----------

def test_perfect_clustering_scores():
    metrics = evaluate_clustering(blobs(), [0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert metrics["silhouette"] > 0.9
    assert metrics["davies_bouldin"] < 0.1


def test_bad_labels_score_worse_than_good_labels():
    good = evaluate_clustering(blobs(), [0, 0, 0, 1, 1, 1, 2, 2, 2])
    bad = evaluate_clustering(blobs(), [0, 1, 2, 0, 1, 2, 0, 1, 2])
    assert bad["silhouette"] < good["silhouette"]
    assert bad["davies_bouldin"] > good["davies_bouldin"]


def test_all_unique_labels_raises_error():
    with pytest.raises(ValueError):
        evaluate_clustering(blobs(), list(range(9)))


def test_k_range_has_all_metric_columns():
    results = evaluate_k_range(blobs(), k_values=range(2, 4))
    assert set(results.columns) == {
        "silhouette", "calinski_harabasz", "davies_bouldin", "inertia",
    }


def test_inertia_decreases_as_k_grows():
    results = evaluate_k_range(blobs(), k_values=range(2, 6))
    assert results["inertia"].is_monotonic_decreasing


# ---------- visualization ----------

def test_pca_variance_ratios_are_sorted():
    _, variance = reduce_pca(random_data())
    assert variance[0] >= variance[1]


def test_pca_is_reproducible():
    c1, _ = reduce_pca(random_data())
    c2, _ = reduce_pca(random_data())
    assert np.allclose(c1, c2)


def test_pca_supports_three_components():
    coords, variance = reduce_pca(random_data(), n_components=3)
    assert coords.shape == (40, 3)
    assert len(variance) == 3


def test_saved_plot_is_not_empty(tmp_path):
    coords, _ = reduce_pca(random_data())
    path = tmp_path / "plot.png"
    plot_clusters(coords, np.array([0, 1] * 20), "Test", "x", "y", str(path))
    assert path.stat().st_size > 0


def test_visualize_segments_creates_both_images(tmp_path):
    variance = visualize_segments(n_clusters=3, out_dir=str(tmp_path))
    assert (tmp_path / "pca_clusters.png").exists()
    assert (tmp_path / "tsne_clusters.png").exists()
    assert 0 < variance.sum() <= 1