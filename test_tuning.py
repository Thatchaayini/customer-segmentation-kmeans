import pandas as pd
import pytest

from tuning import best_params, tune_kmeans

SMALL_GRID = {
    "n_clusters": [2, 3, 4],
    "init": ["k-means++"],
    "n_init": [5],
    "max_iter": [100],
}


def make_data():
    return pd.DataFrame({
        "a": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
        "b": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
    })


def test_returns_one_row_per_combination():
    results = tune_kmeans(make_data(), SMALL_GRID)
    assert len(results) == 3
    assert "silhouette" in results.columns


def test_results_sorted_by_silhouette():
    results = tune_kmeans(make_data(), SMALL_GRID)
    scores = results["silhouette"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_best_k_is_three_for_three_blobs():
    results = tune_kmeans(make_data(), SMALL_GRID)
    assert best_params(results)["n_clusters"] == 3


def test_best_params_has_all_keys():
    results = tune_kmeans(make_data(), SMALL_GRID)
    assert set(best_params(results)) == {"n_clusters", "init", "n_init", "max_iter"}


def test_k_not_smaller_than_samples_is_skipped():
    grid = {**SMALL_GRID, "n_clusters": [2, 50]}
    results = tune_kmeans(make_data(), grid)
    assert results["n_clusters"].tolist() == [2]


def test_no_valid_combination_raises_error():
    grid = {**SMALL_GRID, "n_clusters": [50]}
    with pytest.raises(ValueError):
        tune_kmeans(make_data(), grid)