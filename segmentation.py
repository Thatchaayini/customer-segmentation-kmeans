import numpy as np
from sklearn.cluster import KMeans

from config import (
    DEFAULT_K,
    INIT_METHOD,
    N_INIT,
    RANDOM_STATE,
    SEGMENTED_PATH,
)
from preprocess import preprocess
from validate import validate_inputs


def build_model(n_clusters, random_state=RANDOM_STATE):
    """Create a K-Means model with the project's default settings."""
    return KMeans(
        n_clusters=n_clusters,
        init=INIT_METHOD,
        n_init=N_INIT,
        random_state=random_state,
    )


def _check_inputs(X_scaled, n_clusters):
    """Raise a clear error for inputs K-Means cannot handle."""
    if len(X_scaled) == 0:
        raise ValueError("No data to cluster")
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")


def _max_possible_k(X_scaled, n_clusters):
    """K cannot be larger than the number of rows or distinct points."""
    n_unique = len(np.unique(X_scaled.to_numpy(), axis=0))
    return min(n_clusters, len(X_scaled), n_unique)


def _fit_without_empty_clusters(X_scaled, k):
    """Fit K-Means. If a cluster ends up empty, reduce K and retry."""
    while k >= 1:
        model = build_model(k)
        labels = model.fit_predict(X_scaled)
        if len(set(labels)) == k:
            return model, labels, k
        k -= 1
    raise RuntimeError("Could not fit K-Means")


def segment_customers(df, X_scaled, n_clusters=DEFAULT_K):
    """Run K-Means. Returns (df with Cluster column, fitted model, K used)."""
    _check_inputs(X_scaled, n_clusters)
    k = _max_possible_k(X_scaled, n_clusters)
    model, labels, k = _fit_without_empty_clusters(X_scaled, k)

    result = df.copy()
    result["Cluster"] = labels
    return result, model, k


def cluster_summary(result):
    """Average of each feature per cluster, plus cluster size."""
    summary = result.drop(columns=["CustomerID"]).groupby("Cluster").mean().round(2)
    summary["Size"] = result["Cluster"].value_counts().sort_index()
    return summary


if __name__ == "__main__":
    df, X_scaled, scaler = preprocess()

    errors = validate_inputs(df, X_scaled, n_clusters=DEFAULT_K)
    if errors:
        raise SystemExit(f"Validation failed: {errors}")

    result, model, k = segment_customers(df, X_scaled, DEFAULT_K)
    print(f"Clusters used: {k}\n")
    print(cluster_summary(result).to_string())

    result.to_csv(SEGMENTED_PATH, index=False)
    print(f"\nSaved to {SEGMENTED_PATH}")