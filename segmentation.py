import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from preprocess import preprocess
from validate import validate_inputs


def segment_customers(df, X_scaled, n_clusters=3):
    """Run K-Means and return (df with Cluster column, model, final_k)."""
    # Edge case 1: empty data
    if len(X_scaled) == 0:
        raise ValueError("No data to cluster")

    # Edge case 2: invalid K
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")

    # Edge case 3: K more than rows or distinct points
    n_unique = len(np.unique(X_scaled.to_numpy(), axis=0))
    k = min(n_clusters, len(X_scaled), n_unique)

    # Edge case 4: empty clusters -> reduce K and retry
    while k >= 1:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
        if len(set(labels)) == k:
            break
        k -= 1

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

    errors = validate_inputs(df, X_scaled, n_clusters=3)
    if errors:
        raise SystemExit(f"Validation failed: {errors}")

    result, model, k = segment_customers(df, X_scaled, n_clusters=3)
    print(f"Clusters used: {k}\n")
    print(cluster_summary(result).to_string())

    result.to_csv("data/customers_segmented.csv", index=False)
    print("\nSaved to data/customers_segmented.csv")