import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from preprocess import preprocess


def evaluate_clustering(X_scaled, labels):
    """Return quality metrics for one clustering result."""
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels > len(X_scaled) - 1:
        raise ValueError("Metrics need at least 2 clusters and fewer clusters than samples")

    return {
        "silhouette": round(silhouette_score(X_scaled, labels), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X_scaled, labels), 2),
        "davies_bouldin": round(davies_bouldin_score(X_scaled, labels), 4),
    }


def evaluate_k_range(X_scaled, k_values=range(2, 9)):
    """Run K-Means for each K and compare the metrics."""
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
        metrics = evaluate_clustering(X_scaled, labels)
        metrics["K"] = k
        metrics["inertia"] = round(model.inertia_, 2)
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("K")


if __name__ == "__main__":
    df, X_scaled, scaler = preprocess()
    results = evaluate_k_range(X_scaled)
    print(results.to_string())
    print(f"\nBest K by silhouette: {results['silhouette'].idxmax()}")
    print(f"Best K by Calinski-Harabasz: {results['calinski_harabasz'].idxmax()}")