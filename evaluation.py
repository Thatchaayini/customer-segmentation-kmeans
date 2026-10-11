import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from config import RANDOM_STATE
from preprocess import preprocess
from segmentation import build_model


def evaluate_clustering(X_scaled, labels, sample_size=None):
    """Return quality metrics for one clustering result.

    sample_size: compute the silhouette score on a random sample of this many
    rows. It is much faster and uses less memory on large datasets.
    """
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels > len(X_scaled) - 1:
        raise ValueError(
            "Metrics need at least 2 clusters and fewer clusters than samples"
        )

    silhouette = silhouette_score(
        X_scaled, labels, sample_size=sample_size, random_state=RANDOM_STATE
    )
    return {
        "silhouette": round(silhouette, 4),
        "calinski_harabasz": round(calinski_harabasz_score(X_scaled, labels), 2),
        "davies_bouldin": round(davies_bouldin_score(X_scaled, labels), 4),
    }


def evaluate_k_range(X_scaled, k_values=range(2, 9), sample_size=None):
    """Run K-Means for each K and compare the metrics."""
    rows = []
    for k in k_values:
        model = build_model(k)
        labels = model.fit_predict(X_scaled)
        metrics = evaluate_clustering(X_scaled, labels, sample_size)
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