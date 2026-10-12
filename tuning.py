import itertools

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from config import RANDOM_STATE
from preprocess import preprocess

PARAM_GRID = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8],
    "init": ["k-means++", "random"],
    "n_init": [5, 10, 20],
    "max_iter": [100, 300],
}


def tune_kmeans(X_scaled, param_grid=None, random_state=RANDOM_STATE):
    """Try every parameter combination and rank them by silhouette score."""
    grid = param_grid or PARAM_GRID
    keys = list(grid)
    rows = []

    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        if params["n_clusters"] >= len(X_scaled):
            continue  # K must be smaller than the number of samples

        model = KMeans(random_state=random_state, **params)
        labels = model.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue  # metrics need at least 2 clusters

        rows.append({
            **params,
            "silhouette": round(silhouette_score(X_scaled, labels), 4),
            "calinski_harabasz": round(calinski_harabasz_score(X_scaled, labels), 2),
            "davies_bouldin": round(davies_bouldin_score(X_scaled, labels), 4),
            "inertia": round(model.inertia_, 2),
        })

    if not rows:
        raise ValueError("No valid parameter combination found")

    results = pd.DataFrame(rows)
    return results.sort_values("silhouette", ascending=False).reset_index(drop=True)


def best_params(results):
    """Return the top-ranked parameters as a dict."""
    best = results.iloc[0]
    return {
        "n_clusters": int(best["n_clusters"]),
        "init": str(best["init"]),
        "n_init": int(best["n_init"]),
        "max_iter": int(best["max_iter"]),
    }


if __name__ == "__main__":
    df, X_scaled, scaler = preprocess()
    results = tune_kmeans(X_scaled)

    print(f"Tried {len(results)} combinations\n")
    print("Top 10 by silhouette score:")
    print(results.head(10).to_string())
    print("\nBest parameters:", best_params(results))

    results.to_csv("data/tuning_results.csv", index=False)
    print("\nSaved: data/tuning_results.csv")