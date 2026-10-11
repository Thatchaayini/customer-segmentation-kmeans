import itertools
import time

import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from config import CACHE_DIR, RANDOM_STATE
from preprocess import preprocess
from tuning import PARAM_GRID, tune_kmeans

memory = Memory(CACHE_DIR, verbose=0)


@memory.cache
def fit_and_score(X_scaled, params, random_state=RANDOM_STATE):
    """Fit one parameter combination. The result is cached on disk."""
    model = KMeans(random_state=random_state, **params)
    labels = model.fit_predict(X_scaled)
    if len(set(labels)) < 2:
        return None  # metrics need at least 2 clusters
    return {
        **params,
        "silhouette": round(silhouette_score(X_scaled, labels), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X_scaled, labels), 2),
        "davies_bouldin": round(davies_bouldin_score(X_scaled, labels), 4),
        "inertia": round(model.inertia_, 2),
    }


def tune_kmeans_fast(X_scaled, param_grid=None, n_jobs=-1, random_state=RANDOM_STATE):
    """Same as tune_kmeans, but runs combinations in parallel and caches them."""
    grid = param_grid or PARAM_GRID
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    combos = [p for p in combos if p["n_clusters"] < len(X_scaled)]

    rows = Parallel(n_jobs=n_jobs)(
        delayed(fit_and_score)(X_scaled, p, random_state) for p in combos
    )
    rows = [r for r in rows if r is not None]
    if not rows:
        raise ValueError("No valid parameter combination found")

    results = pd.DataFrame(rows)
    return results.sort_values("silhouette", ascending=False).reset_index(drop=True)


def timed(label, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    print(f"{label}: {time.perf_counter() - start:.2f}s")
    return result


if __name__ == "__main__":
    df, X_scaled, scaler = preprocess()
    memory.clear(warn=False)  # start with an empty cache for fair timing

    timed("Serial (tuning.py)       ", tune_kmeans, X_scaled)
    timed("Parallel, cold cache     ", tune_kmeans_fast, X_scaled)
    timed("Parallel, warm cache     ", tune_kmeans_fast, X_scaled)