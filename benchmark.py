import time
import tracemalloc

import numpy as np
import pandas as pd

from evaluation import evaluate_clustering
from generate_data import generate_data
from preprocess import clean_data, encode_features, scale_features
from segmentation import segment_customers
from validate import validate_inputs
from visualization import reduce_pca

FEATURES = ["Age", "Gender", "AnnualIncome", "PurchaseFrequency", "AvgOrderValue"]
NUMERIC = ["Age", "AnnualIncome", "PurchaseFrequency", "AvgOrderValue"]
SIZES = [200, 1000, 5000, 10000]


def make_dataset(n_rows, seed=42):
    """Build a bigger dataset by resampling the base data with small random noise."""
    base = generate_data(seed)
    rng = np.random.default_rng(seed)
    df = base.sample(n=n_rows, replace=True, random_state=seed).reset_index(drop=True)
    for col in NUMERIC:
        df[col] = df[col] * rng.normal(1.0, 0.05, size=n_rows)
    df["CustomerID"] = range(1, n_rows + 1)
    return df


def timed(func, *args, **kwargs):
    start = time.perf_counter()
    output = func(*args, **kwargs)
    return output, time.perf_counter() - start


def run_workflow(df, n_clusters=3):
    """Run every stage of the workflow and return the time taken by each stage."""
    timings = {}

    df, timings["clean"] = timed(clean_data, df)
    df, timings["encode"] = timed(encode_features, df)
    (X_scaled, _), timings["scale"] = timed(scale_features, df, FEATURES)

    errors, timings["validate"] = timed(validate_inputs, df, X_scaled, n_clusters)
    if errors:
        raise ValueError(f"Validation failed: {errors}")

    (result, model, k), timings["cluster"] = timed(
        segment_customers, df, X_scaled, n_clusters
    )
    _, timings["evaluate"] = timed(
        evaluate_clustering, X_scaled, result["Cluster"].to_numpy()
    )
    _, timings["pca"] = timed(reduce_pca, X_scaled)

    timings["total"] = sum(timings.values())
    return timings


def peak_memory_mb(df):
    tracemalloc.start()
    run_workflow(df)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def benchmark(sizes=SIZES):
    run_workflow(make_dataset(200))  # warm-up run, not measured

    rows = []
    for n in sizes:
        df = make_dataset(n)
        timings = run_workflow(df)
        row = {"rows": n}
        row.update({f"{name}_s": round(sec, 3) for name, sec in timings.items()})
        row["rows_per_sec"] = round(n / timings["total"])
        row["peak_memory_mb"] = round(peak_memory_mb(df), 1)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = benchmark()
    print(results.to_string(index=False))
    results.to_csv("data/performance_results.csv", index=False)
    print("\nSaved: data/performance_results.csv")