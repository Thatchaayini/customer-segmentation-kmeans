import os

from generate_data import generate_data, save_data
from preprocess import preprocess
from validate import validate_inputs
from segmentation import segment_customers, cluster_summary

DATA_PATH = "data/customers.csv"
OUTPUT_PATH = "data/customers_segmented.csv"


def run_pipeline(n_clusters=3, data_path=DATA_PATH, output_path=OUTPUT_PATH):
    """Full workflow: data -> preprocess -> validate -> K-Means -> save."""
    if not os.path.exists(data_path):
        save_data(generate_data(), data_path)

    df, X_scaled, scaler = preprocess(data_path)

    errors = validate_inputs(df, X_scaled, n_clusters)
    if errors:
        raise ValueError(f"Validation failed: {errors}")

    result, model, k = segment_customers(df, X_scaled, n_clusters)
    result.to_csv(output_path, index=False)
    return result, cluster_summary(result), k


if __name__ == "__main__":
    result, summary, k = run_pipeline()
    print(f"Clusters used: {k}\n")
    print(summary.to_string())
    print(f"\nSegmented {len(result)} customers. Saved to {OUTPUT_PATH}")