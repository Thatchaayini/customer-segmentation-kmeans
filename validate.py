import numpy as np
from preprocess import preprocess

REQUIRED_COLUMNS = ["Age", "Gender", "AnnualIncome", "PurchaseFrequency", "AvgOrderValue"]


def validate_inputs(df, X_scaled, n_clusters=3):
    errors = []

    # 1. Required columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # 2. No missing values
    if X_scaled.isnull().any().any():
        errors.append("Scaled data has missing values")

    # 3. All values numeric
    if not all(np.issubdtype(t, np.number) for t in X_scaled.dtypes):
        errors.append("Non-numeric columns found")

    # 4. No infinite values
    if np.isinf(X_scaled.to_numpy()).any():
        errors.append("Infinite values found")

    # 5. No negative values in raw data
    if (df[["Age", "AnnualIncome", "PurchaseFrequency", "AvgOrderValue"]] < 0).any().any():
        errors.append("Negative values found in raw data")

    # 6. Enough samples for K clusters
    if len(X_scaled) < n_clusters:
        errors.append(f"Need at least {n_clusters} rows, got {len(X_scaled)}")

    # 7. Scaling check (mean ~0, std ~1)
    if not np.allclose(X_scaled.mean(), 0, atol=1e-6):
        errors.append("Scaled data mean is not ~0")
    if not np.allclose(X_scaled.std(ddof=0), 1, atol=1e-6):
        errors.append("Scaled data std is not ~1")

    return errors


if __name__ == "__main__":
    df, X_scaled, scaler = preprocess()
    errors = validate_inputs(df, X_scaled, n_clusters=3)

    if errors:
        print("Validation FAILED:")
        for e in errors:
            print(" -", e)
    else:
        print("Validation PASSED: data is ready for K-Means")
        print(f"Rows: {len(X_scaled)}, Features: {X_scaled.shape[1]}")