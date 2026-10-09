import os
import numpy as np
import pandas as pd

# 3 customer groups: (count, age, income k$, frequency/yr, avg order value $)
GROUPS = [
    (70, (22, 32), (15, 40), (20, 50), (20, 60)),
    (70, (30, 50), (45, 80), (8, 20), (60, 120)),
    (60, (35, 60), (80, 140), (3, 10), (150, 400)),
]


def generate_data(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for count, age, income, freq, aov in GROUPS:
        for _ in range(count):
            rows.append({
                "Age": int(rng.integers(*age)),
                "Gender": rng.choice(["Male", "Female"]),
                "AnnualIncome": int(rng.integers(*income)),
                "PurchaseFrequency": int(rng.integers(*freq)),
                "AvgOrderValue": round(float(rng.uniform(*aov)), 2),
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "CustomerID", range(1, len(df) + 1))
    return df


def save_data(df, path="data/customers.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = generate_data()
    save_data(df)
    print(df.head())
    print(f"\nSaved {len(df)} customers to data/customers.csv")