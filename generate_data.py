import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# 3 customer groups: (count, age, income k$, frequency/yr, avg order value $)
groups = [
    (70, (22, 32), (15, 40), (20, 50), (20, 60)),
    (70, (30, 50), (45, 80), (8, 20), (60, 120)),
    (60, (35, 60), (80, 140), (3, 10), (150, 400)),
]

rows = []
for count, age, income, freq, aov in groups:
    for _ in range(count):
        rows.append({
            "Age": int(rng.integers(*age)),
            "Gender": rng.choice(["Male", "Female"]),
            "AnnualIncome": int(rng.integers(*income)),
            "PurchaseFrequency": int(rng.integers(*freq)),
            "AvgOrderValue": round(float(rng.uniform(*aov)), 2),
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "CustomerID", range(1, len(df) + 1))

os.makedirs("data", exist_ok=True)
df.to_csv("data/customers.csv", index=False)
print(df.head())
print(f"\nSaved {len(df)} customers to data/customers.csv")