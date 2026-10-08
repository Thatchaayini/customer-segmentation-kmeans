import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data():
    """Sample customer data (annual income in k$, spending score 1-100)."""
    return pd.DataFrame({
        "CustomerID": range(1, 13),
        "AnnualIncome": [15, 16, 17, 18, 55, 58, 60, 62, 100, 105, 110, 115],
        "SpendingScore": [80, 75, 20, 15, 50, 52, 48, 55, 90, 85, 10, 12],
    })


def build_model(n_clusters=3):
    return KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)


def main():
    df = load_data()
    features = df[["AnnualIncome", "SpendingScore"]]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = build_model(n_clusters=3)
    df["Cluster"] = model.fit_predict(scaled)

    print(df)
    print("\nCustomers per cluster:")
    print(df["Cluster"].value_counts())

    # Test with a new customer
    new_customer = pd.DataFrame({"AnnualIncome": [60], "SpendingScore": [50]})
    cluster = model.predict(scaler.transform(new_customer))[0]
    print(f"\nNew customer belongs to cluster: {cluster}")


if __name__ == "__main__":
    main()