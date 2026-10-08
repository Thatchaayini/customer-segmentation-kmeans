import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path="data/customers.csv"):
    return pd.read_csv(path)


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df.reset_index(drop=True)


def encode_features(df):
    # Gender (Male/Female) -> numbers (0/1)
    df = df.copy()
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    return df


def scale_features(df, columns):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])
    return pd.DataFrame(scaled, columns=columns), scaler


def preprocess(path="data/customers.csv"):
    df = load_data(path)
    df = clean_data(df)
    df = encode_features(df)

    feature_cols = ["Age", "Gender", "AnnualIncome", "PurchaseFrequency", "AvgOrderValue"]
    X_scaled, scaler = scale_features(df, feature_cols)
    return df, X_scaled, scaler


if __name__ == "__main__":
    df, X_scaled, scaler = preprocess()
    print("Cleaned data shape:", df.shape)
    print("\nScaled data (first 5 rows):")
    print(X_scaled.head())
    print("\nMean (should be ~0):")
    print(X_scaled.mean().round(2))