import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, FEATURE_COLUMNS, GENDER_MAP


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df.reset_index(drop=True)


def encode_features(df):
    # Gender (Male/Female) -> numbers (0/1)
    df = df.copy()
    df["Gender"] = df["Gender"].map(GENDER_MAP)
    return df


def scale_features(df, columns):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])
    return pd.DataFrame(scaled, columns=columns), scaler


def preprocess(path=DATA_PATH):
    df = load_data(path)
    df = clean_data(df)
    df = encode_features(df)

    X_scaled, scaler = scale_features(df, FEATURE_COLUMNS)
    return df, X_scaled, scaler


if __name__ == "__main__":
    df, X_scaled, scaler = preprocess()
    print("Cleaned data shape:", df.shape)
    print("\nScaled data (first 5 rows):")
    print(X_scaled.head())
    print("\nMean (should be ~0):")
    print(X_scaled.mean().round(2))