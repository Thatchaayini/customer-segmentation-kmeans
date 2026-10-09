import pandas as pd
import pytest

from segmentation import segment_customers, cluster_summary


def make_data():
    X = pd.DataFrame({
        "a": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
        "b": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
    })
    df = X.copy()
    df.insert(0, "CustomerID", range(1, len(X) + 1))
    return df, X


def test_adds_cluster_column():
    df, X = make_data()
    result, model, k = segment_customers(df, X, n_clusters=3)
    assert "Cluster" in result.columns


def test_correct_number_of_clusters():
    df, X = make_data()
    result, model, k = segment_customers(df, X, n_clusters=3)
    assert k == 3
    assert result["Cluster"].nunique() == 3


def test_every_customer_gets_a_cluster():
    df, X = make_data()
    result, model, k = segment_customers(df, X, n_clusters=3)
    assert len(result) == len(df)
    assert result["Cluster"].notna().all()


def test_empty_data_raises_error():
    empty = pd.DataFrame(columns=["a", "b"])
    with pytest.raises(ValueError):
        segment_customers(empty, empty, n_clusters=3)


def test_invalid_k_raises_error():
    df, X = make_data()
    with pytest.raises(ValueError):
        segment_customers(df, X, n_clusters=0)


def test_k_larger_than_rows_is_reduced():
    df, X = make_data()
    result, model, k = segment_customers(df, X, n_clusters=50)
    assert k <= len(X)


def test_identical_points_no_empty_clusters():
    X = pd.DataFrame({"a": [1.0] * 5, "b": [1.0] * 5})
    df = X.copy()
    df.insert(0, "CustomerID", range(1, 6))
    result, model, k = segment_customers(df, X, n_clusters=3)
    assert k == 1
    assert result["Cluster"].nunique() == 1


def test_cluster_summary_has_size_column():
    df, X = make_data()
    result, model, k = segment_customers(df, X, n_clusters=3)
    summary = cluster_summary(result)
    assert "Size" in summary.columns
    assert summary["Size"].sum() == len(df)