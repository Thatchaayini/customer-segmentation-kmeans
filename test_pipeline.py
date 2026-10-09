import os
import pandas as pd

from generate_data import generate_data, save_data
from main import run_pipeline


def test_full_pipeline(tmp_path):
    data_path = str(tmp_path / "customers.csv")
    output_path = str(tmp_path / "segmented.csv")
    save_data(generate_data(), data_path)

    result, summary, k = run_pipeline(3, data_path, output_path)

    assert k == 3
    assert len(result) == 200
    assert "Cluster" in result.columns
    assert summary["Size"].sum() == 200
    assert os.path.exists(output_path)
    assert len(pd.read_csv(output_path)) == 200


def test_pipeline_generates_data_if_missing(tmp_path):
    data_path = str(tmp_path / "new_customers.csv")
    output_path = str(tmp_path / "segmented.csv")

    result, summary, k = run_pipeline(3, data_path, output_path)

    assert os.path.exists(data_path)
    assert len(result) == 200


def test_pipeline_is_reproducible(tmp_path):
    data_path = str(tmp_path / "customers.csv")
    save_data(generate_data(), data_path)

    r1, _, _ = run_pipeline(3, data_path, str(tmp_path / "a.csv"))
    r2, _, _ = run_pipeline(3, data_path, str(tmp_path / "b.csv"))

    assert r1["Cluster"].tolist() == r2["Cluster"].tolist()