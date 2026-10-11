import pytest

from benchmark import benchmark, make_dataset, run_workflow

STAGES = {
    "clean", "encode", "scale", "validate",
    "cluster", "evaluate", "pca", "total",
}
NUMERIC = ["Age", "AnnualIncome", "PurchaseFrequency", "AvgOrderValue"]


def test_make_dataset_has_requested_rows():
    df = make_dataset(300)
    assert len(df) == 300
    assert df["CustomerID"].is_unique


def test_make_dataset_has_no_negative_values():
    df = make_dataset(300)
    assert (df[NUMERIC] >= 0).all().all()


def test_workflow_reports_every_stage():
    timings = run_workflow(make_dataset(300))
    assert set(timings) == STAGES
    assert all(t >= 0 for t in timings.values())


def test_total_equals_sum_of_stages():
    timings = run_workflow(make_dataset(300))
    parts = sum(v for k, v in timings.items() if k != "total")
    assert timings["total"] == pytest.approx(parts)


def test_workflow_is_fast_enough_for_1000_rows():
    timings = run_workflow(make_dataset(1000))
    assert timings["total"] < 5  # seconds (measured: about 0.1 s)


def test_benchmark_returns_one_row_per_size():
    results = benchmark(sizes=[200, 400])
    assert results["rows"].tolist() == [200, 400]
    assert (results["peak_memory_mb"] > 0).all()
    assert (results["rows_per_sec"] > 0).all()