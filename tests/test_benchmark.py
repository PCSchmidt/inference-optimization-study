"""Tests for the benchmark harness."""

import pandas as pd

from src.benchmark import measure_latency, run_benchmark, results_to_dataframe, BenchmarkResult


def _dummy_encode(texts):
    """Fast no-op encode for testing harness logic."""
    import numpy as np
    return np.random.randn(len(texts), 384).astype("float32")


def test_measure_latency_returns_floats():
    latencies = measure_latency(_dummy_encode, ["hello"] * 5, n_repeats=10, warmup=2)
    assert len(latencies) == 10
    assert all(isinstance(v, float) for v in latencies)
    assert all(v > 0 for v in latencies)


def test_run_benchmark_returns_result():
    result = run_benchmark(
        "test",
        _dummy_encode,
        ["hello"] * 10,
        n_latency_samples=10,
        throughput_duration_s=0.5,
        batch_size=4,
    )
    assert isinstance(result, BenchmarkResult)
    assert result.name == "test"
    assert result.p50_ms > 0
    assert result.throughput_rps > 0


def test_results_to_dataframe():
    result = BenchmarkResult(
        name="test",
        num_requests=10,
        latencies_ms=[1.0] * 10,
        p50_ms=1.0,
        p95_ms=1.0,
        p99_ms=1.0,
        throughput_rps=100.0,
        peak_memory_mb=0.0,
    )
    df = results_to_dataframe([result])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    expected_cols = {"Configuration", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Throughput (req/s)", "Peak Memory (MB)"}
    assert set(df.columns) == expected_cols
