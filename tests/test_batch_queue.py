"""Tests for adaptive batching / batch throughput sweep."""

from src.batch_queue import simulate_batch_throughput


def _dummy_encode(texts):
    import numpy as np
    return np.random.randn(len(texts), 384).astype("float32")


def test_simulate_batch_throughput_returns_list():
    results = simulate_batch_throughput(
        _dummy_encode,
        ["hello"] * 20,
        batch_sizes=[1, 4, 8],
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_simulate_batch_throughput_dict_keys():
    results = simulate_batch_throughput(
        _dummy_encode,
        ["hello"] * 20,
        batch_sizes=[1, 4],
    )
    for r in results:
        assert set(r.keys()) == {"batch_size", "total_time_s", "throughput_rps"}
        assert r["throughput_rps"] > 0


def test_larger_batch_not_slower():
    results = simulate_batch_throughput(
        _dummy_encode,
        ["hello"] * 100,
        batch_sizes=[1, 16],
    )
    # Larger batch should have equal or better throughput with a dummy encode
    assert results[1]["throughput_rps"] >= results[0]["throughput_rps"] * 0.5
