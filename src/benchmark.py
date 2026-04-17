"""Benchmark harness — latency percentiles, throughput, memory footprint."""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

import numpy as np
import psutil


@dataclass
class BenchmarkResult:
    name: str
    num_requests: int
    latencies_ms: list[float]
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    peak_memory_mb: float


def measure_latency(
    encode_fn,
    texts: list[str],
    n_repeats: int = 100,
    warmup: int = 10,
) -> list[float]:
    """Measure per-request latency over n_repeats single-text calls.

    Returns list of latencies in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        encode_fn(texts[:1])

    latencies = []
    for i in range(n_repeats):
        text = texts[i % len(texts)]
        start = time.perf_counter()
        encode_fn([text])
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
    return latencies


def measure_throughput(
    encode_fn,
    texts: list[str],
    batch_size: int = 32,
    duration_s: float = 10.0,
) -> tuple[float, int]:
    """Measure sustained throughput over a time window.

    Returns (requests_per_second, total_requests).
    """
    total = 0
    start = time.perf_counter()
    while (time.perf_counter() - start) < duration_s:
        batch = texts[total % len(texts) : total % len(texts) + batch_size]
        if not batch:
            batch = texts[:batch_size]
        encode_fn(batch)
        total += len(batch)
    elapsed = time.perf_counter() - start
    return total / elapsed, total


def measure_memory(encode_fn, texts: list[str]) -> float:
    """Estimate peak memory usage in MB during inference."""
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    encode_fn(texts)
    mem_after = process.memory_info().rss
    return (mem_after - mem_before) / (1024 * 1024)


def run_benchmark(
    name: str,
    encode_fn,
    texts: list[str],
    n_latency_samples: int = 100,
    throughput_duration_s: float = 10.0,
    batch_size: int = 32,
) -> BenchmarkResult:
    """Run a full benchmark suite for a given encode function."""
    latencies = measure_latency(encode_fn, texts, n_repeats=n_latency_samples)
    rps, total = measure_throughput(
        encode_fn, texts, batch_size=batch_size, duration_s=throughput_duration_s
    )
    peak_mem = measure_memory(encode_fn, texts[:batch_size])

    arr = np.array(latencies)
    return BenchmarkResult(
        name=name,
        num_requests=n_latency_samples,
        latencies_ms=latencies,
        p50_ms=round(float(np.percentile(arr, 50)), 3),
        p95_ms=round(float(np.percentile(arr, 95)), 3),
        p99_ms=round(float(np.percentile(arr, 99)), 3),
        throughput_rps=round(rps, 1),
        peak_memory_mb=round(peak_mem, 2),
    )


def results_to_dataframe(results: list[BenchmarkResult]):
    """Convert a list of BenchmarkResults to a pandas DataFrame."""
    import pandas as pd

    rows = []
    for r in results:
        rows.append({
            "Configuration": r.name,
            "p50 (ms)": r.p50_ms,
            "p95 (ms)": r.p95_ms,
            "p99 (ms)": r.p99_ms,
            "Throughput (req/s)": r.throughput_rps,
            "Peak Memory (MB)": r.peak_memory_mb,
        })
    return pd.DataFrame(rows)
