# Production Inference Optimization Study

[![CI](https://github.com/PCSchmidt/inference-optimization-study/actions/workflows/ci.yml/badge.svg)](https://github.com/PCSchmidt/inference-optimization-study/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.24-purple)
![License](https://img.shields.io/badge/License-MIT-green)

Benchmarking and optimizing sentence-transformer inference for real-world embedding workloads.

## Motivation

The [SkillSwap](https://github.com/PCSchmidt/skillswapappmvp) platform uses `all-MiniLM-L6-v2` (384-dim, 22.7M parameter) embeddings to power semantic skill matching. In production, every skill search, new listing, and match recommendation triggers an embedding call — making inference cost the dominant recurring expense.

This study explores how far we can push CPU throughput **without GPU hardware, model retraining, or architecture changes** through:

- **ONNX export** — convert from PyTorch to ONNX Runtime for faster CPU inference
- **INT8 dynamic quantization** — reduce model size by ~74% and improve throughput
- **Adaptive batching** — amortize per-request overhead across dynamic batch windows

## Key Results

| Configuration | p50 Latency | p95 Latency | p99 Latency | Throughput | Model Size |
|---|---|---|---|---|---|
| PyTorch (single) | 12.1 ms | 19.8 ms | 34.7 ms | 68 req/s | ~88 MB |
| PyTorch (batch=32) | 14.2 ms | 19.0 ms | 21.6 ms | 195 req/s | ~88 MB |
| ONNX Runtime (single) | 8.5 ms | 36.1 ms | 52.1 ms | 151 req/s | ~88 MB |
| ONNX + INT8 (single) | 3.2 ms | 6.3 ms | 7.6 ms | 234 req/s | ~23 MB |
| **ONNX + INT8 (batch=32)** | **2.9 ms** | **7.0 ms** | **9.4 ms** | **602 req/s** | **~23 MB** |

**Highlights:**
- **~9× throughput improvement** (68 → 602 req/s) with the full ONNX + INT8 + batching pipeline
- **74% model size reduction** (88 MB → 23 MB) via INT8 dynamic quantization
- **ONNX output equivalence**: ~1.000 mean cosine similarity vs. PyTorch baseline
- **Quantized accuracy**: ~0.962 mean cosine similarity — well within tolerance for semantic search
- **94% cost reduction**: $0.21 → $0.02 per 1,000 embedding requests on a $0.05/hr CPU instance

> **Note:** Absolute numbers vary between runs and across hardware. The relative optimization ratios (throughput multipliers, size reduction percentages, accuracy preservation) are the meaningful takeaways.

*Benchmarked on CPU (Intel, Windows) · Python 3.13.2 · PyTorch 2.11.0 · ONNX Runtime 1.24.4 · sentence-transformers 5.4.1*

## Methodology

The optimization pipeline is progressive — each step builds on the previous:

```
PyTorch baseline    →    ONNX export    →    INT8 quantization    →    Adaptive batching
  (naive)              (faster runtime)     (smaller + faster)       (amortize overhead)
```

1. **Baseline**: Load `all-MiniLM-L6-v2` via `sentence-transformers` and measure single-request + batched throughput on CPU
2. **ONNX export**: Convert the PyTorch model to ONNX format using HuggingFace Optimum; validate numerical equivalence
3. **INT8 quantization**: Apply dynamic quantization (FP32 → INT8 weights) to the ONNX model; measure size reduction and accuracy impact
4. **Adaptive batching**: Sweep batch sizes (1–64) to find the throughput sweet spot
5. **Benchmark harness**: Run all 5 configurations through a controlled benchmark — 200 latency samples + 10s sustained throughput each

Accuracy is validated by computing cosine similarity between optimized and baseline embeddings across the full test corpus. Values above 0.95 are considered production-safe for semantic search and recommendation workloads.

## Quick Start

```bash
git clone https://github.com/PCSchmidt/inference-optimization-study.git
cd inference-optimization-study

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e .

# Run the notebook
jupyter lab notebooks/inference_optimization_study.ipynb
```

Run all cells top-to-bottom. The full benchmark suite takes ~5 minutes on a modern CPU. No GPU required.

See [DEMO_GUIDE.md](DEMO_GUIDE.md) for a guided walkthrough with expected outputs and talking points.

## Project Structure

```
├── notebooks/
│   └── inference_optimization_study.ipynb   # Main study notebook (29 cells)
├── src/
│   ├── baseline.py        # PyTorch SentenceTransformer baseline
│   ├── onnx_export.py     # ONNX export + validation
│   ├── quantize.py        # INT8 dynamic quantization
│   ├── batch_queue.py     # Adaptive batching simulation
│   └── benchmark.py       # Benchmark harness (latency, throughput, memory)
├── results/
│   ├── benchmark_results.csv   # Full benchmark data
│   ├── cost_analysis.csv       # Cost per 1K requests
│   ├── batch_throughput.csv    # Batch size sweep data
│   ├── latency_data.json       # Raw latency distributions
│   ├── onnx_model/             # Exported ONNX model (~88 MB)
│   └── onnx_model_int8/        # INT8 quantized model (~23 MB)
├── DEMO_GUIDE.md          # Guided demo walkthrough
├── pyproject.toml
└── README.md
```

## Reproducing Results

**Exact environment used:**
- Python 3.13.2 on Windows (Intel CPU)
- PyTorch 2.11.0+cpu
- ONNX Runtime 1.24.4
- sentence-transformers 5.4.1
- HuggingFace Optimum 1.25+

All dependencies are pinned in `pyproject.toml`. Run `pip install -e .` to install the exact package set. Results are saved to `results/` — inspect the CSVs directly without re-running the full benchmark.

Hardware variations will produce different absolute numbers, but the relative optimization ratios should hold across x86 CPU platforms.

## Technologies

PyTorch · Sentence Transformers · ONNX · ONNX Runtime · HuggingFace Optimum · Plotly · NumPy · Pandas
