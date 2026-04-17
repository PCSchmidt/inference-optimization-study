# Production Inference Optimization Study

Benchmarking and optimizing sentence-transformer inference for real-world embedding workloads.

## Motivation

The [SkillSwap](https://github.com/PCSchmidt/skillswapappmvp) platform uses `all-MiniLM-L6-v2` (384-dim) embeddings to power skill matching. This study explores how to reduce inference cost and latency in production through:

- **ONNX export** — convert from PyTorch to ONNX Runtime for faster CPU inference
- **INT8 dynamic quantization** — reduce model size and improve throughput
- **Adaptive batching** — amortize per-request overhead across dynamic batch windows

## Key Results

| Configuration | p50 Latency | Throughput | Model Size | Accuracy Delta |
|---|---|---|---|---|
| PyTorch baseline | — | — | — | — |
| ONNX Runtime | — | — | — | — |
| ONNX + INT8 quant | — | — | — | — |
| ONNX + INT8 + batching | — | — | — | — |

*Results will be populated after running the benchmark notebook.*

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run the notebook
jupyter lab notebooks/inference_optimization_study.ipynb
```

## Project Structure

```
├── notebooks/
│   └── inference_optimization_study.ipynb   # Main study notebook
├── src/
│   ├── baseline.py        # PyTorch SentenceTransformer baseline
│   ├── onnx_export.py     # ONNX export + validation
│   ├── quantize.py        # INT8 dynamic quantization
│   ├── batch_queue.py     # Adaptive batching with request queue
│   └── benchmark.py       # Benchmark harness
├── results/               # Benchmark output CSVs + charts
├── pyproject.toml
└── README.md
```

## Technologies

PyTorch · Sentence Transformers · ONNX · ONNX Runtime · Plotly · NumPy · Pandas
