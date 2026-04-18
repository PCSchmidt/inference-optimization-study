# Inference Optimization Study — Demo Guide

This guide walks through the study notebook for portfolio demos, recruiter walkthroughs, and technical interviews.

## Demo Objective

Demonstrate:

- A systematic, measured approach to optimizing ML inference on commodity CPU hardware
- Progressive optimization pipeline: PyTorch → ONNX → INT8 quantization → adaptive batching
- **17× throughput improvement** (48 → 816 req/s) with <4% embedding accuracy deviation
- Production cost analysis translating performance gains into dollar savings

## Recommended Demo Duration

- **Short version:** 3–5 minutes (skip to results, walk through charts and conclusions)
- **Full version:** 8–12 minutes (run notebook top-to-bottom, explain each optimization step)

## Prerequisites

- Python 3.12+ with `pip`
- ~500 MB disk space (model files)
- No GPU required — the entire study runs on CPU

```bash
git clone https://github.com/PCSchmidt/inference-optimization-study.git
cd inference-optimization-study
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
jupyter lab notebooks/inference_optimization_study.ipynb
```

## Demo Script

### Step 1: Context & Motivation (Cells 1–2)

Open the notebook and read the title and motivation sections.

**Key talking points:**
- This study was motivated by [SkillSwap](https://github.com/PCSchmidt/skillswapappmvp), a skill-matching platform that uses sentence-transformer embeddings for semantic search
- In production, embedding inference is often the largest recurring compute cost
- The approach (ONNX + quantization + batching) was chosen because it works on commodity CPU, requires no retraining, and applies to any PyTorch model

### Step 2: Environment & Corpus (Cells 3–6)

Run cells 3–6. These import libraries, verify the runtime, and define a test corpus of 20 realistic skill descriptions.

**Expected output:**
- Python version, PyTorch version, device = `cpu`
- Test corpus: 20 texts, Benchmark corpus: 1,000 texts

### Step 3: Baseline Measurement (Cells 7–9)

Run cells 7–9. This loads the `all-MiniLM-L6-v2` model and measures unoptimized performance.

**Expected output:**
- Model: 22.7M parameters, ~86 MB, 384-dim embeddings
- Baseline latency: ~1 ms/text (single request)

**Talking point:** This is the starting line. Everything after this is about making it faster without changing the model itself.

### Step 4: ONNX Export (Cells 10–11)

Run cells 10–11. The model is converted from PyTorch format to ONNX and validated for numerical equivalence.

**Expected output:**
- ONNX model exported successfully
- Max absolute diff: < 0.0001 (essentially zero)
- Validation passed: True

**Talking point:** This is a free optimization — no accuracy loss, just a format conversion that unlocks a faster runtime engine (ONNX Runtime).

### Step 5: INT8 Quantization (Cells 12–13)

Run cells 12–13. The ONNX model weights are compressed from 32-bit floats to 8-bit integers.

**Expected output:**
- Original: ~86 MB → Quantized: ~22 MB
- Size reduction: ~74.7%

**Talking point:** 4× smaller model that runs faster on CPU. The trade-off is small numerical differences — we validate this later.

### Step 6: Batch Size Sweep (Cells 14–16)

Run cells 14–16. This sweeps batch sizes from 1 to 64 and produces a bar chart.

**What to show:** The throughput bar chart — steep gains from batch=1 to batch=8, then diminishing returns. This shows where batching helps most.

### Step 7: Full Benchmark (Cells 17–19)

Run cells 17–19. This is the core experiment — 5 configurations benchmarked under identical conditions (200 latency samples, 10s sustained throughput each).

**Expected output table:**

| Configuration | p50 (ms) | p99 (ms) | Throughput (req/s) |
|---|---|---|---|
| PyTorch (single) | ~11 | ~23 | ~48 |
| PyTorch (batch=32) | ~15 | ~24 | ~352 |
| ONNX Runtime (single) | ~4 | ~6 | ~244 |
| ONNX + INT8 (single) | ~2 | ~5 | ~420 |
| **ONNX + INT8 (batch=32)** | **~3** | **~5** | **~816** |

**Talking point:** The full pipeline achieves 17× throughput over the naive baseline on the same CPU hardware.

### Step 8: Accuracy Validation (Cells 20–21)

Run cells 20–21. This verifies the optimized models still produce useful embeddings.

**Expected output:**
- ONNX vs PyTorch: cosine similarity ≈ 1.000 (lossless)
- Quantized vs PyTorch: cosine similarity ≈ 0.962 (excellent for production)

**Talking point:** The ONNX export is numerically identical to PyTorch. Quantization introduces a <4% deviation — well within tolerance for semantic search and recommendations.

### Step 9: Visualizations (Cells 22–24)

Run cells 22–24. Two interactive Plotly charts:
- **Latency box plots:** Full distributions showing not just medians but tail behavior (p95, p99)
- **Throughput bar chart:** The headline comparison across all 5 configurations

### Step 10: Cost Analysis (Cells 25–27)

Run cells 25–27. Translates throughput into estimated cost per 1,000 embedding requests.

**Expected output:** Cost drops from ~$0.29/1K requests (baseline) to ~$0.017/1K (optimized) — a **94% cost reduction**.

**Talking point:** This is the business case. On a $0.05/hr cloud instance, the optimized pipeline processes 17× more requests for the same dollar.

### Step 11: Conclusions (Cell 28)

Read the conclusions section. Summarizes the full optimization stack, recommendations, and connection back to SkillSwap.

### Step 12: Save Results (Cell 29)

Run cell 29 to persist all benchmark data to CSV files in `results/`.

## Interviewer Q&A Prompts

### Why not just use a GPU?

GPU inference is faster, but a GPU instance costs 10× more per hour. This study shows you can get 17× throughput improvement on a $0.05/hr CPU instance — far more cost-effective for startups and small teams. GPU scaling makes sense only when sustained load exceeds what a single optimized CPU instance can handle (~800 req/s).

### Why ONNX instead of TensorRT or other frameworks?

ONNX is framework-agnostic and runs on any hardware (CPU, GPU, ARM). TensorRT is NVIDIA GPU-only. ONNX Runtime is also maintained by Microsoft and has excellent Python integration via HuggingFace Optimum. For a CPU-first deployment, ONNX is the natural choice.

### How do you know quantization doesn't hurt quality?

We measure it directly. Cell 21 computes cosine similarity between PyTorch and quantized embeddings across the full test corpus. The mean similarity is 0.962 — this means the quantized embeddings point in almost the same direction as the originals. For semantic search and matching (where ranking order matters more than exact distances), this is well within acceptable tolerance.

### What would you do differently at larger scale?

Three things: (1) Add GPU inference with TensorRT for workloads above ~1K req/s, (2) Use a dedicated vector database (Pinecone, Qdrant) instead of in-process embedding, (3) Implement model distillation to train a smaller task-specific model rather than quantizing a general-purpose one.

### How does this connect to your other work?

The SkillSwap platform uses the exact same embedding model (`all-MiniLM-L6-v2`) for semantic skill matching. This study benchmarks the optimization pipeline that would be deployed in SkillSwap's FastAPI backend to handle production-scale embedding requests.

## Key Files

| File | Purpose |
|---|---|
| `notebooks/inference_optimization_study.ipynb` | Main study notebook (run this) |
| `src/baseline.py` | PyTorch baseline inference |
| `src/onnx_export.py` | ONNX export + validation |
| `src/quantize.py` | INT8 dynamic quantization |
| `src/batch_queue.py` | Adaptive batching simulation |
| `src/benchmark.py` | Benchmark harness (latency, throughput, memory) |
| `results/benchmark_results.csv` | Raw benchmark data |
| `results/cost_analysis.csv` | Cost analysis data |
