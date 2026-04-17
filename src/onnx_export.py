"""Export SentenceTransformer to ONNX and validate output equivalence."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .baseline import MODEL_NAME

DEFAULT_ONNX_DIR = Path("results") / "onnx_model"


def export_to_onnx(output_dir: Path = DEFAULT_ONNX_DIR) -> Path:
    """Export the model to ONNX format using Hugging Face Optimum."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        f"sentence-transformers/{MODEL_NAME}", export=True
    )
    ort_model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(f"sentence-transformers/{MODEL_NAME}").save_pretrained(
        output_dir
    )
    return output_dir


def load_onnx_model(model_dir: Path = DEFAULT_ONNX_DIR) -> ORTModelForFeatureExtraction:
    """Load an exported ONNX model."""
    return ORTModelForFeatureExtraction.from_pretrained(model_dir)


def validate_onnx(
    pytorch_model: SentenceTransformer,
    onnx_dir: Path = DEFAULT_ONNX_DIR,
    test_texts: list[str] | None = None,
    atol: float = 1e-4,
) -> dict:
    """Compare ONNX outputs against PyTorch baseline.

    Returns dict with max_abs_diff, mean_abs_diff, and passed flag.
    """
    if test_texts is None:
        test_texts = [
            "machine learning optimization",
            "natural language processing",
            "web development with React",
        ]

    # PyTorch embeddings
    pt_emb = pytorch_model.encode(test_texts, normalize_embeddings=True)

    # ONNX embeddings (via Optimum pipeline)
    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
    ort_model = load_onnx_model(onnx_dir)
    inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
    ort_out = ort_model(**inputs).last_hidden_state

    # Mean pooling + normalize (same as SentenceTransformer default)
    attention_mask = inputs["attention_mask"].unsqueeze(-1)
    onnx_emb = (ort_out * attention_mask).sum(1) / attention_mask.sum(1)
    onnx_emb = onnx_emb.detach().numpy()
    onnx_emb = onnx_emb / np.linalg.norm(onnx_emb, axis=1, keepdims=True)

    max_diff = float(np.max(np.abs(pt_emb - onnx_emb)))
    mean_diff = float(np.mean(np.abs(pt_emb - onnx_emb)))
    return {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "passed": max_diff < atol,
    }
