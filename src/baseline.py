"""Baseline PyTorch inference using SentenceTransformer.

Mirrors the production setup from SkillSwap's backend:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, normalize_embeddings=True)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class BaselineResult:
    embeddings: np.ndarray
    elapsed_s: float
    model_size_mb: float


def load_model(device: str = "cpu") -> SentenceTransformer:
    """Load the sentence-transformer model (matches SkillSwap production)."""
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model


def get_model_size_mb(model: SentenceTransformer) -> float:
    """Estimate model size in MB from parameter count."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_bytes / (1024 * 1024)


def run_baseline(
    model: SentenceTransformer,
    texts: list[str],
    normalize: bool = True,
) -> BaselineResult:
    """Run baseline inference and return embeddings + timing."""
    start = time.perf_counter()
    embeddings = model.encode(texts, normalize_embeddings=normalize)
    elapsed = time.perf_counter() - start
    return BaselineResult(
        embeddings=embeddings,
        elapsed_s=elapsed,
        model_size_mb=get_model_size_mb(model),
    )
