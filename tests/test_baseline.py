"""Tests for baseline PyTorch inference."""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.baseline import load_model, get_model_size_mb, run_baseline


def test_load_model_returns_sentence_transformer(model):
    assert isinstance(model, SentenceTransformer)


def test_model_size_is_positive(model):
    size = get_model_size_mb(model)
    assert size > 0


def test_run_baseline_returns_correct_shape(model, tiny_corpus):
    result = run_baseline(model, tiny_corpus)
    assert result.embeddings.shape == (len(tiny_corpus), 384)
    assert result.elapsed_s > 0
    assert result.model_size_mb > 0


def test_baseline_embeddings_are_normalized(model, tiny_corpus):
    result = run_baseline(model, tiny_corpus, normalize=True)
    norms = np.linalg.norm(result.embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
