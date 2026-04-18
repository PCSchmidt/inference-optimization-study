"""Shared fixtures for inference-optimization-study tests."""

import pytest
from sentence_transformers import SentenceTransformer

from src.baseline import MODEL_NAME, load_model

TINY_CORPUS = [
    "machine learning optimization",
    "natural language processing",
    "web development with React",
]


@pytest.fixture(scope="session")
def model() -> SentenceTransformer:
    """Load the model once for the entire test session."""
    return load_model()


@pytest.fixture(scope="session")
def tiny_corpus() -> list[str]:
    return TINY_CORPUS
