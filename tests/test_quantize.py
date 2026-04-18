"""Tests for INT8 dynamic quantization."""

from pathlib import Path

import pytest

from src.onnx_export import DEFAULT_ONNX_DIR
from src.quantize import quantize_model, compare_model_sizes

QUANTIZED_DIR = DEFAULT_ONNX_DIR.parent / "onnx_model_int8"


@pytest.fixture(scope="session")
def quantized_dir() -> Path:
    """Quantize the ONNX model once for the test session (or reuse existing)."""
    if (QUANTIZED_DIR / "model.onnx").exists():
        return QUANTIZED_DIR
    return quantize_model()


def test_quantized_model_file_exists(quantized_dir):
    assert (quantized_dir / "model.onnx").exists()


def test_quantized_model_is_smaller(quantized_dir):
    sizes = compare_model_sizes(DEFAULT_ONNX_DIR, quantized_dir)
    assert sizes["quantized_mb"] < sizes["original_mb"]
    assert sizes["reduction_pct"] > 50  # expect at least 50% reduction


def test_compare_model_sizes_keys(quantized_dir):
    sizes = compare_model_sizes(DEFAULT_ONNX_DIR, quantized_dir)
    assert set(sizes.keys()) == {"original_mb", "quantized_mb", "reduction_pct"}
