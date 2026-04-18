"""Tests for ONNX export and validation."""

from pathlib import Path

import pytest

from src.onnx_export import DEFAULT_ONNX_DIR, export_to_onnx, load_onnx_model, validate_onnx


@pytest.fixture(scope="session")
def onnx_dir() -> Path:
    """Export the model to ONNX once for the test session (or reuse existing)."""
    if (DEFAULT_ONNX_DIR / "model.onnx").exists():
        return DEFAULT_ONNX_DIR
    return export_to_onnx()


def test_export_produces_model_file(onnx_dir):
    assert (onnx_dir / "model.onnx").exists()


def test_export_produces_tokenizer_files(onnx_dir):
    assert (onnx_dir / "tokenizer_config.json").exists()


def test_load_onnx_model_succeeds(onnx_dir):
    ort_model = load_onnx_model(onnx_dir)
    assert ort_model is not None


def test_validate_onnx_passes(model, onnx_dir):
    result = validate_onnx(model, onnx_dir)
    assert result["passed"], f"max_abs_diff={result['max_abs_diff']}"
    assert result["max_abs_diff"] < 1e-4
