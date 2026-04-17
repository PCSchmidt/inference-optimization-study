"""INT8 dynamic quantization for ONNX models."""

from __future__ import annotations

import os
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from .onnx_export import DEFAULT_ONNX_DIR


def quantize_model(
    onnx_dir: Path = DEFAULT_ONNX_DIR,
    output_dir: Path | None = None,
) -> Path:
    """Apply INT8 dynamic quantization to the ONNX model.

    Returns path to the quantized model directory.
    """
    if output_dir is None:
        output_dir = onnx_dir.parent / "onnx_model_int8"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_model = onnx_dir / "model.onnx"
    output_model = output_dir / "model.onnx"

    quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=QuantType.QInt8,
    )

    # Copy tokenizer files
    for f in onnx_dir.iterdir():
        if f.suffix in (".json", ".txt") and not (output_dir / f.name).exists():
            (output_dir / f.name).write_bytes(f.read_bytes())

    return output_dir


def compare_model_sizes(onnx_dir: Path, quantized_dir: Path) -> dict:
    """Compare file sizes of original vs quantized ONNX models."""
    orig = onnx_dir / "model.onnx"
    quant = quantized_dir / "model.onnx"
    orig_mb = orig.stat().st_size / (1024 * 1024) if orig.exists() else 0
    quant_mb = quant.stat().st_size / (1024 * 1024) if quant.exists() else 0
    return {
        "original_mb": round(orig_mb, 2),
        "quantized_mb": round(quant_mb, 2),
        "reduction_pct": round((1 - quant_mb / orig_mb) * 100, 1) if orig_mb else 0,
    }
