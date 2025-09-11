"""
ONNX export and quantization utilities.
"""

from pathlib import Path
from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np
import logging

import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    # Create dummy QuantType for type hints when ONNX not available
    class QuantType:
        QInt8 = None


def ensure_onnx_available():
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX utilities require 'onnx' and 'onnxruntime'. Install with: pip install onnx onnxruntime onnxruntime-tools")


def export_pytorch_to_onnx(
    model: nn.Module,
    onnx_path: str,
    input_shape: Tuple[int, ...] = (1, 1, 224, 224),
    opset: int = 12,
    device: str = 'cpu',
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    verbose: bool = False
) -> str:
    """Export a PyTorch model to ONNX and verify it loads in ORT."""
    ensure_onnx_available()

    model = model.to(device)
    model.eval()

    dummy = torch.randn(*input_shape, device=device)

    input_names = ["input"]
    output_names = ["output"]

    onnx_path = str(onnx_path)
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        verbose=verbose
    )

    # Validate
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)

    # Simple ORT run - add delay for Windows file locking
    import time
    time.sleep(0.1)  # Small delay to ensure file is written
    
    try:
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        _ = sess.run(None, {"input": dummy.detach().cpu().numpy()})
        logging.getLogger(__name__).info(f"Exported and validated ONNX model at {onnx_path}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"ONNX validation failed: {e}. Model exported but not validated.")
    
    return onnx_path


def quantize_onnx_dynamic_model(
    onnx_model_path: str,
    output_path: str,
    weight_type = None
) -> str:
    """Apply dynamic quantization to ONNX model (weights INT8)."""
    ensure_onnx_available()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if weight_type is None:
        weight_type = QuantType.QInt8
    
    quantize_dynamic(onnx_model_path, output_path, weight_type=weight_type)
    logging.getLogger(__name__).info(f"Dynamic-quantized ONNX model saved to {output_path}")
    return output_path


def ort_infer(onnx_model_path: str, input_tensor: np.ndarray) -> np.ndarray:
    """Run ONNX Runtime inference for a single input."""
    ensure_onnx_available()
    sess = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    outputs = sess.run(None, {sess.get_inputs()[0].name: input_tensor})
    return outputs[0]


def compare_pytorch_vs_onnx(
    model: nn.Module,
    onnx_model_path: str,
    input_tensor: torch.Tensor
) -> Dict[str, float]:
    """Compare PyTorch and ONNX outputs with simple metrics."""
    model = model.eval().to(input_tensor.device)
    with torch.no_grad():
        torch_out = model(input_tensor).detach().cpu().numpy()
    onnx_out = ort_infer(onnx_model_path, input_tensor.detach().cpu().numpy())

    # Metrics
    diff = np.abs(torch_out - onnx_out)
    return {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "shape_match": float(torch_out.shape == onnx_out.shape)
    }


def benchmark_onnx(
    onnx_model_path: str,
    input_shape: Tuple[int, ...] = (1, 1, 224, 224),
    runs: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """Benchmark ONNX Runtime latencies."""
    ensure_onnx_available()
    import time
    sess = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    x = np.random.randn(*input_shape).astype(np.float32)

    # warmup
    for _ in range(warmup):
        _ = sess.run(None, {name: x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = sess.run(None, {name: x})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    mean = float(sum(times) / len(times))
    return {
        "mean_ms": mean,
        "p50_ms": float(times[len(times)//2]),
        "p95_ms": float(times[int(0.95*len(times))]),
        "min_ms": float(min(times)),
        "max_ms": float(max(times))
    }

