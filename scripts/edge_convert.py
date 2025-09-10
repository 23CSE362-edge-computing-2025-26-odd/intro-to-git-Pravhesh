#!/usr/bin/env python3
"""
E6 Edge Conversion script: export to ONNX, quantize to INT8, and benchmark parity/latency.
This script avoids committing; run locally and review outputs.
"""

import argparse
import logging
from pathlib import Path

import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ci.model.model import create_feature_extractor
from ci.deployment.onnx_utils import (
    export_pytorch_to_onnx,
    quantize_onnx_dynamic_model,
    compare_pytorch_vs_onnx,
    benchmark_onnx,
    ensure_onnx_available
)


def main():
    parser = argparse.ArgumentParser(description="E6: Edge conversion to ONNX + INT8 quantization")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Model config YAML')
    parser.add_argument('--outdir', type=str, default='edge_artifacts', help='Output directory')
    parser.add_argument('--height', type=int, default=224, help='Input height')
    parser.add_argument('--width', type=int, default=224, help='Input width')
    parser.add_argument('--runs', type=int, default=50, help='Benchmark runs')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ensure_onnx_available()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Create feature extractor (single-channel spectrogram input)
    model = create_feature_extractor(args.config)
    model.eval()

    input_shape = (1, 1, args.height, args.width)

    # 2) Export to ONNX
    onnx_path = outdir / 'model_float.onnx'
    export_pytorch_to_onnx(model, str(onnx_path), input_shape=input_shape)

    # 3) Compare a random input between PyTorch and ONNX
    x = torch.randn(*input_shape)
    metrics = compare_pytorch_vs_onnx(model, str(onnx_path), x)
    logging.info(f"Parity (PyTorch vs ONNX): {metrics}")

    # 4) Quantize dynamically to INT8 (weights) as a portable baseline
    onnx_int8 = outdir / 'model_int8.onnx'
    quantize_onnx_dynamic_model(str(onnx_path), str(onnx_int8))

    # 5) Benchmark ONNX float vs INT8
    bench_float = benchmark_onnx(str(onnx_path), input_shape=input_shape, runs=args.runs)
    bench_int8 = benchmark_onnx(str(onnx_int8), input_shape=input_shape, runs=args.runs)
    logging.info(f"Latency float32: {bench_float}")
    logging.info(f"Latency INT8:   {bench_int8}")

    # Save summary
    summary = outdir / 'edge_summary.txt'
    with open(summary, 'w') as f:
        f.write(f"Parity: {metrics}\n")
        f.write(f"Latency float32: {bench_float}\n")
        f.write(f"Latency INT8:   {bench_int8}\n")
    logging.info(f"Summary saved to {summary}")


if __name__ == '__main__':
    main()

