#!/usr/bin/env python3
"""
Example usage of the edge deployment bundle.
"""

import sys
from pathlib import Path

def run_examples():
    print("=== ECG Edge Deployment Examples ===\n")
    
    # Example 1: Basic inference
    print("1. Basic inference with sample data:")
    print("   python edge_launcher.py --input data/sample_ecg.csv --output results.json\n")
    
    # Example 2: Verbose mode
    print("2. Verbose inference:")
    print("   python edge_launcher.py --input data/sample_ecg.csv --verbose\n")
    
    # Example 3: Direct runtime usage
    print("3. Direct runtime usage:")
    print("   python runtime/runtime_cli.py --onnx models/ecg_model_int8.onnx --input data/sample_ecg.csv\n")
    
    # Example 4: Benchmarking
    print("4. Performance benchmarking:")
    print("   python runtime/benchmark_runtime.py\n")
    
    print("See docs/DEPLOYMENT_GUIDE.md for detailed instructions.")

if __name__ == "__main__":
    run_examples()