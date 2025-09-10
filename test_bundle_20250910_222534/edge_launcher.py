#!/usr/bin/env python3
"""
Edge ECG Diagnosis Launcher

Simplified launcher for edge deployment with automatic model selection
and configuration based on available resources.
"""

import sys
import os
from pathlib import Path
import json
import time
import argparse

# Add runtime modules to path
runtime_dir = Path(__file__).parent
sys.path.insert(0, str(runtime_dir))

try:
    from runtime_cli import ECGRuntimeEngine, load_ecg_data
    RUNTIME_AVAILABLE = True
except ImportError as e:
    print(f"Error importing runtime: {e}")
    RUNTIME_AVAILABLE = False

def detect_system_resources():
    """Detect available system resources."""
    import psutil
    
    return {
        'memory_mb': psutil.virtual_memory().available / (1024*1024),
        'cpu_count': psutil.cpu_count(),
        'disk_free_gb': psutil.disk_usage('.').free / (1024*1024*1024)
    }

def select_optimal_model(resources: dict):
    """Select optimal model based on available resources."""
    models_dir = Path(__file__).parent / "models"
    
    # Check available models
    float32_model = models_dir / "ecg_model_float32.onnx"
    int8_model = models_dir / "ecg_model_int8.onnx"
    
    if resources['memory_mb'] < 512 and int8_model.exists():
        return str(int8_model), "int8"
    elif float32_model.exists():
        return str(float32_model), "float32"
    else:
        return None, None

def run_diagnosis(input_file: str, output_file: str = None):
    """Run ECG diagnosis on input file."""
    if not RUNTIME_AVAILABLE:
        print("Runtime not available. Please check installation.")
        return False
    
    print("=== Edge ECG Diagnosis System ===")
    
    # Detect system resources
    try:
        resources = detect_system_resources()
        print(f"System resources: {resources['memory_mb']:.0f}MB RAM, {resources['cpu_count']} CPUs")
    except ImportError:
        resources = {'memory_mb': 1024, 'cpu_count': 1}  # Default assumption
        print("Resource detection not available, using defaults")
    
    # Select optimal model
    model_path, model_type = select_optimal_model(resources)
    if not model_path:
        print("No suitable models found in models/ directory")
        return False
    
    print(f"Selected model: {model_type} ({Path(model_path).name})")
    
    # Initialize runtime engine
    try:
        engine = ECGRuntimeEngine(
            onnx_path=model_path,
            fuzzy_config_path="configs/fuzzy_config.yaml" if Path("configs/fuzzy_config.yaml").exists() else None
        )
        print("✓ Runtime engine initialized")
    except Exception as e:
        print(f"✗ Failed to initialize engine: {e}")
        return False
    
    # Load and process ECG data
    try:
        ecg_data = load_ecg_data(input_file)
        print(f"✓ Loaded ECG data: {ecg_data.shape[0]:,} samples")
    except Exception as e:
        print(f"✗ Failed to load ECG data: {e}")
        return False
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    
    try:
        results = engine.predict(ecg_data)
        inference_time = time.time() - start_time
        
        if 'error' in results:
            print(f"✗ Inference failed: {results['error']}")
            return False
        
        print(f"✓ Inference completed in {inference_time*1000:.1f}ms")
        print(f"  Diagnosis: {results.get('prediction', 'Unknown')}")
        print(f"  Confidence: {results.get('confidence', 0):.3f}")
        
        if 'fuzzy_risk_score' in results:
            print(f"  Risk Score: {results['fuzzy_risk_score']:.3f}")
            print(f"  Risk Level: {results['fuzzy_risk_level']}")
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✓ Results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Edge ECG Diagnosis Launcher")
    parser.add_argument('--input', '-i', required=True, help='Input ECG file (CSV/NPY)')
    parser.add_argument('--output', '-o', help='Output results file (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    success = run_diagnosis(args.input, args.output)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())