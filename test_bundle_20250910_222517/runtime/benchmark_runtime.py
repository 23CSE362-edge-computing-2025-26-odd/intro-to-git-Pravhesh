#!/usr/bin/env python3
"""
E7 Runtime Performance Benchmarks

This script validates the runtime system performance by comparing:
1. PyTorch Float32 model
2. ONNX Float32 model  
3. ONNX INT8 model (if supported)
4. With and without fuzzy logic processing
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import time
import json
import numpy as np
import torch
from typing import Dict, List, Any
import statistics

# Import our runtime components
sys.path.insert(0, str(project_root / 'scripts'))
from runtime_cli import ECGRuntimeEngine, load_ecg_data, create_sample_ecg_data

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    

def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    
    print("=== E7 Runtime Performance Benchmarks ===\n")
    
    # Create test data
    test_data_path = "benchmark_ecg.csv"
    if not Path(test_data_path).exists():
        print("Creating benchmark ECG data...")
        create_sample_ecg_data(test_data_path, duration_seconds=30)
    
    # Load test data
    ecg_data = load_ecg_data(test_data_path)
    print(f"Loaded ECG data: {ecg_data.shape[0]:,} samples ({ecg_data.shape[0]/360:.1f}s @ 360Hz)\n")
    
    # Define benchmark configurations
    configs = [
        {
            'name': 'PyTorch Float32',
            'model_path': None,
            'onnx_path': None,
            'fuzzy': False
        },
        {
            'name': 'PyTorch Float32 + Fuzzy',
            'model_path': None,
            'onnx_path': None,
            'fuzzy': True
        }
    ]
    
    # Add ONNX configs if available
    if ONNX_AVAILABLE:
        if Path("demo_outputs/ecg_model.onnx").exists():
            configs.extend([
                {
                    'name': 'ONNX Float32',
                    'model_path': None,
                    'onnx_path': 'demo_outputs/ecg_model.onnx',
                    'fuzzy': False
                },
                {
                    'name': 'ONNX Float32 + Fuzzy',
                    'model_path': None,
                    'onnx_path': 'demo_outputs/ecg_model.onnx',
                    'fuzzy': True
                }
            ])
        
        # Try INT8 model (may not work on all systems)
        if Path("demo_outputs/ecg_model_int8.onnx").exists():
            configs.append({
                'name': 'ONNX INT8',
                'model_path': None,
                'onnx_path': 'demo_outputs/ecg_model_int8.onnx',
                'fuzzy': False
            })
    
    results = []
    
    # Run benchmarks
    for config in configs:
        print(f"Benchmarking: {config['name']}")
        
        try:
            # Initialize engine
            engine = ECGRuntimeEngine(
                model_path=config['model_path'],
                onnx_path=config['onnx_path'],
                fuzzy_config_path='configs/fuzzy_config.yaml' if config['fuzzy'] else None
            )
            
            # Warm-up runs
            print("  Warming up...")
            for _ in range(3):
                engine.predict(ecg_data)
            
            # Benchmark runs
            print("  Running benchmark...")
            latencies = []
            successful_runs = 0
            
            for i in range(20):
                start_time = time.time()
                result = engine.predict(ecg_data)
                end_time = time.time()
                
                if 'error' not in result:
                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
                    successful_runs += 1
                else:
                    print(f"    Run {i+1} failed: {result.get('error', 'Unknown error')}")
            
            if latencies:
                # Calculate statistics
                stats = {
                    'configuration': config['name'],
                    'successful_runs': successful_runs,
                    'total_runs': 20,
                    'success_rate': successful_runs / 20 * 100,
                    'mean_latency_ms': statistics.mean(latencies),
                    'median_latency_ms': statistics.median(latencies),
                    'min_latency_ms': min(latencies),
                    'max_latency_ms': max(latencies),
                    'p95_latency_ms': np.percentile(latencies, 95),
                    'p99_latency_ms': np.percentile(latencies, 99),
                    'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                    'throughput_samples_per_sec': ecg_data.shape[0] / (statistics.mean(latencies) / 1000),
                    'has_fuzzy': config['fuzzy'],
                    'model_type': 'ONNX' if config['onnx_path'] else 'PyTorch'
                }
                
                results.append(stats)
                
                print(f"  ✓ Mean: {stats['mean_latency_ms']:.1f}ms")
                print(f"    P95: {stats['p95_latency_ms']:.1f}ms")
                print(f"    Throughput: {stats['throughput_samples_per_sec']:.0f} samples/sec")
                print(f"    Success rate: {stats['success_rate']:.1f}%")
                
            else:
                print(f"  ✗ All runs failed")
            
        except Exception as e:
            print(f"  ✗ Configuration failed: {e}")
        
        print()
    
    # Analysis and comparison
    print("=== Performance Analysis ===\n")
    
    if results:
        # Find baseline (PyTorch Float32 without fuzzy)
        baseline = None
        for result in results:
            if result['model_type'] == 'PyTorch' and not result['has_fuzzy']:
                baseline = result
                break
        
        # Performance comparison table
        print("Configuration                    | Mean Latency | P95 Latency | Throughput | Speedup")
        print("-" * 80)
        
        for result in results:
            speedup = baseline['mean_latency_ms'] / result['mean_latency_ms'] if baseline else 1.0
            print(f"{result['configuration']:<32} | {result['mean_latency_ms']:>10.1f}ms | "
                  f"{result['p95_latency_ms']:>9.1f}ms | {result['throughput_samples_per_sec']:>8.0f}/s | "
                  f"{speedup:>5.2f}x")
        
        print()
        
        # Fuzzy logic overhead analysis
        pytorch_base = None
        pytorch_fuzzy = None
        
        for result in results:
            if result['model_type'] == 'PyTorch':
                if not result['has_fuzzy']:
                    pytorch_base = result
                else:
                    pytorch_fuzzy = result
        
        if pytorch_base and pytorch_fuzzy:
            fuzzy_overhead = pytorch_fuzzy['mean_latency_ms'] - pytorch_base['mean_latency_ms']
            fuzzy_overhead_pct = (fuzzy_overhead / pytorch_base['mean_latency_ms']) * 100
            print(f"Fuzzy Logic Overhead: +{fuzzy_overhead:.1f}ms ({fuzzy_overhead_pct:.1f}%)")
        
        # ONNX vs PyTorch comparison
        onnx_results = [r for r in results if r['model_type'] == 'ONNX' and not r['has_fuzzy']]
        pytorch_results = [r for r in results if r['model_type'] == 'PyTorch' and not r['has_fuzzy']]
        
        if onnx_results and pytorch_results:
            onnx_speedup = pytorch_results[0]['mean_latency_ms'] / onnx_results[0]['mean_latency_ms']
            print(f"ONNX Speedup: {onnx_speedup:.2f}x faster than PyTorch")
        
        # Model size comparison
        print("\n=== Model Size Analysis ===")
        
        model_files = [
            ('PyTorch Model (estimated)', None, 'N/A'),
            ('ONNX Float32', 'demo_outputs/ecg_model.onnx', None),
            ('ONNX INT8', 'demo_outputs/ecg_model_int8.onnx', None)
        ]
        
        for name, path, size_str in model_files:
            if path and Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                print(f"{name}: {size_mb:.2f} MB")
            elif size_str:
                print(f"{name}: {size_str}")
        
        # Save results
        results_file = "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'ecg_data_samples': int(ecg_data.shape[0]),
                'ecg_duration_seconds': ecg_data.shape[0] / 360,
                'results': results
            }, f, indent=2)
        
        print(f"\n✓ Detailed results saved to: {results_file}")
        
        # Summary
        print("\n=== Summary ===")
        best_result = min(results, key=lambda x: x['mean_latency_ms'])
        print(f"Best performance: {best_result['configuration']}")
        print(f"  Latency: {best_result['mean_latency_ms']:.1f}ms")
        print(f"  Throughput: {best_result['throughput_samples_per_sec']:.0f} samples/sec")
        
        print("\n✓ E7 runtime performance validation completed!")
        
    else:
        print("No successful benchmark results to analyze.")
        

def validate_model_accuracy():
    """Quick validation that all models produce similar outputs."""
    
    print("\n=== Model Accuracy Validation ===")
    
    # Create small test signal
    test_signal = np.random.randn(3600)  # 10 seconds at 360Hz
    
    results = {}
    
    # Test PyTorch model
    try:
        engine = ECGRuntimeEngine()
        result = engine.predict(test_signal)
        if 'error' not in result:
            results['PyTorch'] = result['probabilities']
            print("✓ PyTorch model prediction successful")
    except Exception as e:
        print(f"✗ PyTorch model failed: {e}")
    
    # Test ONNX model
    if ONNX_AVAILABLE and Path("demo_outputs/ecg_model.onnx").exists():
        try:
            engine = ECGRuntimeEngine(onnx_path="demo_outputs/ecg_model.onnx")
            result = engine.predict(test_signal)
            if 'error' not in result:
                results['ONNX'] = result['probabilities']
                print("✓ ONNX model prediction successful")
        except Exception as e:
            print(f"✗ ONNX model failed: {e}")
    
    # Compare results
    if len(results) >= 2:
        keys = list(results.keys())
        probs1 = np.array(results[keys[0]])
        probs2 = np.array(results[keys[1]])
        
        diff = np.abs(probs1 - probs2).mean()
        print(f"Mean probability difference between {keys[0]} and {keys[1]}: {diff:.6f}")
        
        if diff < 0.01:
            print("✓ Models show good agreement")
        else:
            print("⚠ Models show significant differences")


if __name__ == "__main__":
    run_benchmark_suite()
    validate_model_accuracy()
