"""
Advanced quantization utilities for edge deployment.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CalibrationDataLoader:
    """Data loader for calibration during quantization."""
    
    def __init__(self, data_func: Callable[[], np.ndarray], num_samples: int = 100):
        """
        Args:
            data_func: Function that returns a representative data sample
            num_samples: Number of calibration samples
        """
        self.data_func = data_func
        self.num_samples = num_samples
        
    def __iter__(self):
        for _ in range(self.num_samples):
            yield self.data_func()
    
    def __len__(self):
        return self.num_samples


class PyTorchQuantizer:
    """PyTorch-based quantization utilities."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def quantize_dynamic(self, qconfig_spec=None, dtype=torch.qint8):
        """Apply dynamic quantization."""
        if qconfig_spec is None:
            qconfig_spec = {torch.nn.Linear}
        
        quantized = torch.quantization.quantize_dynamic(
            self.model, qconfig_spec, dtype=dtype
        )
        logger.info(f"Applied dynamic quantization with dtype {dtype}")
        return quantized
    
    def prepare_qat(self):
        """Prepare model for Quantization Aware Training."""
        # Set quantization config
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        prepared = torch.quantization.prepare_qat(self.model)
        logger.info("Model prepared for QAT")
        return prepared
    
    def quantize_qat(self, prepared_model):
        """Convert QAT model to quantized version."""
        quantized = torch.quantization.convert(prepared_model)
        logger.info("Converted QAT model to quantized version")
        return quantized
    
    def fuse_modules(self, modules_to_fuse: List[List[str]]):
        """Fuse modules for better quantization."""
        try:
            fused = torch.quantization.fuse_modules(self.model, modules_to_fuse)
            logger.info(f"Fused modules: {modules_to_fuse}")
            return fused
        except Exception as e:
            logger.warning(f"Module fusion failed: {e}")
            return self.model


def create_representative_dataset(
    input_shape: Tuple[int, ...],
    num_samples: int = 100,
    data_type: str = "random"
) -> Callable[[], np.ndarray]:
    """Create representative dataset for calibration."""
    
    if data_type == "random":
        def random_data():
            return np.random.randn(*input_shape).astype(np.float32)
        return random_data
    
    elif data_type == "gaussian":
        def gaussian_data():
            return np.random.normal(0, 0.5, input_shape).astype(np.float32)
        return gaussian_data
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def validate_quantized_model(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_inputs: List[torch.Tensor],
    tolerance: float = 1e-2
) -> Dict[str, Any]:
    """Validate quantized model against original."""
    
    original_model.eval()
    quantized_model.eval()
    
    differences = []
    
    with torch.no_grad():
        for test_input in test_inputs:
            # Original model output
            orig_output = original_model(test_input).cpu().numpy()
            
            # Quantized model output
            if hasattr(quantized_model, 'forward'):
                quant_output = quantized_model(test_input).cpu().numpy()
            else:
                # Handle different quantized model formats
                quant_output = quantized_model(test_input.cpu()).detach().numpy()
            
            # Calculate difference
            diff = np.abs(orig_output - quant_output)
            differences.append({
                'mean_abs_diff': float(np.mean(diff)),
                'max_abs_diff': float(np.max(diff)),
                'mse': float(np.mean((orig_output - quant_output) ** 2))
            })
    
    # Aggregate results
    mean_diff = np.mean([d['mean_abs_diff'] for d in differences])
    max_diff = np.max([d['max_abs_diff'] for d in differences])
    mean_mse = np.mean([d['mse'] for d in differences])
    
    validation_passed = mean_diff < tolerance
    
    return {
        'validation_passed': validation_passed,
        'mean_abs_diff': mean_diff,
        'max_abs_diff': max_diff,
        'mean_mse': mean_mse,
        'tolerance': tolerance,
        'per_sample_results': differences
    }


def benchmark_quantization_impact(
    original_model: nn.Module,
    quantized_model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 224, 224),
    num_runs: int = 100
) -> Dict[str, Any]:
    """Benchmark the impact of quantization on performance."""
    
    from .optimize import benchmark_model
    
    # Benchmark original model
    orig_stats = benchmark_model(
        original_model, input_shape, num_runs=num_runs, warmup_runs=10
    )
    
    # Benchmark quantized model
    quant_stats = benchmark_model(
        quantized_model, input_shape, num_runs=num_runs, warmup_runs=10
    )
    
    # Calculate improvements
    speedup = orig_stats['mean_ms'] / quant_stats['mean_ms']
    
    return {
        'original_model': orig_stats,
        'quantized_model': quant_stats,
        'speedup': speedup,
        'latency_reduction_percent': (1 - quant_stats['mean_ms'] / orig_stats['mean_ms']) * 100
    }


def get_model_size(model: nn.Module, include_buffers: bool = True) -> Dict[str, float]:
    """Calculate model size in different units."""
    
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    if include_buffers:
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    else:
        buffer_size = 0
    
    total_size = param_size + buffer_size
    
    return {
        'total_bytes': total_size,
        'total_mb': total_size / (1024 * 1024),
        'total_kb': total_size / 1024,
        'param_bytes': param_size,
        'buffer_bytes': buffer_size
    }


def compare_model_sizes(
    original_model: nn.Module,
    quantized_model: nn.Module
) -> Dict[str, Any]:
    """Compare sizes of original and quantized models."""
    
    orig_size = get_model_size(original_model)
    quant_size = get_model_size(quantized_model)
    
    size_reduction = orig_size['total_bytes'] - quant_size['total_bytes']
    compression_ratio = orig_size['total_bytes'] / quant_size['total_bytes']
    size_reduction_percent = (size_reduction / orig_size['total_bytes']) * 100
    
    return {
        'original_model': orig_size,
        'quantized_model': quant_size,
        'size_reduction_bytes': size_reduction,
        'size_reduction_mb': size_reduction / (1024 * 1024),
        'compression_ratio': compression_ratio,
        'size_reduction_percent': size_reduction_percent
    }


class QuantizationPipeline:
    """Complete quantization pipeline for edge deployment."""
    
    def __init__(self, model: nn.Module, calibration_loader: Optional[CalibrationDataLoader] = None):
        self.model = model
        self.calibration_loader = calibration_loader
        self.quantizer = PyTorchQuantizer(model)
        
    def run_dynamic_quantization(self) -> nn.Module:
        """Run dynamic quantization pipeline."""
        logger.info("Starting dynamic quantization pipeline")
        
        # Fuse modules if possible
        try:
            fused_model = self.quantizer.fuse_modules([])  # Auto-detect fusible modules
            self.quantizer.model = fused_model
        except:
            logger.info("Module fusion skipped")
        
        # Apply dynamic quantization
        quantized = self.quantizer.quantize_dynamic()
        
        logger.info("Dynamic quantization pipeline completed")
        return quantized
    
    def run_full_pipeline(
        self,
        validation_inputs: List[torch.Tensor],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete quantization and validation pipeline."""
        
        # 1. Dynamic quantization
        quantized = self.run_dynamic_quantization()
        
        # 2. Validation
        validation_results = validate_quantized_model(
            self.model, quantized, validation_inputs
        )
        
        # 3. Size comparison
        size_comparison = compare_model_sizes(self.model, quantized)
        
        # 4. Performance benchmark
        performance_results = benchmark_quantization_impact(
            self.model, quantized
        )
        
        results = {
            'quantized_model': quantized,
            'validation': validation_results,
            'size_comparison': size_comparison,
            'performance': performance_results,
            'success': validation_results['validation_passed']
        }
        
        # Save results if output directory provided
        if output_dir:
            self._save_results(results, output_dir)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Save quantization results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save quantized model
        torch.save(results['quantized_model'], output_path / 'quantized_model.pth')
        
        # Save metrics
        import json
        metrics = {k: v for k, v in results.items() if k != 'quantized_model'}
        
        with open(output_path / 'quantization_results.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Quantization results saved to {output_path}")


def quantize_model_for_edge(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 224, 224),
    num_calibration_samples: int = 100,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Convenient function to quantize a model for edge deployment."""
    
    # Create calibration data
    data_func = create_representative_dataset(input_shape, data_type="gaussian")
    calibration_loader = CalibrationDataLoader(data_func, num_calibration_samples)
    
    # Create validation inputs
    validation_inputs = [torch.randn(*input_shape) for _ in range(10)]
    
    # Run quantization pipeline
    pipeline = QuantizationPipeline(model, calibration_loader)
    results = pipeline.run_full_pipeline(validation_inputs, output_dir)
    
    return results
