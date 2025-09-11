"""
Model optimization utilities for deployment.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging


class ModelOptimizer:
    """
    Utility class for model optimization.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize model optimizer.
        
        Args:
            model: PyTorch model to optimize
            device: Device the model is on
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.model.eval()
    
    def quantize_dynamic(
        self,
        qconfig_spec: Optional[Dict] = None,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization to the model.
        
        Args:
            qconfig_spec: Quantization configuration
            dtype: Target quantization dtype
            
        Returns:
            Quantized model
        """
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                qconfig_spec=qconfig_spec,
                dtype=dtype
            )
            
            self.logger.info(f"Model quantized to {dtype}")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Dynamic quantization failed: {e}")
            raise
    
    def prune_model(
        self,
        pruning_amount: float = 0.2,
        structured: bool = False
    ) -> nn.Module:
        """
        Prune the model to reduce size.
        
        Args:
            pruning_amount: Fraction of weights to prune (0.0 to 1.0)
            structured: Whether to use structured pruning
            
        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune
        except ImportError:
            raise ImportError("Pruning requires torch.nn.utils.prune")
        
        parameters_to_prune = []
        
        # Collect prunable parameters
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        if structured:
            # Structured pruning (not implemented in this basic version)
            self.logger.warning("Structured pruning not implemented, using unstructured")
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_amount,
        )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        self.logger.info(f"Model pruned by {pruning_amount * 100:.1f}%")
        return self.model
    
    def fuse_modules(self) -> nn.Module:
        """
        Fuse consecutive modules for better performance.
        
        Returns:
            Model with fused modules
        """
        try:
            # This is a simplified example - real implementation would need
            # to identify fusible module patterns
            fused_model = torch.quantization.fuse_modules(
                self.model,
                [['conv', 'bn', 'relu']] if hasattr(self.model, 'conv') else [],
                inplace=False
            )
            
            self.logger.info("Modules fused for optimization")
            return fused_model
            
        except Exception as e:
            self.logger.warning(f"Module fusion failed: {e}")
            return self.model


def benchmark_model(
    model: nn.Module,
    input_shape: tuple = (1, 1, 224, 224),
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Benchmark model inference performance.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        device: Device to run benchmark on
        
    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Synchronize for accurate timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    times = sorted(times)
    
    return {
        'mean_ms': sum(times) / len(times),
        'median_ms': times[len(times) // 2],
        'min_ms': min(times),
        'max_ms': max(times),
        'p95_ms': times[int(0.95 * len(times))],
        'p99_ms': times[int(0.99 * len(times))],
        'std_ms': (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5
    }
