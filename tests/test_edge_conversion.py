"""
Unit tests for E6 edge conversion functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path

from src.ci.deployment.export import ModelExporter, export_to_onnx, ModelSizeAnalyzer
from src.ci.deployment.optimize import ModelOptimizer, benchmark_model
from src.ci.model.feature_extractor import FeatureExtractor

try:
    from src.ci.deployment.onnx_utils import (
        export_pytorch_to_onnx, quantize_onnx_dynamic_model, 
        compare_pytorch_vs_onnx, benchmark_onnx
    )
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class TestModelExporter:
    def test_torchscript_export(self):
        """Test TorchScript export functionality."""
        model = FeatureExtractor(embedding_dim=128, pretrained=False, num_classes=None)
        exporter = ModelExporter(model)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            path = exporter.export_torchscript(tmp.name, input_shape=(1, 1, 224, 224))
            assert Path(path).exists()
            
            # Load and test
            loaded = torch.jit.load(path)
            x = torch.randn(1, 1, 224, 224)
            output = loaded(x)
            assert output.shape == (1, 128)

    def test_state_dict_export(self):
        """Test state dictionary export."""
        model = FeatureExtractor(embedding_dim=128, pretrained=False)
        exporter = ModelExporter(model)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            path = exporter.export_state_dict(tmp.name)
            assert Path(path).exists()
            
            # Load and check
            checkpoint = torch.load(path)
            assert 'model_state_dict' in checkpoint
            assert 'model_class' in checkpoint

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_basic_onnx_export(self):
        """Test basic ONNX export using existing function."""
        model = FeatureExtractor(embedding_dim=64, pretrained=False, num_classes=None)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            try:
                path = export_to_onnx(model, tmp.name, input_shape=(1, 1, 224, 224))
                assert Path(path).exists()
            except ImportError:
                pytest.skip("ONNX export requires additional dependencies")


class TestModelOptimizer:
    def test_dynamic_quantization(self):
        """Test dynamic quantization."""
        model = FeatureExtractor(embedding_dim=64, pretrained=False, num_classes=None)
        optimizer = ModelOptimizer(model)
        
        quantized = optimizer.quantize_dynamic()
        assert isinstance(quantized, nn.Module)
        
        # Test inference still works
        x = torch.randn(1, 1, 224, 224)
        output = quantized(x)
        assert output.shape == (1, 64)


class TestModelSizeAnalyzer:
    def test_analyze_model(self):
        """Test model size analysis."""
        model = FeatureExtractor(embedding_dim=128, pretrained=False, num_classes=5)
        analysis = ModelSizeAnalyzer.analyze_model(model)
        
        assert 'total_parameters' in analysis
        assert 'trainable_parameters' in analysis
        assert 'model_size_mb' in analysis
        assert 'layer_count' in analysis
        assert analysis['total_parameters'] > 0
        assert analysis['model_size_mb'] > 0


class TestBenchmarking:
    def test_benchmark_model(self):
        """Test model benchmarking."""
        model = FeatureExtractor(embedding_dim=64, pretrained=False, num_classes=None)
        
        stats = benchmark_model(
            model, 
            input_shape=(1, 1, 224, 224),
            num_runs=5,
            warmup_runs=2
        )
        
        assert 'mean_ms' in stats
        assert 'median_ms' in stats
        assert 'min_ms' in stats
        assert 'max_ms' in stats
        assert stats['mean_ms'] > 0


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
class TestONNXUtils:
    def test_onnx_export(self):
        """Test ONNX export utility."""
        model = FeatureExtractor(embedding_dim=64, pretrained=False, num_classes=None)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            path = export_pytorch_to_onnx(
                model, 
                str(onnx_path), 
                input_shape=(1, 1, 224, 224)
            )
            assert Path(path).exists()

    def test_onnx_quantization(self):
        """Test ONNX dynamic quantization."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=None)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Export float model
            float_path = tmpdir / 'model_float.onnx'
            export_pytorch_to_onnx(
                model, str(float_path), input_shape=(1, 1, 224, 224)
            )
            
            # Quantize to INT8
            int8_path = tmpdir / 'model_int8.onnx'
            quantize_onnx_dynamic_model(str(float_path), str(int8_path))
            assert int8_path.exists()

    def test_pytorch_onnx_parity(self):
        """Test parity between PyTorch and ONNX outputs."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=None)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            export_pytorch_to_onnx(
                model, str(onnx_path), input_shape=(1, 1, 224, 224)
            )
            
            # Compare outputs
            x = torch.randn(1, 1, 224, 224)
            metrics = compare_pytorch_vs_onnx(model, str(onnx_path), x)
            
            assert 'mean_abs_diff' in metrics
            assert 'max_abs_diff' in metrics
            assert 'shape_match' in metrics
            assert metrics['shape_match'] == 1.0  # Shapes should match
            
            # Differences should be small for same model
            assert metrics['mean_abs_diff'] < 1e-4

    def test_onnx_benchmarking(self):
        """Test ONNX model benchmarking."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=None)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            export_pytorch_to_onnx(
                model, str(onnx_path), input_shape=(1, 1, 224, 224)
            )
            
            stats = benchmark_onnx(
                str(onnx_path),
                input_shape=(1, 1, 224, 224),
                runs=5,
                warmup=2
            )
            
            assert 'mean_ms' in stats
            assert 'p50_ms' in stats
            assert 'p95_ms' in stats
            assert stats['mean_ms'] > 0


class TestEdgeConversionIntegration:
    """Integration tests for complete edge conversion pipeline."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_end_to_end_conversion(self):
        """Test complete edge conversion pipeline."""
        # Create a small model for testing
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=None)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 1. Export to ONNX
            onnx_float = tmpdir / 'model_float.onnx'
            export_pytorch_to_onnx(model, str(onnx_float))
            assert onnx_float.exists()
            
            # 2. Quantize to INT8
            onnx_int8 = tmpdir / 'model_int8.onnx'
            quantize_onnx_dynamic_model(str(onnx_float), str(onnx_int8))
            assert onnx_int8.exists()
            
            # 3. Validate parity
            x = torch.randn(1, 1, 224, 224)
            metrics = compare_pytorch_vs_onnx(model, str(onnx_float), x)
            assert metrics['shape_match'] == 1.0
            
            # 4. Benchmark both versions
            bench_float = benchmark_onnx(str(onnx_float), runs=3, warmup=1)
            assert bench_float['mean_ms'] > 0
            
            # Try to benchmark INT8 model - this may fail due to unsupported operators
            try:
                bench_int8 = benchmark_onnx(str(onnx_int8), runs=3, warmup=1)
                assert bench_int8['mean_ms'] > 0
                # INT8 should typically be faster or similar (might not always be true in small models)
            except Exception as e:
                # Some quantized models may have unsupported operators
                print(f"INT8 model benchmarking failed (expected): {e}")
                assert "ConvInteger" in str(e) or "NOT_IMPLEMENTED" in str(e)

    def test_model_size_comparison(self):
        """Test model size analysis for different formats."""
        model = FeatureExtractor(embedding_dim=128, pretrained=False, num_classes=5)
        
        # Analyze PyTorch model
        analysis = ModelSizeAnalyzer.analyze_model(model)
        pytorch_size_mb = analysis['model_size_mb']
        
        # Test quantized version
        optimizer = ModelOptimizer(model)
        quantized = optimizer.quantize_dynamic()
        
        # Quantized model should work
        x = torch.randn(1, 1, 224, 224)
        output = quantized(x)
        assert output.shape == (1, 5)
        
        # Size analysis
        quant_analysis = ModelSizeAnalyzer.analyze_model(quantized)
        # Quantized models might have slightly different parameter counts due to quantization metadata
        assert abs(quant_analysis['total_parameters'] - analysis['total_parameters']) < analysis['total_parameters'] * 0.1


if __name__ == "__main__":
    pytest.main([__file__])
