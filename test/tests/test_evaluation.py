"""
Tests for evaluation, inference, and deployment components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path
import yaml
import json

from src.ci.model import FeatureExtractor
from src.ci.evaluation import ClassificationMetrics, MetricsCalculator, ModelEvaluator, ECGInferenceEngine
from src.ci.evaluation.inference import InferenceConfig
from src.ci.deployment import ModelExporter, export_to_onnx, ModelOptimizer, ModelServer
from src.ci.preprocess import WindowConfig, SpectrogramConfig


class TestMetricsCalculator:
    """Test cases for MetricsCalculator class."""
    
    def test_metrics_calculator_init(self):
        """Test MetricsCalculator initialization."""
        class_names = ['Class_A', 'Class_B', 'Class_C']
        calculator = MetricsCalculator(class_names)
        
        assert calculator.class_names == class_names
    
    def test_calculate_metrics_binary(self):
        """Test metrics calculation for binary classification."""
        calculator = MetricsCalculator(['Normal', 'Abnormal'])
        
        # Simple binary classification example
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_pred_proba = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.9, 0.1], [0.7, 0.3]])
        
        metrics = calculator.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert isinstance(metrics, ClassificationMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
        assert metrics.roc_auc is not None
        assert 0.0 <= metrics.roc_auc <= 1.0
        assert metrics.confusion_matrix.shape == (2, 2)
    
    def test_calculate_metrics_multiclass(self):
        """Test metrics calculation for multi-class classification."""
        calculator = MetricsCalculator(['Class_0', 'Class_1', 'Class_2'])
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        y_pred_proba = np.array([
            [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5],
            [0.9, 0.05, 0.05], [0.3, 0.2, 0.5], [0.1, 0.1, 0.8]
        ])
        
        metrics = calculator.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert isinstance(metrics, ClassificationMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.confusion_matrix.shape == (3, 3)
        assert len(metrics.per_class_precision) == 3
        assert len(metrics.per_class_recall) == 3
        assert len(metrics.per_class_f1) == 3
    
    def test_create_metrics_summary(self):
        """Test metrics summary creation."""
        calculator = MetricsCalculator(['Normal', 'Abnormal'])
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        metrics = calculator.calculate_metrics(y_true, y_pred)
        summary = calculator.create_metrics_summary(metrics)
        
        assert "ECG Classification Metrics Summary" in summary
        assert "Overall Performance:" in summary
        assert "Per-Class Performance:" in summary
        assert "Detailed Classification Report:" in summary


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def test_model_evaluator_init(self):
        """Test ModelEvaluator initialization."""
        model = FeatureExtractor(embedding_dim=64, pretrained=False, num_classes=3)
        class_names = ['Normal', 'Abnormal_1', 'Abnormal_2']
        
        evaluator = ModelEvaluator(model, device='cpu', class_names=class_names)
        
        assert evaluator.model is not None
        assert evaluator.device == 'cpu'
        assert evaluator.class_names == class_names
        assert evaluator.model.training == False  # Should be in eval mode
    
    def test_evaluate_single_batch(self):
        """Test single batch evaluation."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        evaluator = ModelEvaluator(model, device='cpu')
        
        # Create dummy batch
        batch_size = 4
        data = torch.randn(batch_size, 1, 224, 224)
        targets = torch.randint(0, 2, (batch_size,))
        
        metrics, predictions = evaluator.evaluate_single_batch(data, targets)
        
        assert isinstance(metrics, ClassificationMetrics)
        assert predictions['predictions'].shape == (batch_size,)
        assert predictions['probabilities'].shape == (batch_size, 2)
        assert predictions['labels'].shape == (batch_size,)
        assert 'loss' in predictions


class TestECGInferenceEngine:
    """Test cases for ECGInferenceEngine class."""
    
    def test_inference_engine_init(self):
        """Test ECGInferenceEngine initialization."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=3)
        class_names = ['Normal', 'Abnormal_1', 'Abnormal_2']
        
        engine = ECGInferenceEngine(
            model=model,
            device='cpu',
            class_names=class_names
        )
        
        assert engine.model is not None
        assert engine.device == 'cpu'
        assert engine.class_names == class_names
        assert engine.pipeline is not None
    
    def test_predict_signal(self):
        """Test signal prediction."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        
        engine = ECGInferenceEngine(
            model=model,
            device='cpu',
            class_names=['Normal', 'Abnormal']
        )
        
        # Create dummy ECG signal (10 seconds at 360 Hz)
        signal = np.random.randn(3600)
        
        result = engine.predict_signal(signal)
        
        assert 'error' not in result
        assert 'final_prediction' in result
        assert 'final_probabilities' in result
        assert 'num_windows' in result
        assert result['final_prediction'] in ['Normal', 'Abnormal']
        assert len(result['final_probabilities']) == 2
        assert abs(sum(result['final_probabilities']) - 1.0) < 1e-5  # Probabilities sum to 1
    
    def test_predict_signal_short(self):
        """Test prediction on short signal."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        
        engine = ECGInferenceEngine(
            model=model,
            device='cpu',
            class_names=['Normal', 'Abnormal']
        )
        
        # Create short signal (1 second at 360 Hz)
        signal = np.random.randn(360)
        
        result = engine.predict_signal(signal)
        
        # Should still work but might have fewer windows
        assert 'final_prediction' in result or 'error' in result


class TestModelExporter:
    """Test cases for ModelExporter class."""
    
    def test_model_exporter_init(self):
        """Test ModelExporter initialization."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        exporter = ModelExporter(model, device='cpu')
        
        assert exporter.model is not None
        assert exporter.device == 'cpu'
        assert exporter.model.training == False  # Should be in eval mode
    
    def test_export_state_dict(self):
        """Test state dictionary export."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        exporter = ModelExporter(model, device='cpu')
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            saved_path = exporter.export_state_dict(temp_path, include_metadata=True)
            
            assert Path(saved_path).exists()
            
            # Load and verify
            checkpoint = torch.load(saved_path, map_location='cpu')
            assert 'model_state_dict' in checkpoint
            assert 'model_class' in checkpoint
            assert 'total_parameters' in checkpoint
            
        finally:
            Path(temp_path).unlink()
    
    def test_export_torchscript(self):
        """Test TorchScript export."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        exporter = ModelExporter(model, device='cpu')
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            saved_path = exporter.export_torchscript(temp_path, input_shape=(1, 1, 224, 224))
            
            assert Path(saved_path).exists()
            
            # Load and test
            traced_model = torch.jit.load(saved_path)
            test_input = torch.randn(1, 1, 224, 224)
            output = traced_model(test_input)
            assert output.shape == (1, 2)
            
        finally:
            Path(temp_path).unlink()


class TestModelOptimizer:
    """Test cases for ModelOptimizer class."""
    
    def test_model_optimizer_init(self):
        """Test ModelOptimizer initialization."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        optimizer = ModelOptimizer(model, device='cpu')
        
        assert optimizer.model is not None
        assert optimizer.device == 'cpu'
    
    def test_quantize_dynamic(self):
        """Test dynamic quantization."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        optimizer = ModelOptimizer(model, device='cpu')
        
        # Test quantization
        quantized_model = optimizer.quantize_dynamic()
        
        assert quantized_model is not None
        
        # Test that quantized model still works
        test_input = torch.randn(1, 1, 224, 224)
        output = quantized_model(test_input)
        assert output.shape == (1, 2)


class TestModelServer:
    """Test cases for ModelServer class."""
    
    def test_model_server_init(self):
        """Test ModelServer initialization."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        class_names = ['Normal', 'Abnormal']
        
        server = ModelServer(
            model=model,
            class_names=class_names,
            device='cpu'
        )
        
        assert server.inference_engine is not None
        assert server.stats['requests_processed'] == 0
    
    def test_predict(self):
        """Test server prediction."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        
        server = ModelServer(
            model=model,
            class_names=['Normal', 'Abnormal'],
            device='cpu'
        )
        
        # Test prediction
        signal = np.random.randn(1800)  # 5 seconds at 360 Hz
        result = server.predict(signal)
        
        assert result['success'] is True
        assert 'prediction' in result
        assert 'request_id' in result
        assert server.stats['requests_processed'] == 1
    
    def test_batch_predict(self):
        """Test batch prediction."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        
        server = ModelServer(
            model=model,
            class_names=['Normal', 'Abnormal'],
            device='cpu'
        )
        
        # Test batch prediction
        signals = [np.random.randn(1800) for _ in range(3)]
        results = server.batch_predict(signals)
        
        assert len(results) == 3
        assert all(r['success'] for r in results)
        assert server.stats['requests_processed'] == 3
    
    def test_health_check(self):
        """Test server health check."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        
        server = ModelServer(
            model=model,
            class_names=['Normal', 'Abnormal'],
            device='cpu'
        )
        
        health = server.health_check()
        
        assert 'status' in health
        assert health['status'] in ['healthy', 'unhealthy']
        assert 'model_loaded' in health
        assert 'inference_engine_ready' in health
    
    def test_get_model_info(self):
        """Test get model info."""
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        
        server = ModelServer(
            model=model,
            class_names=['Normal', 'Abnormal'],
            device='cpu'
        )
        
        info = server.get_model_info()
        
        assert 'model_class' in info
        assert 'device' in info
        assert 'class_names' in info
        assert 'total_parameters' in info
        assert 'inference_config' in info
        assert info['class_names'] == ['Normal', 'Abnormal']


class TestInferenceConfig:
    """Test cases for InferenceConfig class."""
    
    def test_inference_config_defaults(self):
        """Test InferenceConfig with default values."""
        config = InferenceConfig()
        
        assert config.window_seconds == 5.0
        assert config.overlap_fraction == 0.5
        assert config.sampling_rate_hz == 360
        assert config.n_fft == 256
        assert config.normalize == 'zscore'
    
    def test_inference_config_custom(self):
        """Test InferenceConfig with custom values."""
        config = InferenceConfig(
            window_seconds=10.0,
            overlap_fraction=0.25,
            sampling_rate_hz=1000,
            n_fft=512,
            normalize='minmax'
        )
        
        assert config.window_seconds == 10.0
        assert config.overlap_fraction == 0.25
        assert config.sampling_rate_hz == 1000
        assert config.n_fft == 512
        assert config.normalize == 'minmax'


class TestIntegrationE4:
    """Integration tests for Task E4 components."""
    
    def test_end_to_end_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Create model
        model = FeatureExtractor(embedding_dim=64, pretrained=False, num_classes=3)
        class_names = ['Normal', 'Abnormal_1', 'Abnormal_2']
        
        # Create evaluator
        evaluator = ModelEvaluator(model, device='cpu', class_names=class_names)
        
        # Create dummy data
        batch_size = 8
        data = torch.randn(batch_size, 1, 224, 224)
        targets = torch.randint(0, 3, (batch_size,))
        
        # Evaluate
        metrics, predictions = evaluator.evaluate_single_batch(data, targets)
        
        # Verify results
        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.accuracy >= 0.0
        assert len(metrics.per_class_precision) == 3
        assert predictions is not None
    
    def test_end_to_end_inference_pipeline(self):
        """Test complete inference pipeline."""
        # Create model
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        
        # Create inference engine
        engine = ECGInferenceEngine(
            model=model,
            device='cpu',
            class_names=['Normal', 'Abnormal']
        )
        
        # Create test signal
        signal = np.random.randn(7200)  # 20 seconds at 360 Hz
        
        # Make prediction
        result = engine.predict_signal(signal)
        
        # Verify results
        assert 'error' not in result
        assert result['final_prediction'] in ['Normal', 'Abnormal']
        assert len(result['final_probabilities']) == 2
        assert result['num_windows'] > 0
    
    def test_end_to_end_deployment_pipeline(self):
        """Test complete deployment pipeline."""
        # Create model
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        
        # Export model
        exporter = ModelExporter(model, device='cpu')
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # Export state dict
            exporter.export_state_dict(checkpoint_path, include_metadata=True)
            
            # Create server from checkpoint
            server = ModelServer(
                model=model,
                class_names=['Normal', 'Abnormal'],
                device='cpu'
            )
            
            # Test server functionality
            signal = np.random.randn(1800)
            result = server.predict(signal)
            
            assert result['success'] is True
            assert 'prediction' in result
            
            # Test health check
            health = server.health_check()
            assert health['status'] == 'healthy'
            
        finally:
            Path(checkpoint_path).unlink()
    
    def test_model_export_and_optimization(self):
        """Test model export and optimization workflow."""
        # Create model
        model = FeatureExtractor(embedding_dim=32, pretrained=False, num_classes=2)
        
        # Test export
        exporter = ModelExporter(model, device='cpu')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export TorchScript
            torchscript_path = Path(temp_dir) / "model.pt"
            exporter.export_torchscript(str(torchscript_path))
            assert torchscript_path.exists()
            
            # Test optimization
            optimizer = ModelOptimizer(model, device='cpu')
            quantized_model = optimizer.quantize_dynamic()
            
            # Verify quantized model works
            test_input = torch.randn(1, 1, 224, 224)
            output = quantized_model(test_input)
            assert output.shape == (1, 2)


if __name__ == "__main__":
    pytest.main([__file__])
