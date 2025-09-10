"""
Simple model server for deployment.
"""

from typing import Dict, Any, Optional, List
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from src.ci.evaluation.inference import ECGInferenceEngine, InferenceConfig


class ModelServer:
    """
    Simple model server for serving ECG classification models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inference_config: Optional[InferenceConfig] = None,
        class_names: Optional[List[str]] = None,
        device: str = 'cpu'
    ):
        """
        Initialize model server.
        
        Args:
            model: Trained PyTorch model
            inference_config: Configuration for inference pipeline
            class_names: List of class names
            device: Device to run inference on
        """
        self.inference_engine = ECGInferenceEngine(
            model=model,
            inference_config=inference_config,
            device=device,
            class_names=class_names
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Server statistics
        self.stats = {
            'requests_processed': 0,
            'successful_predictions': 0,
            'failed_predictions': 0
        }
    
    def predict(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction on ECG signal.
        
        Args:
            signal_data: 1D numpy array of ECG values
            
        Returns:
            Prediction results dictionary
        """
        try:
            self.stats['requests_processed'] += 1
            
            # Validate input
            if not isinstance(signal_data, np.ndarray):
                signal_data = np.array(signal_data)
            
            if signal_data.ndim != 1:
                raise ValueError("Signal data must be 1-dimensional")
            
            # Make prediction
            result = self.inference_engine.predict_signal(signal_data)
            
            if 'error' not in result:
                self.stats['successful_predictions'] += 1
            else:
                self.stats['failed_predictions'] += 1
            
            return {
                'success': True,
                'prediction': result,
                'input_shape': signal_data.shape,
                'request_id': self.stats['requests_processed']
            }
            
        except Exception as e:
            self.stats['failed_predictions'] += 1
            self.logger.error(f"Prediction failed: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'request_id': self.stats['requests_processed']
            }
    
    def batch_predict(self, signals: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple ECG signals.
        
        Args:
            signals: List of 1D numpy arrays
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, signal in enumerate(signals):
            try:
                result = self.predict(signal)
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Batch prediction failed for index {i}: {str(e)}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'batch_index': i,
                    'request_id': self.stats['requests_processed']
                })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.
        
        Returns:
            Dictionary with server statistics
        """
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_predictions'] / max(1, self.stats['requests_processed'])
            ) * 100
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status dictionary
        """
        try:
            # Test with dummy signal
            dummy_signal = np.random.randn(1000)  # 1000 samples
            result = self.predict(dummy_signal)
            
            is_healthy = result['success']
            
            return {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'model_loaded': True,
                'inference_engine_ready': True,
                'test_prediction_success': is_healthy,
                'stats': self.get_stats()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_loaded': False,
                'inference_engine_ready': False
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        model = self.inference_engine.model
        
        return {
            'model_class': model.__class__.__name__,
            'device': str(self.inference_engine.device),
            'class_names': self.inference_engine.class_names,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_config': getattr(model, 'get_config', lambda: {})(),
            'inference_config': {
                'window_seconds': self.inference_engine.pipeline.window_config.window_seconds,
                'overlap_fraction': self.inference_engine.pipeline.window_config.overlap_fraction,
                'sampling_rate_hz': self.inference_engine.pipeline.window_config.sampling_rate_hz,
                'n_fft': self.inference_engine.pipeline.spec_config.n_fft,
                'normalize': self.inference_engine.pipeline.spec_config.normalize
            }
        }


def create_server_from_checkpoint(
    checkpoint_path: str,
    model_class: type,
    model_kwargs: Optional[Dict[str, Any]] = None,
    inference_config: Optional[InferenceConfig] = None,
    class_names: Optional[List[str]] = None,
    device: str = 'cpu'
) -> ModelServer:
    """
    Create model server from checkpoint file.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model initialization
        inference_config: Configuration for inference pipeline
        class_names: List of class names
        device: Device to run inference on
        
    Returns:
        ModelServer instance
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create and load model
    model = model_class(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Create server
    server = ModelServer(
        model=model,
        inference_config=inference_config,
        class_names=class_names,
        device=device
    )
    
    return server


def save_server_config(
    server: ModelServer,
    config_path: str
):
    """
    Save server configuration to file.
    
    Args:
        server: ModelServer instance
        config_path: Path to save configuration
    """
    config = {
        'model_info': server.get_model_info(),
        'stats': server.get_stats(),
        'health': server.health_check()
    }
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logging.getLogger(__name__).info(f"Server config saved to: {config_path}")


# Example usage and testing
if __name__ == "__main__":
    # This would typically be used in a web service or API
    logging.basicConfig(level=logging.INFO)
    
    print("ModelServer class is ready for deployment!")
    print("Integration with web frameworks like Flask or FastAPI recommended for production use.")
