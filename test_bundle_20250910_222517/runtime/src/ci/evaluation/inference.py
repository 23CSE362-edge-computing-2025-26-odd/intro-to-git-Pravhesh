"""
Inference engine for ECG classification using preprocessing pipeline.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from src.ci.preprocess import PreprocessingPipeline, WindowConfig, SpectrogramConfig


@dataclass
class InferenceConfig:
    window_seconds: float = 5.0
    overlap_fraction: float = 0.5
    sampling_rate_hz: int = 360
    n_fft: int = 256
    hop_length: int = 64
    win_length: int = 256
    power: float = 2.0
    mel_scale: bool = True
    n_mels: int = 64
    normalize: str = 'zscore'


class ECGInferenceEngine:
    """
    Inference engine that combines preprocessing pipeline and model prediction.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inference_config: Optional[InferenceConfig] = None,
        device: str = 'cpu',
        class_names: Optional[List[str]] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.model.eval()
        
        # Build preprocessing pipeline
        if inference_config is None:
            inference_config = InferenceConfig()
        
        window_config = WindowConfig(
            window_seconds=inference_config.window_seconds,
            overlap_fraction=inference_config.overlap_fraction,
            sampling_rate_hz=inference_config.sampling_rate_hz
        )
        
        spec_config = SpectrogramConfig(
            n_fft=inference_config.n_fft,
            hop_length=inference_config.hop_length,
            win_length=inference_config.win_length,
            power=inference_config.power,
            mel_scale=inference_config.mel_scale,
            n_mels=inference_config.n_mels,
            normalize=inference_config.normalize
        )
        
        self.pipeline = PreprocessingPipeline(window_config, spec_config)
    
    def predict_signal(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Predict class for a raw ECG signal.
        
        Args:
            signal: 1D numpy array of ECG values
            
        Returns:
            Dictionary with predictions, probabilities, and window-wise results
        """
        # Process signal through pipeline
        spectrograms = self.pipeline.process_signal(signal)
        
        if len(spectrograms) == 0:
            return {'error': 'No spectrograms generated from input signal'}
        
        # Convert spectrograms to tensor batch
        inputs = [torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0) for spec in spectrograms]
        inputs = torch.cat(inputs, dim=0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # Aggregate window predictions
        probs_np = probabilities.cpu().numpy()
        preds_np = predictions.cpu().numpy()
        
        # Average probabilities across windows
        avg_probs = probs_np.mean(axis=0)
        final_pred = int(np.argmax(avg_probs))
        
        result = {
            'final_prediction': self._class_name(final_pred),
            'final_prediction_index': final_pred,
            'final_probabilities': avg_probs.tolist(),
            'window_predictions': [self._class_name(int(p)) for p in preds_np],
            'window_probabilities': probs_np.tolist(),
            'num_windows': len(spectrograms)
        }
        
        return result
    
    def _class_name(self, idx: int) -> str:
        if self.class_names and 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return f'Class_{idx}'

