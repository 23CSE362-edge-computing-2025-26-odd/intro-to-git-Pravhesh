import numpy as np
from typing import List, Tuple
from .windowing import SlidingWindowGenerator, WindowConfig
from .spectrogram import SpectrogramGenerator, SpectrogramConfig


class PreprocessingPipeline:
    def __init__(self, window_config: WindowConfig, spec_config: SpectrogramConfig):
        self.window_generator = SlidingWindowGenerator(window_config)
        self.spec_generator = SpectrogramGenerator(spec_config)
        
    def process_signal(self, signal: np.ndarray) -> List[np.ndarray]:
        """Process a single signal: windowing -> spectrograms."""
        windows = self.window_generator.generate_windows(signal)
        spectrograms = self.spec_generator.generate_batch(windows)
        return spectrograms
    
    def process_batch(self, signals: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Process a batch of signals."""
        return [self.process_signal(signal) for signal in signals]
    
    def get_expected_output_shape(self, signal_length: int) -> tuple:
        """Get expected spectrogram shape for a given signal length."""
        return self.spec_generator.get_spectrogram_shape(signal_length)
    
    def get_window_info(self, signal_length: int) -> Tuple[int, int, int]:
        """Get windowing information."""
        return self.window_generator.get_window_info(signal_length)
