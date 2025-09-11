import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class WindowConfig:
    window_seconds: float
    overlap_fraction: float
    sampling_rate_hz: int
    pad_mode: str = 'reflect'


class SlidingWindowGenerator:
    def __init__(self, config: WindowConfig):
        self.config = config
        self.window_samples = int(config.window_seconds * config.sampling_rate_hz)
        self.step_samples = int(self.window_samples * (1 - config.overlap_fraction))
        
    def generate_windows(self, signal: np.ndarray) -> List[np.ndarray]:
        """Generate sliding windows from a 1D signal."""
        if len(signal) < self.window_samples:
            # Pad short signals
            if self.config.pad_mode == 'reflect':
                padded = np.pad(signal, (0, self.window_samples - len(signal)), mode='reflect')
            else:
                padded = np.pad(signal, (0, self.window_samples - len(signal)), mode='constant')
            return [padded]
        
        windows = []
        for start in range(0, len(signal) - self.window_samples + 1, self.step_samples):
            window = signal[start:start + self.window_samples]
            windows.append(window)
        
        # Handle the last window if it doesn't fit exactly
        if len(signal) > self.window_samples:
            last_start = len(signal) - self.window_samples
            if last_start % self.step_samples != 0:
                last_window = signal[last_start:last_start + self.window_samples]
                windows.append(last_window)
        
        return windows
    
    def get_window_info(self, signal_length: int) -> Tuple[int, int, int]:
        """Get number of windows, window size, and step size."""
        if signal_length < self.window_samples:
            return 1, self.window_samples, 0
        
        num_windows = ((signal_length - self.window_samples) // self.step_samples) + 1
        return num_windows, self.window_samples, self.step_samples
