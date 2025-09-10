import numpy as np
import librosa
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SpectrogramConfig:
    n_fft: int
    hop_length: int
    win_length: int
    power: float
    mel_scale: bool
    n_mels: int
    normalize: str  # 'zscore', 'minmax', 'none'


class SpectrogramGenerator:
    def __init__(self, config: SpectrogramConfig):
        self.config = config
        
    def generate_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        """Generate spectrogram from 1D signal."""
        # STFT
        stft = librosa.stft(
            signal,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        
        # Convert to magnitude and apply power
        magnitude = np.abs(stft)
        if self.config.power != 1.0:
            magnitude = magnitude ** self.config.power
        
        # Optional Mel scale conversion
        if self.config.mel_scale:
            mel_spec = librosa.feature.melspectrogram(
                S=magnitude,
                n_mels=self.config.n_mels,
                fmax=8000  # Typical for ECG
            )
            spectrogram = mel_spec
        else:
            spectrogram = magnitude
        
        # Normalize
        if self.config.normalize == 'zscore':
            spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
        elif self.config.normalize == 'minmax':
            spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
        
        return spectrogram
    
    def generate_batch(self, signals: List[np.ndarray]) -> List[np.ndarray]:
        """Generate spectrograms for a batch of signals."""
        return [self.generate_spectrogram(signal) for signal in signals]
    
    def get_spectrogram_shape(self, signal_length: int) -> tuple:
        """Get expected spectrogram shape for a given signal length."""
        n_frames = 1 + (signal_length - self.config.n_fft) // self.config.hop_length
        if self.config.mel_scale:
            return (self.config.n_mels, n_frames)
        else:
            return (self.config.n_fft // 2 + 1, n_frames)
