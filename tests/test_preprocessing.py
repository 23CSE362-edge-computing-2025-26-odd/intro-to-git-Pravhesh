import numpy as np
import pytest
from ci.preprocess.windowing import SlidingWindowGenerator, WindowConfig
from ci.preprocess.spectrogram import SpectrogramGenerator, SpectrogramConfig
from ci.preprocess.pipeline import PreprocessingPipeline


def test_windowing_basic():
    """Test basic windowing functionality."""
    config = WindowConfig(
        window_seconds=2.0,
        overlap_fraction=0.5,
        sampling_rate_hz=100
    )
    generator = SlidingWindowGenerator(config)
    
    # Test signal: 5 seconds at 100 Hz = 500 samples
    signal = np.random.randn(500)
    windows = generator.generate_windows(signal)
    
    # Should have 3 windows: 0-200, 100-300, 200-400, 300-500
    assert len(windows) == 4
    assert all(len(w) == 200 for w in windows)  # 2 seconds * 100 Hz
    
    # Check overlap
    assert np.allclose(windows[0][100:200], windows[1][0:100])


def test_windowing_short_signal():
    """Test windowing with signal shorter than window."""
    config = WindowConfig(
        window_seconds=2.0,
        overlap_fraction=0.5,
        sampling_rate_hz=100
    )
    generator = SlidingWindowGenerator(config)
    
    # Short signal: 1 second = 100 samples
    signal = np.random.randn(100)
    windows = generator.generate_windows(signal)
    
    # Should pad to window size
    assert len(windows) == 1
    assert len(windows[0]) == 200


def test_spectrogram_basic():
    """Test basic spectrogram generation."""
    config = SpectrogramConfig(
        n_fft=64,
        hop_length=16,
        win_length=64,
        power=2.0,
        mel_scale=False,
        n_mels=128,
        normalize='zscore'
    )
    generator = SpectrogramGenerator(config)
    
    # Test signal: 2 seconds at 100 Hz = 200 samples
    signal = np.random.randn(200)
    spec = generator.generate_spectrogram(signal)
    
    # Check shape: (n_fft//2 + 1, n_frames)
    # librosa uses different frame calculation - let's check actual shape
    assert spec.shape[0] == 33  # 64//2 + 1 = 33
    assert spec.shape[1] > 0  # Should have some frames
    
    # Check normalization (z-score should have mean ~0, std ~1)
    assert abs(np.mean(spec)) < 0.1
    assert abs(np.std(spec) - 1.0) < 0.1


def test_spectrogram_mel_scale():
    """Test spectrogram with Mel scale conversion."""
    config = SpectrogramConfig(
        n_fft=64,
        hop_length=16,
        win_length=64,
        power=2.0,
        mel_scale=True,
        n_mels=32,
        normalize='minmax'
    )
    generator = SpectrogramGenerator(config)
    
    signal = np.random.randn(200)
    spec = generator.generate_spectrogram(signal)
    
    # Check shape: (n_mels, n_frames)
    assert spec.shape[0] == 32  # n_mels
    assert spec.shape[1] > 0  # Should have some frames
    
    # Check minmax normalization (should be in [0, 1])
    assert np.min(spec) >= 0.0
    assert np.max(spec) <= 1.0


def test_pipeline_integration():
    """Test full preprocessing pipeline."""
    window_config = WindowConfig(
        window_seconds=1.0,
        overlap_fraction=0.5,
        sampling_rate_hz=100
    )
    spec_config = SpectrogramConfig(
        n_fft=128,
        hop_length=32,
        win_length=128,
        power=2.0,
        mel_scale=False,
        n_mels=64,
        normalize='zscore'
    )
    
    pipeline = PreprocessingPipeline(window_config, spec_config)
    
    # Test signal: 3 seconds = 300 samples
    signal = np.random.randn(300)
    spectrograms = pipeline.process_signal(signal)
    
    # Should have 5 windows: 0-100, 50-150, 100-200, 150-250, 200-300
    assert len(spectrograms) == 5
    
    # All spectrograms should have same shape
    shapes = [spec.shape for spec in spectrograms]
    assert all(shape == shapes[0] for shape in shapes)
    
    # Check window info
    num_windows, window_size, step_size = pipeline.get_window_info(300)
    assert num_windows == 5
    assert window_size == 100
    assert step_size == 50


def test_deterministic_output():
    """Test that preprocessing is deterministic with same seed."""
    np.random.seed(42)
    signal1 = np.random.randn(200)
    
    np.random.seed(42)
    signal2 = np.random.randn(200)
    
    config = WindowConfig(1.0, 0.5, 100)
    spec_config = SpectrogramConfig(128, 32, 128, 2.0, False, 64, 'zscore')
    pipeline = PreprocessingPipeline(config, spec_config)
    
    specs1 = pipeline.process_signal(signal1)
    specs2 = pipeline.process_signal(signal2)
    
    # Should be identical
    assert len(specs1) == len(specs2)
    for s1, s2 in zip(specs1, specs2):
        assert np.allclose(s1, s2)
