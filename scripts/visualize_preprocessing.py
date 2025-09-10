import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ci.preprocess.windowing import SlidingWindowGenerator, WindowConfig
from src.ci.preprocess.spectrogram import SpectrogramGenerator, SpectrogramConfig
from src.ci.preprocess.pipeline import PreprocessingPipeline


def create_sample_ecg_signal(duration_seconds=10, sampling_rate=360):
    """Create a synthetic ECG-like signal for demonstration."""
    t = np.linspace(0, duration_seconds, int(duration_seconds * sampling_rate))
    
    # Basic ECG components
    heart_rate = 72  # BPM
    rr_interval = 60 / heart_rate  # seconds
    
    # P wave (atrial depolarization)
    p_wave = 0.1 * np.exp(-((t % rr_interval - 0.1) / 0.02) ** 2)
    
    # QRS complex (ventricular depolarization)
    qrs_complex = 0.8 * np.exp(-((t % rr_interval - 0.3) / 0.05) ** 2)
    
    # T wave (ventricular repolarization)
    t_wave = 0.3 * np.exp(-((t % rr_interval - 0.6) / 0.1) ** 2)
    
    # Add some noise
    noise = 0.05 * np.random.randn(len(t))
    
    signal = p_wave + qrs_complex + t_wave + noise
    return t, signal


def visualize_preprocessing_pipeline():
    """Visualize the complete preprocessing pipeline with heatmaps."""
    # Create sample signal
    t, signal = create_sample_ecg_signal(duration_seconds=10, sampling_rate=360)
    
    # Configure preprocessing
    window_config = WindowConfig(
        window_seconds=5.0,
        overlap_fraction=0.5,
        sampling_rate_hz=360
    )
    
    spec_config = SpectrogramConfig(
        n_fft=256,
        hop_length=64,
        win_length=256,
        power=2.0,
        mel_scale=True,
        n_mels=64,
        normalize='zscore'
    )
    
    # Create pipeline
    pipeline = PreprocessingPipeline(window_config, spec_config)
    
    # Process signal
    spectrograms = pipeline.process_signal(signal)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ECG Preprocessing Pipeline Visualization', fontsize=16)
    
    # 1. Original signal
    axes[0, 0].plot(t, signal, 'b-', linewidth=0.8)
    axes[0, 0].set_title('Original ECG Signal')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Windowed signals (show first 3 windows)
    window_generator = SlidingWindowGenerator(window_config)
    windows = window_generator.generate_windows(signal)
    
    for i in range(min(3, len(windows))):
        window_time = np.linspace(0, window_config.window_seconds, len(windows[i]))
        axes[0, 1].plot(window_time, windows[i], alpha=0.7, label=f'Window {i+1}')
    
    axes[0, 1].set_title('Sliding Windows (First 3)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Spectrogram heatmap (first window)
    if spectrograms:
        spec = spectrograms[0]
        im1 = axes[0, 2].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 2].set_title('Spectrogram Heatmap (Window 1)')
        axes[0, 2].set_xlabel('Time Frames')
        axes[0, 2].set_ylabel('Frequency Bins')
        plt.colorbar(im1, ax=axes[0, 2], label='Magnitude')
    
    # 4. All spectrograms overview
    if len(spectrograms) > 1:
        # Stack all spectrograms horizontally
        stacked_specs = np.hstack(spectrograms)
        im2 = axes[1, 0].imshow(stacked_specs, aspect='auto', origin='lower', cmap='plasma')
        axes[1, 0].set_title('All Spectrograms (Concatenated)')
        axes[1, 0].set_xlabel('Time Frames')
        axes[1, 0].set_ylabel('Frequency Bins')
        plt.colorbar(im2, ax=axes[1, 0], label='Magnitude')
    
    # 5. Frequency content analysis
    if spectrograms:
        # Average across time for frequency profile
        freq_profile = np.mean(spectrograms[0], axis=1)
        axes[1, 1].plot(freq_profile, 'g-', linewidth=2)
        axes[1, 1].set_title('Frequency Profile (Window 1)')
        axes[1, 1].set_xlabel('Frequency Bin')
        axes[1, 1].set_ylabel('Average Magnitude')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Statistics summary
    if spectrograms:
        all_values = np.concatenate([spec.flatten() for spec in spectrograms])
        axes[1, 2].hist(all_values, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 2].set_title('Spectrogram Value Distribution')
        axes[1, 2].set_xlabel('Magnitude Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('preprocessing_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nPreprocessing Summary:")
    print(f"Original signal length: {len(signal)} samples ({len(signal)/360:.1f}s)")
    print(f"Number of windows: {len(spectrograms)}")
    print(f"Window size: {window_config.window_seconds}s ({window_config.window_seconds * 360} samples)")
    print(f"Spectrogram shape: {spectrograms[0].shape if spectrograms else 'N/A'}")
    print(f"Overlap: {window_config.overlap_fraction * 100:.0f}%")
    
    return spectrograms


if __name__ == '__main__':
    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Run visualization
    spectrograms = visualize_preprocessing_pipeline()
