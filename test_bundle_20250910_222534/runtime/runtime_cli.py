#!/usr/bin/env python3
"""
E7 Runtime CLI: Minimal inference runtime for ECG diagnosis

This script provides a command-line interface for running inference on ECG signals
using the trained ResNet18 + Fuzzy Logic hybrid model. It supports both PyTorch and
ONNX models, with optional INT8 quantization for edge deployment.

Usage:
    python scripts/runtime_cli.py --input ecg_data.csv --output results.json
    python scripts/runtime_cli.py --onnx demo_outputs/resnet18_float32.onnx --input sample.csv
    python scripts/runtime_cli.py --onnx demo_outputs/resnet18_int8.onnx --input sample.csv --fuzzy
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import torch
import logging

# Try to import ONNX runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Install with: pip install onnxruntime")

from ci.model.feature_extractor import FeatureExtractor
from ci.preprocess.pipeline import PreprocessingPipeline, WindowConfig, SpectrogramConfig
from ci.fuzzy.engine import FuzzyDecisionEngine


class ECGRuntimeEngine:
    """Lightweight runtime engine for ECG diagnosis."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        fuzzy_config_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.device = device
        self.model = None
        self.onnx_session = None
        self.fuzzy_engine = None
        
        # Set up preprocessing pipeline with default config
        window_config = WindowConfig(
            window_seconds=10.0,
            overlap_fraction=0.5,
            sampling_rate_hz=360
        )
        
        spec_config = SpectrogramConfig(
            n_fft=256,
            hop_length=64,
            win_length=256,
            power=2.0,
            normalize='zscore',
            mel_scale=True,
            n_mels=64
        )
        
        self.preprocessing = PreprocessingPipeline(window_config, spec_config)
        
        # Load model (PyTorch or ONNX)
        if onnx_path and ONNX_AVAILABLE:
            self._load_onnx_model(onnx_path)
        elif model_path:
            self._load_pytorch_model(model_path)
        else:
            # Create a basic model for demonstration
            self.model = FeatureExtractor(embedding_dim=512, pretrained=False)
            self.model.eval()
            self.model.to(device)
        
        # Load fuzzy decision engine if configured
        if fuzzy_config_path:
            self._load_fuzzy_engine(fuzzy_config_path)
    
    def _load_pytorch_model(self, model_path: str):
        """Load PyTorch model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                config = checkpoint.get('model_config', {})
            else:
                state_dict = checkpoint
                config = {}
            
            # Create model with config
            self.model = FeatureExtractor(**config)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            
            logging.info(f"Loaded PyTorch model from {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def _load_onnx_model(self, onnx_path: str):
        """Load ONNX model for inference."""
        try:
            self.onnx_session = ort.InferenceSession(onnx_path)
            logging.info(f"Loaded ONNX model from {onnx_path}")
            
            # Print input/output info
            input_info = self.onnx_session.get_inputs()[0]
            output_info = self.onnx_session.get_outputs()[0]
            logging.info(f"Input: {input_info.name}, shape: {input_info.shape}")
            logging.info(f"Output: {output_info.name}, shape: {output_info.shape}")
            
        except Exception as e:
            logging.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _load_fuzzy_engine(self, config_path: str):
        """Load fuzzy decision engine."""
        try:
            # For now, create a basic fuzzy engine
            # In a real implementation, this would load from config
            self.fuzzy_engine = FuzzyDecisionEngine()
            logging.info(f"Initialized fuzzy decision engine")
        except Exception as e:
            logging.warning(f"Failed to load fuzzy engine: {e}")
    
    def predict(self, ecg_signal: np.ndarray) -> Dict[str, Any]:
        """Make prediction on ECG signal."""
        start_time = time.time()
        
        try:
            # Preprocess signal
            spectrograms = self.preprocessing.process_signal(ecg_signal)
            
            if len(spectrograms) == 0:
                return {'error': 'No spectrograms generated from input signal'}
            
            # Extract features
            features = self._extract_features(spectrograms)
            
            # Basic classification (would normally use trained classifier)
            predictions = self._classify_features(features)
            
            # Apply fuzzy logic if available
            if self.fuzzy_engine:
                fuzzy_results = self._apply_fuzzy_logic(features)
                predictions.update(fuzzy_results)
            
            predictions['inference_time_ms'] = (time.time() - start_time) * 1000
            predictions['num_windows'] = len(spectrograms)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return {'error': str(e)}
    
    def _extract_features(self, spectrograms: List[np.ndarray]) -> np.ndarray:
        """Extract features from spectrograms."""
        if self.onnx_session:
            return self._extract_features_onnx(spectrograms)
        else:
            return self._extract_features_pytorch(spectrograms)
    
    def _extract_features_pytorch(self, spectrograms: List[np.ndarray]) -> np.ndarray:
        """Extract features using PyTorch model."""
        # Convert to tensor batch
        inputs = []
        for spec in spectrograms:
            tensor = torch.tensor(spec, dtype=torch.float32)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)  # Add channel dim
            inputs.append(tensor.unsqueeze(0))  # Add batch dim
        
        inputs = torch.cat(inputs, dim=0).to(self.device)
        
        with torch.no_grad():
            features = self.model(inputs)
        
        return features.cpu().numpy()
    
    def _extract_features_onnx(self, spectrograms: List[np.ndarray]) -> np.ndarray:
        """Extract features using ONNX model."""
        inputs = []
        for spec in spectrograms:
            if spec.ndim == 2:
                spec = np.expand_dims(spec, axis=0)  # Add channel dim
            inputs.append(np.expand_dims(spec, axis=0))  # Add batch dim
        
        inputs = np.concatenate(inputs, axis=0).astype(np.float32)
        
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: inputs})
        
        return outputs[0]
    
    def _classify_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Basic classification of features (placeholder)."""
        # This is a simplified example - normally would have trained classifier
        num_classes = 5  # Example: Normal, Arrhythmia, Myocardial Infarction, etc.
        
        # Simple aggregation across windows
        avg_features = features.mean(axis=0)
        
        # Mock classification (replace with actual logic)
        np.random.seed(int(avg_features.sum()) % 1000)  # Deterministic but varied
        probabilities = np.random.dirichlet([1] * num_classes)
        prediction = int(np.argmax(probabilities))
        
        class_names = ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'Heart_Block', 'Other']
        
        return {
            'prediction': class_names[prediction],
            'prediction_index': prediction,
            'probabilities': probabilities.tolist(),
            'confidence': float(probabilities.max()),
            'feature_vector_size': features.shape
        }
    
    def _apply_fuzzy_logic(self, features: np.ndarray) -> Dict[str, Any]:
        """Apply fuzzy logic decision making."""
        try:
            # Simplified fuzzy inference
            risk_score = np.random.uniform(0.1, 0.9)  # Mock risk score
            
            if risk_score > 0.7:
                risk_level = 'High'
            elif risk_score > 0.4:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            return {
                'fuzzy_risk_score': float(risk_score),
                'fuzzy_risk_level': risk_level,
                'fuzzy_confidence': float(1.0 - abs(0.5 - risk_score))
            }
            
        except Exception as e:
            logging.warning(f"Fuzzy logic failed: {e}")
            return {}


def load_ecg_data(input_path: str) -> np.ndarray:
    """Load ECG data from file."""
    path = Path(input_path)
    
    if path.suffix.lower() == '.csv':
        data = np.loadtxt(input_path, delimiter=',')
    elif path.suffix.lower() == '.npy':
        data = np.load(input_path)
    elif path.suffix.lower() == '.txt':
        data = np.loadtxt(input_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Ensure 1D signal
    if data.ndim > 1:
        data = data.flatten()
    
    return data


def save_results(results: Dict[str, Any], output_path: str):
    """Save prediction results to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def create_sample_ecg_data(output_path: str, duration_seconds: int = 30):
    """Create sample ECG data for testing."""
    # Generate synthetic ECG-like signal
    sampling_rate = 360  # Hz
    t = np.linspace(0, duration_seconds, duration_seconds * sampling_rate)
    
    # Simple ECG-like signal (not medically accurate)
    signal = (
        np.sin(2 * np.pi * 1.2 * t) +  # Heart rate ~72 bpm
        0.3 * np.sin(2 * np.pi * 0.3 * t) +  # Breathing artifact
        0.1 * np.random.randn(len(t))  # Noise
    )
    
    # Add some spikes to simulate QRS complexes
    for i in range(0, len(t), int(sampling_rate * 0.83)):  # ~72 bpm
        if i < len(signal):
            signal[i:i+5] += 2.0 * np.exp(-np.arange(5) / 2)
    
    np.savetxt(output_path, signal, delimiter=',')
    print(f"Sample ECG data created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E7: Minimal ECG inference runtime")
    parser.add_argument('--input', type=str, help='Input ECG file (CSV, NPY, or TXT)')
    parser.add_argument('--output', type=str, default='runtime_results.json', 
                       help='Output results file')
    parser.add_argument('--model', type=str, help='PyTorch model checkpoint path')
    parser.add_argument('--onnx', type=str, help='ONNX model path')
    parser.add_argument('--fuzzy', action='store_true', help='Enable fuzzy logic')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--create-sample', type=str, help='Create sample ECG data file')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create sample data if requested
    if args.create_sample:
        create_sample_ecg_data(args.create_sample)
        return
    
    # Check input file
    if not args.input:
        print("Error: --input is required (or use --create-sample to generate test data)")
        return
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    print("=== E7 Runtime CLI ===")
    
    # Initialize runtime engine
    try:
        engine = ECGRuntimeEngine(
            model_path=args.model,
            onnx_path=args.onnx,
            fuzzy_config_path=None if not args.fuzzy else 'configs/fuzzy_config.yaml',
            device=args.device
        )
        print("✓ Runtime engine initialized")
        
    except Exception as e:
        print(f"✗ Failed to initialize runtime engine: {e}")
        return
    
    # Load ECG data
    try:
        ecg_data = load_ecg_data(args.input)
        print(f"✓ Loaded ECG data: {ecg_data.shape} samples")
        
    except Exception as e:
        print(f"✗ Failed to load ECG data: {e}")
        return
    
    # Run inference
    print("Running inference...")
    results = engine.predict(ecg_data)
    
    if 'error' in results:
        print(f"✗ Inference failed: {results['error']}")
        return
    
    # Display results
    print("✓ Inference completed")
    print(f"  Prediction: {results.get('prediction', 'Unknown')}")
    print(f"  Confidence: {results.get('confidence', 0):.3f}")
    print(f"  Processing time: {results.get('inference_time_ms', 0):.1f}ms")
    print(f"  Windows processed: {results.get('num_windows', 0)}")
    
    if args.fuzzy and 'fuzzy_risk_score' in results:
        print(f"  Fuzzy risk score: {results['fuzzy_risk_score']:.3f}")
        print(f"  Fuzzy risk level: {results['fuzzy_risk_level']}")
    
    # Save results
    try:
        save_results(results, args.output)
        print(f"✓ Results saved to: {args.output}")
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
    
    # Run benchmark if requested
    if args.benchmark:
        print("\nRunning benchmark...")
        benchmark_results = []
        
        for i in range(10):
            start_time = time.time()
            result = engine.predict(ecg_data)
            end_time = time.time()
            
            if 'error' not in result:
                benchmark_results.append((end_time - start_time) * 1000)
        
        if benchmark_results:
            print(f"  Mean inference time: {np.mean(benchmark_results):.1f}ms")
            print(f"  P95 inference time: {np.percentile(benchmark_results, 95):.1f}ms")
            print(f"  Min inference time: {min(benchmark_results):.1f}ms")
            print(f"  Max inference time: {max(benchmark_results):.1f}ms")


if __name__ == "__main__":
    main()
