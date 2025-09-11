# CI Project

A hybrid Computational Intelligence system for cardiac condition diagnosis using ResNet18 + Fuzzy Logic + Swarm Optimization, optimized for edge deployment.

## Overview

This project implements a complete pipeline for ECG signal analysis and cardiac diagnosis:
- **Signal Processing**: Sliding window + STFT/Mel spectrogram preprocessing
- **Feature Extraction**: ResNet18-based deep learning for robust feature embeddings
- **Decision Making**: Fuzzy logic system with PSO-optimized parameters
- **Edge Deployment**: ONNX export with INT8 quantization for resource-constrained devices
- **Runtime System**: Production-ready CLI with comprehensive benchmarking

## Performance Results

| Configuration | Latency | Throughput | Model Size | Compression |
|---------------|---------|------------|------------|-------------|
| PyTorch Float32 | 136.9ms | 78,916 samples/sec | ~44MB | 1.0x |
| ONNX Float32 | 137.4ms | 78,575 samples/sec | 43.59MB | 1.0x |
| ONNX INT8 | - | - | 10.95MB | **3.98x** |
| + Fuzzy Logic | +4.9ms | - | - | 3.6% overhead |

## Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision librosa numpy scipy scikit-learn
pip install onnx onnxruntime onnxruntime-tools  # For ONNX support
```

### Basic Usage

```bash
# 1. Create sample ECG data
python scripts/runtime_cli.py --create-sample sample_ecg.csv

# 2. Run inference with PyTorch
python scripts/runtime_cli.py --input sample_ecg.csv --verbose

# 3. Run with ONNX model + fuzzy logic
python scripts/runtime_cli.py --onnx demo_outputs/ecg_model.onnx --input sample_ecg.csv --fuzzy

# 4. Benchmark performance
python scripts/benchmark_runtime.py
```

## Project Structure

```
CI/
├── configs/                    # Configuration files
│   ├── config.yaml            # Main configuration
│   └── labels_pnc.yaml        # Label mapping
├── src/ci/                    # Core implementation
│   ├── data/                  # Data loading utilities
│   ├── preprocess/           # Signal preprocessing
│   ├── model/                # Feature extraction models
│   ├── fuzzy/               # Fuzzy logic engine
│   ├── evaluation/          # Inference and evaluation
│   └── deployment/          # Edge deployment utilities
├── scripts/                  # Executable scripts
│   ├── runtime_cli.py       # Main inference runtime
│   ├── benchmark_runtime.py # Performance benchmarking
│   ├── export_ecg_onnx.py   # ONNX export utility
│   └── ifog_adapter.py      # iFogSim simulation adapter
├── tests/                   # Unit tests
├── demo_outputs/           # Generated models
└── docs/                   # Documentation
```

## Setup & Installation

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd CI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

```bash
# Validate data loading
python scripts/prepare_data.py

# Test preprocessing pipeline
python scripts/visualize_preprocessing.py
```

### 3. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_loader.py -v
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_model.py -v
python -m pytest tests/test_fuzzy.py -v
```

## Training & Evaluation

### Feature Extraction

```bash
# Extract embeddings from ECG signals
python scripts/extract_embeddings.py

# Evaluate linear probe baseline
python scripts/evaluate_linear_probe.py
```

### Fuzzy Logic Optimization

```bash
# Optimize fuzzy parameters with PSO
python scripts/optimize_fuzzy_parameters.py
```

## Edge Deployment

### ONNX Export & Quantization

```bash
# Export ECG-optimized ONNX models
python scripts/export_ecg_onnx.py

# Run complete E6 edge conversion demo
python scripts/demo_e6.py
```

This creates:
- `demo_outputs/ecg_model.onnx` (43.59 MB Float32 model)
- `demo_outputs/ecg_model_int8.onnx` (10.95 MB INT8 model, 3.98x compression)

### Runtime Deployment

```bash
# Single inference
python scripts/runtime_cli.py --input ecg_file.csv --output results.json

# Batch processing with benchmarking
python scripts/runtime_cli.py --input ecg_file.csv --benchmark --verbose

# ONNX model inference
python scripts/runtime_cli.py --onnx demo_outputs/ecg_model.onnx --input ecg_file.csv

# Full pipeline with fuzzy logic
python scripts/runtime_cli.py --input ecg_file.csv --fuzzy --benchmark
```

## Log Computing Simulation

### Generate iFogSim Configuration

```bash
# Create iFogSim deployment profiles
python scripts/ifog_adapter.py
```

This generates:
- `ifog_simulation/device_profiles.json` - Device specifications
- `ifog_simulation/application_modules.json` - Application components
- `ifog_simulation/network_topology.json` - Network configuration
- `ifog_simulation/deployment_scenarios.json` - Deployment strategies
- `ifog_simulation/ECGDiagnosisSimulation.java` - iFogSim template

### Deployment Scenarios

| Scenario | Description | Use Case |
|----------|-------------|----------|
| **All Edge** | Process on edge gateway | Low latency, network independence |
| **Edge-Fog Hybrid** | Preprocessing at edge, inference at fog | Balance latency/resources |
| **Fog Only** | All processing at fog node | Centralized processing |
| **Cloud Only** | All processing in cloud | Maximum compute power |

## 📊 Performance Benchmarking

### Comprehensive Benchmarking

```bash
# Run full benchmark suite
python scripts/benchmark_runtime.py
```

**Sample Results:**

```
Configuration                    | Mean Latency | P95 Latency | Throughput | Speedup
--------------------------------------------------------------------------------
PyTorch Float32                  |      136.9ms |     164.5ms |    78916/s |  1.00x
PyTorch Float32 + Fuzzy          |      141.8ms |     161.1ms |    76173/s |  0.97x
ONNX Float32                     |      137.4ms |     160.8ms |    78575/s |  1.00x
ONNX Float32 + Fuzzy             |      146.5ms |     171.7ms |    73712/s |  0.93x

Fuzzy Logic Overhead: +4.9ms (3.6%)
ONNX Speedup: 1.00x faster than PyTorch
```

## Architecture
 
See: [Architecture Flowcharts](docs/architecture_flowcharts.md) for complete Mermaid diagrams (pipeline, deployment, E6).

### Signal Processing Pipeline

```
ECG Signal (30s @ 360Hz) 
    ↓ [Sliding Window: 10s, 50% overlap]
Windowed Segments (5 windows)
    ↓ [STFT + Mel Spectrogram: 64 mels]
Spectrograms (5 × 64×57)
    ↓ [ResNet18 Feature Extractor]
Feature Embeddings (5 × 512-d)
    ↓ [Aggregation + Classification]
Class Probabilities (5 classes)
    ↓ [Fuzzy Logic Decision Engine]
Final Diagnosis + Risk Score
```

### Fuzzy Logic System

- **Membership Functions**: Triangular, Trapezoidal, Gaussian
- **Inference Method**: Mamdani with centroid defuzzification  
- **Optimization**: PSO with 30 particles, early stopping
- **Rules**: 5 expert-defined rules for cardiac conditions

### Edge Deployment Architecture

```
ECG Sensor → Edge Gateway → Fog Node → Cloud Server
     │            │            │           │
   Data         Preproc     Inference   Analytics
Collection   Spectrogram   Deep+Fuzzy   Long-term
```

## Model Details

### ResNet18 Feature Extractor

- **Architecture**: Modified ResNet18 with single-channel input
- **Input**: Mel spectrograms (1×64×57)
- **Output**: 512-dimensional embeddings
- **Modifications**: 
  - First conv layer adapted for single-channel
  - Optional classification head
  - Frozen layers except last block for transfer learning

### Fuzzy Decision Engine

- **Input Features**: Aggregated deep learning embeddings
- **Output**: Risk scores and diagnostic categories
- **Rules**: Expert knowledge encoded as if-then rules
- **Optimization**: PSO-tuned membership functions

## Validation Results

### Core Pipeline Tests
- ✅ **Data Loading**: PTB (549 files) + MIT-BIH (48 records)
- ✅ **Preprocessing**: 6/6 tests passing (windowing, spectrogram, normalization)
- ✅ **Feature Extraction**: Model architecture and inference validated
- ✅ **Fuzzy Logic**: Rule firing and boundary conditions tested
- ✅ **Edge Conversion**: ONNX export with <1e-4 parity difference

### Performance Validation
- ✅ **Latency**: 136.9ms end-to-end processing (30s ECG signal)
- ✅ **Throughput**: 78,916 samples/sec sustained processing
- ✅ **Compression**: 3.98x model size reduction with INT8 quantization
- ✅ **Accuracy**: Model parity maintained across PyTorch/ONNX formats

## 🛠️ Advanced Usage

### Custom Model Training

```python
from src.ci.model.feature_extractor import FeatureExtractor
from src.ci.fuzzy.engine import FuzzyDecisionEngine

# Create custom feature extractor
model = FeatureExtractor(
    embedding_dim=512,
    pretrained=True,
    num_classes=5  # For classification
)

# Custom fuzzy engine
fuzzy_engine = FuzzyDecisionEngine()
```

### Batch Processing

```python
from scripts.runtime_cli import ECGRuntimeEngine

# Initialize runtime
engine = ECGRuntimeEngine(
    onnx_path="demo_outputs/ecg_model.onnx",
    fuzzy_config_path="configs/fuzzy_config.yaml"
)

# Process multiple signals
for signal_file in signal_files:
    signal = load_ecg_data(signal_file)
    result = engine.predict(signal)
    print(f"Diagnosis: {result['prediction']}")
```

## Configuration

### Model Configuration (`configs/config.yaml`)

```yaml
model:
  backbone: "resnet18"
  embedding_dim: 512
  pretrained: true
  num_classes: 5

preprocessing:
  window_seconds: 10.0
  overlap_fraction: 0.5
  sampling_rate_hz: 360
  n_fft: 256
  n_mels: 64
  normalize: "zscore"

fuzzy:
  membership_functions: ["triangular", "gaussian"]
  num_rules: 5
  defuzzification: "centroid"
```

### Label Mapping (`configs/labels_pnc.yaml`)

```yaml
labels:
  0: "Normal"
  1: "Arrhythmia" 
  2: "Myocardial_Infarction"
  3: "Heart_Block"
  4: "Other"
```

## Troubleshooting

### Common Issues

**ONNX Runtime Not Found**
```bash
pip install onnxruntime
```

**Model Loading Errors**
```bash
# Regenerate ONNX models
python scripts/export_ecg_onnx.py
```

**Performance Issues**
```bash
# Check system resources
python scripts/benchmark_runtime.py --verbose
```

### Debug Mode

```bash
# Enable verbose logging
python scripts/runtime_cli.py --input data.csv --verbose

# Run with debugging
PYTHONPATH=src python -m pdb scripts/runtime_cli.py --input data.csv
```
