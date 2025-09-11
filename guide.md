# CI Medical Diagnosis Project - Comprehensive Guide (guide.md)

This guide provides a step-by-step walkthrough for using the CI Medical Diagnosis Project:
- How to set up the environment and project
- How to train and evaluate the model
- How to export and deploy models for edge/ONNX
- How to run and operate the runtime/CLI
- How to simulate deployment using iFogSim
- How to package an edge deployment bundle

If you get stuck, check README.md and scripts/ help messages.

---

## 1) Prerequisites & Setup

- Python 3.8+ (recommended: 3.10/3.11)
- Windows, macOS, or Linux
- Optional (for iFogSim): JDK 8+, Maven/Gradle, iFogSim repo

Install Python dependencies:

```bash
# Core dependencies
pip install torch torchvision librosa numpy scipy scikit-learn pyyaml

# ONNX for edge export/runtime
pip install onnx onnxruntime onnxruntime-tools

# Optional: plotting for simulation charts
pip install matplotlib seaborn

# Dev & tests
pip install pytest
```

Project layout reference:

```
CI/
├── src/ci/                    # Core source (data, preprocess, model, fuzzy, deployment, evaluation)
├── scripts/                   # Executable scripts
│   ├── runtime_cli.py         # Inference runtime (PyTorch/ONNX)
│   ├── benchmark_runtime.py   # Performance benchmarks
│   ├── export_ecg_onnx.py     # ECG-optimized ONNX export
│   ├── demo_e6.py             # E6 demo (export/quantize/parity/latency)
│   ├── ifog_adapter.py        # iFogSim adapter (exports configs)
│   ├── deployment_simulation.py # Scenario analysis (latency/cost/power)
│   ├── package_edge_bundle.py # Build edge bundle
│   ├── train_model.py         # Training script (if present)
│   ├── extract_embeddings.py  # Feature extraction
│   └── evaluate_linear_probe.py# Linear probe baseline
├── configs/                   # YAML configs (model, labels)
├── demo_outputs/              # Exported ONNX models
├── ifog_simulation/           # Generated iFogSim configs
├── tests/                     # Unit tests
├── README.md                  # Documentation
└── guide.md                   # This guide
```

---

## 2) Data Preparation

- Supported datasets: PTB, MIT-BIH (via loaders in src/ci/data)
- Validate data & preprocessing:

```bash
# Validate data loading
python scripts/prepare_data.py

# Visualize preprocessing and spectrograms
python scripts/visualize_preprocessing.py
```

---

## 3) Training the Model

There are two typical flows:
- Feature extraction + linear probe baseline
- End-to-end training (if train script/model is configured)

A) Extract embeddings + Linear probe

```bash
# Extract embeddings from ECG spectrograms (adjust paths inside script/config)
python scripts/extract_embeddings.py

# Train/evaluate a linear probe classifier on the embeddings
python scripts/evaluate_linear_probe.py
```

B) End-to-end training (if available)

```bash
# Train the model end-to-end
# Check scripts/train_model.py for arguments (epochs, lr, batch_size, data dirs)
python scripts/train_model.py --config configs/config.yaml --epochs 20 --batch-size 32

# Evaluate a checkpoint
python scripts/evaluate_model.py --checkpoint outputs/checkpoints/best.pth
```

Notes:
- Model factory/layers live in src/ci/model
- Fuzzy decision engine in src/ci/fuzzy with PSO optimizer in src/ci/fuzzy/optimize.py

---

## 4) Export to ONNX and Quantize (Edge)

There are two options:

1) Quick ECG-optimized export

```bash
# Creates demo_outputs/ecg_model.onnx and INT8 version
python scripts/export_ecg_onnx.py
```

2) Full E6 pipeline demo (export + quantize + parity + benchmarking)

```bash
python scripts/demo_e6.py
```

Artifacts:
- demo_outputs/ecg_model.onnx (Float32)
- demo_outputs/ecg_model_int8.onnx (INT8, ~3.98x smaller)

---

## 5) Operating the Runtime (Inference CLI)

Run the runtime with either PyTorch or ONNX models.

Create sample ECG data:

```bash
python scripts/runtime_cli.py --create-sample sample_ecg.csv
```

Run inference:

```bash
# PyTorch mode (default)
python scripts/runtime_cli.py --input sample_ecg.csv --verbose

# ONNX Float32 model
python scripts/runtime_cli.py --onnx demo_outputs/ecg_model.onnx --input sample_ecg.csv

# With fuzzy decision engine
a) Ensure configs/fuzzy_config.yaml exists (optional)
b) python scripts/runtime_cli.py --input sample_ecg.csv --fuzzy --benchmark
```

Benchmark runtime end-to-end:

```bash
python scripts/benchmark_runtime.py
```

Outputs: probabilities, predicted class, optional fuzzy risk score, latency metrics.

---

## 6) Simulate Deployment with iFogSim

This project generates iFogSim-compatible profiles and a Java template.

Step 1: Generate iFogSim configuration from benchmarked results

```bash
python scripts/ifog_adapter.py
```

This creates the folder:

```
ifog_simulation/
├── device_profiles.json
├── application_modules.json
├── network_topology.json
├── deployment_scenarios.json
├── workload_profile.json
├── devices.csv
├── latency_matrix.csv
├── processing_requirements.csv
└── ECGDiagnosisSimulation.java   # Template
```

Step 2: Prepare iFogSim environment (Java)

- Install JDK 8+ and Maven/Gradle
- Clone iFogSim repository
- Import into your IDE (Eclipse/IntelliJ) or build via Maven

Step 3: Integrate the generated Java template

- Copy ifog_simulation/ECGDiagnosisSimulation.java into an iFogSim example module/package
- In the Java template, replace the placeholder loading sections with JSON parsing of:
  - device_profiles.json
  - application_modules.json
  - network_topology.json
  - deployment_scenarios.json
  - workload_profile.json
- Map them to iFogSim types (FogDevice, Application, Edges, ModuleMapping)
- Choose a scenario from deployment_scenarios.json (e.g., "Edge_Fog_Hybrid") to populate ModuleMapping

Pseudo-code inside iFogSim (Java):

```java
// Load JSON files (use Gson/Jackson)
DeviceProfiles profiles = load("device_profiles.json");
ApplicationModules modules = load("application_modules.json");
NetworkTopology topology = load("network_topology.json");
Scenarios scenarios = load("deployment_scenarios.json");
Workload workload = load("workload_profile.json");

// Create FogDevices from device profiles
List<FogDevice> fogDevices = createDevices(profiles);

// Build Application from modules
Application app = createApplication(modules);

// Define data flows (edges) based on pipeline
addAppEdges(app, topology);

// Choose a scenario and create ModuleMapping
ModuleMapping mapping = mappingFromScenario(scenarios.get("Edge_Fog_Hybrid"));

// Submit application
FogBroker broker = new FogBroker("ECG_Broker");
broker.submitApplication(app, new ModulePlacementEdgewards(
    fogDevices, sensors, actuators, app, mapping));

CloudSim.startSimulation();
```

Step 4: Run iFogSim simulation

- Build & run from IDE or via Maven
- Observe latency, resource utilization, and energy metrics
- Try multiple scenarios from deployment_scenarios.json

Tip: You can also use the CSVs (devices.csv, latency_matrix.csv, processing_requirements.csv) with your own simulators.

---

## 7) Deployment Scenario Analysis (Python Simulator)

For quick what-if analysis without Java:

```bash
python scripts/deployment_simulation.py --duration 60 --arrival-rate 6 --visualization
```

Outputs (simulation_results/):
- simulation_results.json (full results)
- scenario_comparison.csv, device_utilization.csv
- create_charts.py (optional chart generator)

---

## 8) Package Edge Deployment Bundle

Create a self-contained bundle for edge devices:

```bash
# Creates ecg_edge_bundle_<timestamp>/ and zip archive
python scripts/package_edge_bundle.py --name ecg_edge_bundle
```

Bundle contents:
- models/: ecg_model_float32.onnx, ecg_model_int8.onnx
- configs/: main config, labels, edge_config.yaml
- runtime/: runtime_cli, benchmarks, and required modules
- data/: sample_ecg.csv
- docs/: DEPLOYMENT_GUIDE.md
- edge_launcher.py (auto-selects model based on RAM)
- requirements_minimal.txt

Deploy on device:

```bash
# On the device
unzip ecg_edge_bundle_<timestamp>.zip
cd ecg_edge_bundle_<timestamp>

# Install minimal deps
pip install -r requirements_minimal.txt

# Run diagnosis
python edge_launcher.py --input data/sample_ecg.csv --output results.json
```

---

## 9) Troubleshooting & Tips

- ONNX Runtime missing: `pip install onnxruntime`
- Large differences after export: ensure same preprocessing and spectrogram shape (1×64×T)
- ONNX INT8 not supported for some ops: use Float32 ONNX or PyTorch
- Windows Unicode errors: use UTF-8 encoding (fixed in bundle packager)
- Performance: use ONNX Float32 for portability; INT8 for small devices (if supported)

---

## 10) Quick Command Reference

Training & Eval
```bash
python scripts/extract_embeddings.py
python scripts/evaluate_linear_probe.py
python scripts/train_model.py --config configs/config.yaml  # if used
```

Export & Quantize
```bash
python scripts/export_ecg_onnx.py
python scripts/demo_e6.py
```

Runtime & Benchmarks
```bash
python scripts/runtime_cli.py --create-sample sample.csv
python scripts/runtime_cli.py --input sample.csv --fuzzy --benchmark
python scripts/runtime_cli.py --onnx demo_outputs/ecg_model.onnx --input sample.csv
python scripts/benchmark_runtime.py
```

iFogSim & Simulation
```bash
python scripts/ifog_adapter.py
python scripts/deployment_simulation.py --duration 60 --arrival-rate 6 --visualization
```

Packaging
```bash
python scripts/package_edge_bundle.py --name ecg_edge_bundle
```

---

## 11) FAQ

- Q: Which input shape does the model expect?
  - A: Single-channel Mel spectrograms of size 1×64×T (T is variable). Runtime handles batching.
- Q: Can I substitute another backbone?
  - A: Yes, update src/ci/model/feature_extractor.py and config as needed.
- Q: How do I enable fuzzy logic?
  - A: Provide configs/fuzzy_config.yaml (optional) and pass `--fuzzy` to runtime_cli.py.
- Q: What is the preferred deployment path?
  - A: ONNX Float32 for compatibility; try INT8 where ConvInteger kernels are supported.

---

Happy building and deploying! 🚀

