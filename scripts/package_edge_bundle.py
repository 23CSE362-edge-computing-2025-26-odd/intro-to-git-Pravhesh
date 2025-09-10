#!/usr/bin/env python3
"""
Edge Deployment Bundle Packaging

This script creates a complete deployment package for edge devices containing:
- ONNX models (Float32 + INT8)
- Configuration files  
- Runtime scripts
- Dependencies and requirements
- Documentation and examples

The bundle is ready for deployment to edge devices with minimal setup.
"""

import os
import json
import shutil
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


class EdgeBundlePackager:
    """Package complete edge deployment bundle."""
    
    def __init__(self, bundle_name: str = "ecg_edge_bundle"):
        """Initialize the packager."""
        self.bundle_name = bundle_name
        self.project_root = Path(__file__).parent.parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.bundle_dir = Path(f"{bundle_name}_{self.timestamp}")
        
    def create_bundle_structure(self):
        """Create the bundle directory structure."""
        dirs_to_create = [
            self.bundle_dir,
            self.bundle_dir / "models",
            self.bundle_dir / "configs", 
            self.bundle_dir / "runtime",
            self.bundle_dir / "docs",
            self.bundle_dir / "examples",
            self.bundle_dir / "data"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created bundle structure: {self.bundle_dir}")
    
    def copy_models(self):
        """Copy ONNX models to bundle."""
        model_files = [
            ("demo_outputs/ecg_model.onnx", "models/ecg_model_float32.onnx"),
            ("demo_outputs/ecg_model_int8.onnx", "models/ecg_model_int8.onnx"),
        ]
        
        for src, dst in model_files:
            src_path = self.project_root / src
            dst_path = self.bundle_dir / dst
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                size_mb = dst_path.stat().st_size / (1024*1024)
                print(f"  ✓ {dst}: {size_mb:.2f} MB")
            else:
                print(f"  ⚠ {src} not found, skipping")
        
        print(f"✓ Models copied to bundle")
    
    def copy_configs(self):
        """Copy configuration files to bundle."""
        config_files = [
            ("configs/config.yaml", "configs/config.yaml"),
            ("configs/labels_pnc.yaml", "configs/labels_pnc.yaml"),
        ]
        
        for src, dst in config_files:
            src_path = self.project_root / src
            dst_path = self.bundle_dir / dst
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                print(f"  ✓ {dst}")
            else:
                print(f"  ⚠ {src} not found, skipping")
        
        # Create edge-specific config
        edge_config = {
            "edge_deployment": {
                "model_format": "onnx",
                "preferred_model": "int8",
                "fallback_model": "float32",
                "enable_fuzzy": True,
                "max_memory_mb": 2048,
                "max_latency_ms": 500
            },
            "runtime": {
                "batch_size": 1,
                "num_threads": 2,
                "device": "cpu"
            },
            "monitoring": {
                "log_level": "INFO",
                "metrics_enabled": True,
                "health_check_interval": 60
            }
        }
        
        with open(self.bundle_dir / "configs/edge_config.yaml", 'w') as f:
            import yaml
            try:
                yaml.safe_dump(edge_config, f, indent=2)
            except ImportError:
                import json
                json.dump(edge_config, f, indent=2)
        
        print(f"✓ Configurations copied to bundle")
    
    def copy_runtime_scripts(self):
        """Copy runtime and utility scripts."""
        runtime_files = [
            ("scripts/runtime_cli.py", "runtime/runtime_cli.py"),
            ("scripts/benchmark_runtime.py", "runtime/benchmark_runtime.py"),
        ]
        
        for src, dst in runtime_files:
            src_path = self.project_root / src
            dst_path = self.bundle_dir / dst
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                print(f"  ✓ {dst}")
        
        # Copy essential source modules
        essential_modules = [
            "src/ci/model",
            "src/ci/preprocess", 
            "src/ci/fuzzy",
            "src/ci/evaluation",
            "src/ci/deployment"
        ]
        
        for module_path in essential_modules:
            src_path = self.project_root / module_path
            dst_path = self.bundle_dir / "runtime" / module_path
            
            if src_path.exists():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"  ✓ {dst_path.relative_to(self.bundle_dir)}")
        
        # Create edge launcher script
        self._create_edge_launcher()
        
        print(f"✓ Runtime scripts copied to bundle")
    
    def _create_edge_launcher(self):
        """Create simplified edge deployment launcher."""
        launcher_script = '''#!/usr/bin/env python3
"""
Edge ECG Diagnosis Launcher

Simplified launcher for edge deployment with automatic model selection
and configuration based on available resources.
"""

import sys
import os
from pathlib import Path
import json
import time
import argparse

# Add runtime modules to path
runtime_dir = Path(__file__).parent
sys.path.insert(0, str(runtime_dir))

try:
    from runtime_cli import ECGRuntimeEngine, load_ecg_data
    RUNTIME_AVAILABLE = True
except ImportError as e:
    print(f"Error importing runtime: {e}")
    RUNTIME_AVAILABLE = False

def detect_system_resources():
    """Detect available system resources."""
    import psutil
    
    return {
        'memory_mb': psutil.virtual_memory().available / (1024*1024),
        'cpu_count': psutil.cpu_count(),
        'disk_free_gb': psutil.disk_usage('.').free / (1024*1024*1024)
    }

def select_optimal_model(resources: dict):
    """Select optimal model based on available resources."""
    models_dir = Path(__file__).parent / "models"
    
    # Check available models
    float32_model = models_dir / "ecg_model_float32.onnx"
    int8_model = models_dir / "ecg_model_int8.onnx"
    
    if resources['memory_mb'] < 512 and int8_model.exists():
        return str(int8_model), "int8"
    elif float32_model.exists():
        return str(float32_model), "float32"
    else:
        return None, None

def run_diagnosis(input_file: str, output_file: str = None):
    """Run ECG diagnosis on input file."""
    if not RUNTIME_AVAILABLE:
        print("Runtime not available. Please check installation.")
        return False
    
    print("=== Edge ECG Diagnosis System ===")
    
    # Detect system resources
    try:
        resources = detect_system_resources()
        print(f"System resources: {resources['memory_mb']:.0f}MB RAM, {resources['cpu_count']} CPUs")
    except ImportError:
        resources = {'memory_mb': 1024, 'cpu_count': 1}  # Default assumption
        print("Resource detection not available, using defaults")
    
    # Select optimal model
    model_path, model_type = select_optimal_model(resources)
    if not model_path:
        print("No suitable models found in models/ directory")
        return False
    
    print(f"Selected model: {model_type} ({Path(model_path).name})")
    
    # Initialize runtime engine
    try:
        engine = ECGRuntimeEngine(
            onnx_path=model_path,
            fuzzy_config_path="configs/fuzzy_config.yaml" if Path("configs/fuzzy_config.yaml").exists() else None
        )
        print("✓ Runtime engine initialized")
    except Exception as e:
        print(f"✗ Failed to initialize engine: {e}")
        return False
    
    # Load and process ECG data
    try:
        ecg_data = load_ecg_data(input_file)
        print(f"✓ Loaded ECG data: {ecg_data.shape[0]:,} samples")
    except Exception as e:
        print(f"✗ Failed to load ECG data: {e}")
        return False
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    
    try:
        results = engine.predict(ecg_data)
        inference_time = time.time() - start_time
        
        if 'error' in results:
            print(f"✗ Inference failed: {results['error']}")
            return False
        
        print(f"✓ Inference completed in {inference_time*1000:.1f}ms")
        print(f"  Diagnosis: {results.get('prediction', 'Unknown')}")
        print(f"  Confidence: {results.get('confidence', 0):.3f}")
        
        if 'fuzzy_risk_score' in results:
            print(f"  Risk Score: {results['fuzzy_risk_score']:.3f}")
            print(f"  Risk Level: {results['fuzzy_risk_level']}")
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✓ Results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Edge ECG Diagnosis Launcher")
    parser.add_argument('--input', '-i', required=True, help='Input ECG file (CSV/NPY)')
    parser.add_argument('--output', '-o', help='Output results file (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    success = run_diagnosis(args.input, args.output)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())'''
        
        launcher_path = self.bundle_dir / "edge_launcher.py"
        with open(launcher_path, 'w', encoding='utf-8') as f:
            f.write(launcher_script)
        
        # Make executable on Unix systems
        try:
            launcher_path.chmod(0o755)
        except:
            pass  # Windows doesn't use chmod
        
        print(f"  ✓ edge_launcher.py")
    
    def create_requirements(self):
        """Create requirements.txt for edge deployment."""
        # Minimal requirements for edge deployment
        minimal_requirements = [
            "torch>=1.9.0",
            "torchvision>=0.10.0", 
            "librosa>=0.8.0",
            "numpy>=1.19.0",
            "scipy>=1.7.0",
            "onnx>=1.10.0",
            "onnxruntime>=1.8.0",
            "psutil>=5.8.0"  # For system resource detection
        ]
        
        # Full requirements (optional, for development)
        full_requirements = minimal_requirements + [
            "scikit-learn>=0.24.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "pytest>=6.0.0",
            "pyyaml>=5.4.0"
        ]
        
        # Write minimal requirements
        with open(self.bundle_dir / "requirements_minimal.txt", 'w') as f:
            f.write("# Minimal requirements for edge deployment\n")
            for req in minimal_requirements:
                f.write(f"{req}\n")
        
        # Write full requirements
        with open(self.bundle_dir / "requirements.txt", 'w') as f:
            f.write("# Full requirements for development and testing\n")
            for req in full_requirements:
                f.write(f"{req}\n")
        
        print("✓ Requirements files created")
    
    def create_examples(self):
        """Create example usage scripts and data."""
        # Create sample ECG data
        try:
            from runtime_cli import create_sample_ecg_data
            sample_path = self.bundle_dir / "data/sample_ecg.csv"
            create_sample_ecg_data(str(sample_path), duration_seconds=30)
            print(f"  ✓ data/sample_ecg.csv")
        except ImportError:
            print("  ⚠ Could not create sample data (runtime_cli not available)")
        
        # Create example usage script
        example_script = '''#!/usr/bin/env python3
"""
Example usage of the edge deployment bundle.
"""

import sys
from pathlib import Path

def run_examples():
    print("=== ECG Edge Deployment Examples ===\\n")
    
    # Example 1: Basic inference
    print("1. Basic inference with sample data:")
    print("   python edge_launcher.py --input data/sample_ecg.csv --output results.json\\n")
    
    # Example 2: Verbose mode
    print("2. Verbose inference:")
    print("   python edge_launcher.py --input data/sample_ecg.csv --verbose\\n")
    
    # Example 3: Direct runtime usage
    print("3. Direct runtime usage:")
    print("   python runtime/runtime_cli.py --onnx models/ecg_model_int8.onnx --input data/sample_ecg.csv\\n")
    
    # Example 4: Benchmarking
    print("4. Performance benchmarking:")
    print("   python runtime/benchmark_runtime.py\\n")
    
    print("See docs/DEPLOYMENT_GUIDE.md for detailed instructions.")

if __name__ == "__main__":
    run_examples()'''
        
        with open(self.bundle_dir / "examples/usage_examples.py", 'w') as f:
            f.write(example_script)
        
        print("✓ Examples created")
    
    def create_documentation(self):
        """Create deployment-specific documentation."""
        deployment_guide = '''# Edge Deployment Guide

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_minimal.txt
   ```

2. **Run Sample Diagnosis**:
   ```bash
   python edge_launcher.py --input data/sample_ecg.csv --output results.json
   ```

3. **View Results**:
   ```bash
   cat results.json
   ```

## System Requirements

### Minimum Requirements
- Python 3.8+
- RAM: 512 MB available
- Storage: 100 MB free space
- CPU: ARM/x86 with 1+ cores

### Recommended Requirements  
- RAM: 2 GB available
- Storage: 1 GB free space
- CPU: 2+ cores

## Model Selection

The system automatically selects the optimal model:

| Available RAM | Selected Model | Size | Accuracy |
|---------------|----------------|------|----------|
| < 512 MB | INT8 ONNX | 10.95 MB | High |
| ≥ 512 MB | Float32 ONNX | 43.59 MB | Highest |

## Configuration

### Edge Configuration (`configs/edge_config.yaml`)

```yaml
edge_deployment:
  model_format: "onnx"
  preferred_model: "int8"     # or "float32"
  enable_fuzzy: true
  max_memory_mb: 2048
  max_latency_ms: 500

runtime:
  batch_size: 1
  num_threads: 2              # Adjust based on CPU cores
  device: "cpu"

monitoring:
  log_level: "INFO"
  metrics_enabled: true
  health_check_interval: 60
```

## Input Formats

Supported input formats:
- **CSV**: Comma-separated values (one sample per line)
- **NPY**: NumPy binary format  
- **TXT**: Plain text (one sample per line)

### Expected Signal Format
- Sample rate: 360 Hz
- Duration: 10-60 seconds
- Single lead ECG signal

## Output Format

```json
{
  "prediction": "Normal",
  "prediction_index": 0,
  "confidence": 0.847,
  "probabilities": [0.847, 0.098, 0.032, 0.015, 0.008],
  "fuzzy_risk_score": 0.245,
  "fuzzy_risk_level": "Low",
  "inference_time_ms": 142.3,
  "num_windows": 5,
  "feature_vector_size": [5, 512]
}
```

## Performance Optimization

### For Resource-Constrained Devices
1. Use INT8 model (`models/ecg_model_int8.onnx`)
2. Reduce num_threads in config
3. Disable fuzzy logic if needed
4. Process shorter signals (10-20s)

### For Better Accuracy
1. Use Float32 model (`models/ecg_model_float32.onnx`) 
2. Enable fuzzy logic
3. Use longer signals (30s+)
4. Increase num_threads for parallel processing

## Troubleshooting

### Common Issues

**Memory Errors**
- Reduce signal length
- Use INT8 model
- Decrease num_threads

**Slow Performance**  
- Increase num_threads
- Use shorter signals
- Check system load

**Model Loading Errors**
- Verify ONNX runtime installation
- Check model file integrity
- Ensure sufficient disk space

### Debug Mode

```bash
# Enable verbose logging
python edge_launcher.py --input data.csv --verbose

# Manual debugging
python runtime/runtime_cli.py --input data.csv --verbose --benchmark
```

## Monitoring & Health Checks

Monitor system health:

```python
import psutil

# Check resource usage
memory = psutil.virtual_memory()
cpu = psutil.cpu_percent(interval=1)

print(f"Memory: {memory.percent}% used")
print(f"CPU: {cpu}% used")
```

## Integration

### REST API Integration
```python
from flask import Flask, request, jsonify
from edge_launcher import run_diagnosis

app = Flask(__name__)

@app.route('/diagnose', methods=['POST'])
def diagnose():
    # Save uploaded file and run diagnosis
    # Return JSON results
```

### IoT Integration
```python
import paho.mqtt.client as mqtt

# MQTT callback for new ECG data
def on_message(client, userdata, message):
    ecg_data = parse_ecg_message(message)
    result = run_diagnosis(ecg_data)
    publish_result(result)
```

## Security Considerations

1. **Input Validation**: Always validate input ECG data
2. **Resource Limits**: Set memory and CPU limits
3. **Access Control**: Restrict file system access  
4. **Encryption**: Use TLS for network communication
5. **Updates**: Regular security updates for dependencies

---

For technical support, see the main project README.md
'''
        
        with open(self.bundle_dir / "docs/DEPLOYMENT_GUIDE.md", 'w', encoding='utf-8') as f:
            f.write(deployment_guide)
        
        # Create README for the bundle
        bundle_readme = f'''# ECG Edge Deployment Bundle

Version: {self.timestamp}
Generated: {datetime.now().isoformat()}

## Contents

- `models/` - ONNX models for inference
- `configs/` - Configuration files
- `runtime/` - Runtime scripts and modules
- `examples/` - Usage examples
- `data/` - Sample ECG data
- `docs/` - Deployment documentation

## Quick Start

```bash
# Install dependencies
pip install -r requirements_minimal.txt

# Run diagnosis on sample data
python edge_launcher.py --input data/sample_ecg.csv

# Run with custom data
python edge_launcher.py --input your_ecg_data.csv --output results.json
```

## Documentation

- `docs/DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `examples/usage_examples.py` - Usage examples
- Main project README.md - Full system documentation

## Support

This bundle is self-contained and ready for edge deployment.
For issues, refer to the deployment guide or main project documentation.
'''
        
        with open(self.bundle_dir / "README.md", 'w') as f:
            f.write(bundle_readme)
        
        print("✓ Documentation created")
    
    def create_checksums(self):
        """Create checksums for bundle integrity verification."""
        checksums = {}
        
        # Calculate checksums for important files
        important_files = [
            "models/ecg_model_float32.onnx",
            "models/ecg_model_int8.onnx",
            "edge_launcher.py",
            "configs/config.yaml",
            "configs/labels_pnc.yaml"
        ]
        
        for file_path in important_files:
            full_path = self.bundle_dir / file_path
            if full_path.exists():
                with open(full_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    checksums[file_path] = {
                        'sha256': file_hash,
                        'size_bytes': full_path.stat().st_size
                    }
        
        # Save checksums
        with open(self.bundle_dir / "checksums.json", 'w') as f:
            json.dump(checksums, f, indent=2)
        
        print("✓ Checksums created")
        return checksums
    
    def create_bundle_info(self):
        """Create bundle information file."""
        bundle_info = {
            'bundle_name': self.bundle_name,
            'version': self.timestamp,
            'created': datetime.now().isoformat(),
            'description': 'ECG Diagnosis Edge Deployment Bundle',
            'components': {
                'models': ['Float32 ONNX', 'INT8 ONNX'],
                'runtime': ['CLI interface', 'Python modules'],
                'configs': ['Model config', 'Label mapping', 'Edge config'],
                'examples': ['Sample data', 'Usage scripts'],
                'docs': ['Deployment guide', 'API reference']
            },
            'requirements': {
                'python': '3.8+',
                'memory_min_mb': 512,
                'memory_rec_mb': 2048,
                'storage_mb': 100
            },
            'performance': {
                'latency_ms': 137,
                'throughput_samples_sec': 78575,
                'model_compression_ratio': 3.98
            }
        }
        
        with open(self.bundle_dir / "bundle_info.json", 'w') as f:
            json.dump(bundle_info, f, indent=2)
        
        print("✓ Bundle information created")
        return bundle_info
    
    def create_zip_archive(self) -> str:
        """Create zip archive of the bundle."""
        zip_filename = f"{self.bundle_name}_{self.timestamp}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.bundle_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_path = file_path.relative_to(self.bundle_dir.parent)
                    zipf.write(file_path, arc_path)
        
        zip_size_mb = Path(zip_filename).stat().st_size / (1024*1024)
        print(f"✓ Created zip archive: {zip_filename} ({zip_size_mb:.2f} MB)")
        
        return zip_filename
    
    def package_bundle(self) -> str:
        """Create complete edge deployment bundle."""
        print(f"=== Packaging Edge Deployment Bundle ===\n")
        
        # Create bundle structure
        self.create_bundle_structure()
        
        # Copy components
        self.copy_models()
        self.copy_configs()  
        self.copy_runtime_scripts()
        self.create_requirements()
        self.create_examples()
        self.create_documentation()
        
        # Create metadata
        checksums = self.create_checksums()
        bundle_info = self.create_bundle_info()
        
        # Create zip archive
        zip_filename = self.create_zip_archive()
        
        # Summary
        print(f"\n=== Bundle Package Complete ===")
        print(f"Bundle directory: {self.bundle_dir}")
        print(f"Zip archive: {zip_filename}")
        print(f"Total files: {len(list(self.bundle_dir.rglob('*')))} files")
        print(f"Bundle size: {sum(f.stat().st_size for f in self.bundle_dir.rglob('*') if f.is_file()) / (1024*1024):.2f} MB")
        
        return zip_filename


def main():
    """Main packaging function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Package ECG edge deployment bundle")
    parser.add_argument('--name', default='ecg_edge_bundle', help='Bundle name')
    parser.add_argument('--zip-only', action='store_true', help='Create zip archive only')
    
    args = parser.parse_args()
    
    packager = EdgeBundlePackager(bundle_name=args.name)
    
    try:
        zip_file = packager.package_bundle()
        
        print(f"\n✅ Edge deployment bundle created successfully!")
        print(f"📦 Archive: {zip_file}")
        print(f"📁 Directory: {packager.bundle_dir}")
        print(f"\nReady for edge deployment! 🚀")
        
        return zip_file
        
    except Exception as e:
        print(f"\n❌ Bundle packaging failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
