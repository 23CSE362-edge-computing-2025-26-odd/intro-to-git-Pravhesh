# ECG Edge Deployment Bundle

Version: 20250910_222617
Generated: 2025-09-10T22:26:26.883249

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
