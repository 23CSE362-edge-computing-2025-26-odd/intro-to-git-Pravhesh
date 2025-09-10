# Edge Deployment Guide

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
