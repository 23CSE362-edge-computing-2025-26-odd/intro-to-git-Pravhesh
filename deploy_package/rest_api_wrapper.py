#!/usr/bin/env python3
"""
REST API Wrapper for ECG Edge Model
Enables Java iFogSim integration via HTTP calls
"""

from flask import Flask, request, jsonify, send_file
import os
import json
import tempfile
import csv
from pathlib import Path
import numpy as np
import io
import base64

app = Flask(__name__)

# Add path to edge bundle
BUNDLE_PATH = Path(__file__).parent.parent / "dist" / "ecg_edge_bundle_20250910_222617"
import sys
sys.path.insert(0, str(BUNDLE_PATH / "runtime"))

try:
    from runtime_cli import ECGRuntimeEngine
    ENGINE_AVAILABLE = True
    
    # Initialize the engine
    model_path = BUNDLE_PATH / "models" / "ecg_model_int8.onnx"
    if model_path.exists():
        ecg_engine = ECGRuntimeEngine(onnx_path=str(model_path))
        print(f"✓ ECG Engine initialized with model: {model_path}")
    else:
        print("⚠ Model not found, API will return mock data")
        ecg_engine = None
        
except ImportError as e:
    print(f"⚠ Runtime engine not available: {e}")
    ENGINE_AVAILABLE = False
    ecg_engine = None

@app.route('/')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "service": "ECG Diagnosis API",
        "engine_available": ENGINE_AVAILABLE,
        "model_loaded": ecg_engine is not None
    })

@app.route('/diagnose', methods=['POST'])
def diagnose_ecg():
    """
    Main diagnosis endpoint
    Accepts ECG data in multiple formats:
    - JSON: {"ecg_data": [1.2, 0.8, ...]}
    - CSV file upload
    - Base64 encoded CSV data
    """
    try:
        # Handle different input formats
        ecg_data = None
        
        if 'file' in request.files:
            # File upload
            file = request.files['file']
            content = file.read().decode('utf-8')
            ecg_data = parse_csv_data(content)
            
        elif request.is_json:
            json_data = request.get_json()
            if 'ecg_data' in json_data:
                ecg_data = np.array(json_data['ecg_data'])
            elif 'csv_base64' in json_data:
                # Base64 encoded CSV
                csv_content = base64.b64decode(json_data['csv_base64']).decode('utf-8')
                ecg_data = parse_csv_data(csv_content)
        
        if ecg_data is None:
            return jsonify({"error": "No valid ECG data provided"}), 400
            
        # Run diagnosis
        if ecg_engine and ENGINE_AVAILABLE:
            result = ecg_engine.predict(ecg_data)
        else:
            # Mock response for testing when engine not available
            result = {
                "prediction": "Normal",
                "prediction_index": 0,
                "confidence": 0.847,
                "probabilities": [0.847, 0.098, 0.032, 0.015, 0.008],
                "fuzzy_risk_score": 0.245,
                "fuzzy_risk_level": "Low",
                "inference_time_ms": 142.3,
                "num_windows": len(ecg_data) // 3600 if len(ecg_data) > 3600 else 1,
                "feature_vector_size": [1, 512],
                "note": "Mock data - engine not available"
            }
        
        # Add metadata
        result["api_version"] = "1.0"
        result["timestamp"] = pd.Timestamp.now().isoformat()
        result["input_samples"] = len(ecg_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_diagnose', methods=['POST'])
def batch_diagnose():
    """
    Batch diagnosis endpoint for multiple ECG samples
    Input: {"samples": [{"id": "001", "ecg_data": [...]}, ...]}
    """
    try:
        if not request.is_json:
            return jsonify({"error": "JSON input required"}), 400
            
        data = request.get_json()
        if 'samples' not in data:
            return jsonify({"error": "No samples provided"}), 400
            
        results = []
        for sample in data['samples']:
            sample_id = sample.get('id', 'unknown')
            ecg_data = np.array(sample['ecg_data'])
            
            if ecg_engine and ENGINE_AVAILABLE:
                result = ecg_engine.predict(ecg_data)
            else:
                result = create_mock_result(ecg_data)
                
            result['sample_id'] = sample_id
            results.append(result)
            
        return jsonify({
            "batch_results": results,
            "total_samples": len(results),
            "api_version": "1.0"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model_info')
def model_info():
    """Get model information and capabilities"""
    return jsonify({
        "model_format": "ONNX",
        "model_type": "INT8 Quantized",
        "input_format": "Single-lead ECG, 360 Hz",
        "output_classes": ["Normal", "Atrial Fibrillation", "Other Arrhythmia", "Noisy", "Other"],
        "expected_sample_rate": 360,
        "min_duration_seconds": 10,
        "max_duration_seconds": 60,
        "features": {
            "fuzzy_logic": True,
            "risk_assessment": True,
            "confidence_scoring": True
        },
        "performance": {
            "avg_latency_ms": 137,
            "throughput_samples_sec": 78575
        }
    })

def parse_csv_data(csv_content):
    """Parse CSV content to numpy array"""
    try:
        # Try to read as CSV with header
        lines = csv_content.strip().split('\n')
        if len(lines) == 1:
            # Single line, comma-separated
            return np.array([float(x.strip()) for x in lines[0].split(',')])
        else:
            # Multiple lines
            data = []
            for line in lines:
                if line.strip():
                    data.extend([float(x.strip()) for x in line.split(',')])
            return np.array(data)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV data: {e}")

def create_mock_result(ecg_data):
    """Create mock result when engine not available"""
    import random
    random.seed(42)  # Consistent results
    
    classes = ["Normal", "Atrial Fibrillation", "Other Arrhythmia", "Noisy", "Other"]
    pred_idx = random.randint(0, len(classes)-1)
    
    return {
        "prediction": classes[pred_idx],
        "prediction_index": pred_idx,
        "confidence": round(random.uniform(0.6, 0.95), 3),
        "probabilities": [round(random.uniform(0, 1), 3) for _ in classes],
        "fuzzy_risk_score": round(random.uniform(0.1, 0.8), 3),
        "fuzzy_risk_level": random.choice(["Low", "Medium", "High"]),
        "inference_time_ms": round(random.uniform(100, 200), 1),
        "num_windows": max(1, len(ecg_data) // 3600),
        "feature_vector_size": [1, 512],
        "note": "Mock data - engine not available"
    }

if __name__ == '__main__':
    print("Starting ECG Diagnosis REST API...")
    print(f"Engine available: {ENGINE_AVAILABLE}")
    print(f"Bundle path: {BUNDLE_PATH}")
    
    # Import pandas for timestamp if available
    try:
        import pandas as pd
    except ImportError:
        import datetime
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    class MockTimestamp:
                        def isoformat(self):
                            return datetime.datetime.now().isoformat()
                    return MockTimestamp()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
