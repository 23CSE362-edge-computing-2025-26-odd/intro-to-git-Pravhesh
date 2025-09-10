#!/usr/bin/env python3
"""
Export ECG-specific ONNX model with correct input dimensions for spectrogram data.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import numpy as np
from ci.model.feature_extractor import FeatureExtractor

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not available. Install with: pip install onnx onnxruntime")
    exit(1)


def export_ecg_onnx_model():
    """Export ECG-specific ONNX model with proper dimensions."""
    
    print("Creating ECG-specific model for ONNX export...")
    
    # Create model optimized for ECG spectrograms (single channel)
    model = FeatureExtractor(
        embedding_dim=512,
        pretrained=False,  # Don't use ImageNet weights
        num_classes=None   # Feature extraction mode
    )
    model.eval()
    
    # Expected ECG spectrogram dimensions: [batch, 1, 64, ~57]
    # This comes from Mel spectrogram: n_mels=64, time frames depend on signal length
    batch_size = 1
    channels = 1
    height = 64  # n_mels
    width = 57   # Typical for 10s window with hop_length=64
    
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
    
    # Export to ONNX
    output_path = "demo_outputs/ecg_model.onnx"
    Path("demo_outputs").mkdir(exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['spectrogram'],
        output_names=['features'],
        dynamic_axes={
            'spectrogram': {0: 'batch_size', 3: 'time_frames'},
            'features': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ECG ONNX model exported to: {output_path}")
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")
    
    # Test inference
    session = ort.InferenceSession(output_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Test with different time dimensions
    for width in [57, 100, 150]:
        test_input = np.random.randn(1, 1, 64, width).astype(np.float32)
        onnx_output = session.run([output_name], {input_name: test_input})[0]
        
        # Compare with PyTorch
        torch_input = torch.from_numpy(test_input)
        with torch.no_grad():
            torch_output = model(torch_input).numpy()
        
        diff = np.abs(onnx_output - torch_output).mean()
        print(f"✓ Width {width}: Mean difference = {diff:.6f}")
    
    # Quantize to INT8
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    quantized_path = "demo_outputs/ecg_model_int8.onnx"
    quantize_dynamic(
        output_path,
        quantized_path,
        weight_type=QuantType.QInt8
    )
    
    print(f"✓ INT8 quantized model saved to: {quantized_path}")
    
    # Check file sizes
    float_size = Path(output_path).stat().st_size / (1024*1024)
    int8_size = Path(quantized_path).stat().st_size / (1024*1024)
    compression = float_size / int8_size if int8_size > 0 else 0
    
    print(f"  Float32 size: {float_size:.2f} MB")
    print(f"  INT8 size: {int8_size:.2f} MB") 
    print(f"  Compression ratio: {compression:.2f}x")
    
    return output_path, quantized_path


if __name__ == "__main__":
    if ONNX_AVAILABLE:
        export_ecg_onnx_model()
    else:
        print("ONNX dependencies not available!")
