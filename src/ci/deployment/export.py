"""
Model export utilities for deployment.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json
import logging


class ModelExporter:
    """
    Utility class for exporting models to various formats.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize model exporter.
        
        Args:
            model: PyTorch model to export
            device: Device the model is on
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Set to eval mode for export
        self.model.eval()
    
    def export_torchscript(
        self,
        save_path: str,
        input_shape: Tuple[int, ...] = (1, 1, 224, 224),
        method: str = 'trace'
    ) -> str:
        """
        Export model to TorchScript format.
        
        Args:
            save_path: Path to save the model
            input_shape: Input tensor shape for tracing
            method: 'trace' or 'script'
            
        Returns:
            Path to saved model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if method == 'trace':
            # Create dummy input for tracing
            dummy_input = torch.randn(*input_shape).to(self.device)
            
            # Trace the model
            traced_model = torch.jit.trace(self.model, dummy_input)
            
            # Save traced model
            torch.jit.save(traced_model, str(save_path))
            
        elif method == 'script':
            # Script the model (for models with control flow)
            scripted_model = torch.jit.script(self.model)
            
            # Save scripted model
            torch.jit.save(scripted_model, str(save_path))
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'.")
        
        self.logger.info(f"Model exported to TorchScript: {save_path}")
        return str(save_path)
    
    def export_state_dict(self, save_path: str, include_metadata: bool = True) -> str:
        """
        Export model state dictionary.
        
        Args:
            save_path: Path to save the state dict
            include_metadata: Whether to include model metadata
            
        Returns:
            Path to saved state dict
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_dict = {
            'model_state_dict': self.model.state_dict()
        }
        
        if include_metadata:
            export_dict.update({
                'model_class': self.model.__class__.__name__,
                'model_config': getattr(self.model, 'get_config', lambda: {})(),
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            })
        
        torch.save(export_dict, str(save_path))
        self.logger.info(f"State dict exported: {save_path}")
        return str(save_path)
    
    def export_full_model(self, save_path: str) -> str:
        """
        Export the full model (architecture + weights).
        
        Args:
            save_path: Path to save the model
            
        Returns:
            Path to saved model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model, str(save_path))
        self.logger.info(f"Full model exported: {save_path}")
        return str(save_path)


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 1, 224, 224),
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[Dict[str, Any]] = None,
    opset_version: int = 11,
    device: str = 'cpu'
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        save_path: Path to save ONNX model
        input_shape: Shape of input tensor
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axes specification
        opset_version: ONNX opset version
        device: Device for export
        
    Returns:
        Path to exported ONNX model
    """
    try:
        import onnx
        import onnxruntime
    except ImportError:
        raise ImportError("ONNX export requires 'onnx' and 'onnxruntime' packages. Install with: pip install onnx onnxruntime")
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Default names
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # Verify the exported model
    try:
        onnx_model = onnx.load(str(save_path))
        onnx.checker.check_model(onnx_model)
        
        # Test with ONNX Runtime
        ort_session = onnxruntime.InferenceSession(str(save_path))
        ort_inputs = {input_names[0]: dummy_input.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        logging.getLogger(__name__).info(f"ONNX model exported and verified: {save_path}")
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"ONNX model verification failed: {e}")
    
    return str(save_path)


def load_torchscript_model(model_path: str, device: str = 'cpu') -> torch.jit.ScriptModule:
    """
    Load TorchScript model.
    
    Args:
        model_path: Path to TorchScript model
        device: Device to load model on
        
    Returns:
        Loaded TorchScript model
    """
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def load_state_dict_model(
    model_class: type,
    state_dict_path: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    device: str = 'cpu'
) -> nn.Module:
    """
    Load model from state dictionary.
    
    Args:
        model_class: Model class to instantiate
        state_dict_path: Path to state dictionary file
        model_kwargs: Keyword arguments for model initialization
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    # Load checkpoint
    checkpoint = torch.load(state_dict_path, map_location=device)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create model instance
    model = model_class(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def convert_to_mobile(
    model_path: str,
    output_path: str,
    optimize: bool = True
) -> str:
    """
    Convert TorchScript model for mobile deployment.
    
    Args:
        model_path: Path to TorchScript model
        output_path: Path to save mobile model
        optimize: Whether to apply mobile optimizations
        
    Returns:
        Path to mobile-optimized model
    """
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
    except ImportError:
        raise ImportError("Mobile optimization requires torch.utils.mobile_optimizer")
    
    # Load model
    model = torch.jit.load(model_path)
    
    if optimize:
        # Optimize for mobile
        mobile_model = optimize_for_mobile(model)
    else:
        mobile_model = model
    
    # Save mobile model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mobile_model._save_for_lite_interpreter(str(output_path))
    
    logging.getLogger(__name__).info(f"Mobile model saved: {output_path}")
    return str(output_path)


class ModelSizeAnalyzer:
    """
    Analyzer for model size and complexity.
    """
    
    @staticmethod
    def analyze_model(model: nn.Module) -> Dict[str, Any]:
        """
        Analyze model size and complexity.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary with analysis results
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size in MB (assuming float32)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        # Layer analysis
        layer_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_params = sum(p.numel() for p in module.parameters())
                layer_info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': module_params
                })
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'layer_count': len(layer_info),
            'layers': layer_info
        }
