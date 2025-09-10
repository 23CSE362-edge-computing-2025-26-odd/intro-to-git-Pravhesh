"""
Model factory and utilities for creating models from configuration.
"""

import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor, AdaptiveCNNExtractor


def create_model(config: Union[Dict[str, Any], str, Path], num_classes: Optional[int] = None) -> nn.Module:
    """
    Create a model from configuration.
    
    Args:
        config: Configuration dictionary or path to config file
        num_classes: Number of classes for classification (overrides config)
        
    Returns:
        Initialized model
    """
    # Load config if path provided
    if isinstance(config, (str, Path)):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Extract model config
    model_config = config.get('model', {})
    
    # Get model parameters
    backbone = model_config.get('backbone', 'resnet18')
    embedding_dim = model_config.get('embedding_dim', 512)
    pretrained = model_config.get('pretrained', True)
    finetune_last_block = model_config.get('finetune_last_block', False)
    
    # Create model based on backbone
    if backbone == 'resnet18':
        model = FeatureExtractor(
            backbone=backbone,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            finetune_last_block=finetune_last_block,
            num_classes=num_classes
        )
    elif backbone == 'adaptive_cnn':
        model = AdaptiveCNNExtractor(
            embedding_dim=embedding_dim,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    return model


def create_feature_extractor(config: Union[Dict[str, Any], str, Path]) -> nn.Module:
    """
    Create a feature extractor (no classification head) from configuration.
    
    Args:
        config: Configuration dictionary or path to config file
        
    Returns:
        Feature extractor model
    """
    return create_model(config, num_classes=None)


def create_classifier(config: Union[Dict[str, Any], str, Path], num_classes: int) -> nn.Module:
    """
    Create a classifier model from configuration.
    
    Args:
        config: Configuration dictionary or path to config file
        num_classes: Number of classes for classification
        
    Returns:
        Classification model
    """
    return create_model(config, num_classes=num_classes)


class ModelConfig:
    """
    Configuration class for model parameters with validation.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        embedding_dim: int = 512,
        pretrained: bool = True,
        finetune_last_block: bool = False,
        num_classes: Optional[int] = None
    ):
        """
        Initialize model configuration.
        
        Args:
            backbone: Model backbone architecture
            embedding_dim: Dimension of feature embeddings
            pretrained: Whether to use pretrained weights
            finetune_last_block: Whether to finetune the last block only
            num_classes: Number of output classes (None for feature extraction)
        """
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.finetune_last_block = finetune_last_block
        self.num_classes = num_classes
        
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        valid_backbones = ['resnet18', 'adaptive_cnn']
        if self.backbone not in valid_backbones:
            raise ValueError(f"backbone must be one of {valid_backbones}")
        
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        
        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError("num_classes must be positive if specified")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'backbone': self.backbone,
            'embedding_dim': self.embedding_dim,
            'pretrained': self.pretrained,
            'finetune_last_block': self.finetune_last_block,
            'num_classes': self.num_classes
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'ModelConfig':
        """Create from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config.get('model', {})
        return cls.from_dict(model_config)


def get_model_summary(model: nn.Module, input_shape: tuple = (1, 1, 224, 224)) -> str:
    """
    Get a summary of model architecture and parameters.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        
    Returns:
        Model summary string
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    
    # Try to get model output shape
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(*input_shape)
            output = model(dummy_input)
            output_shape = tuple(output.shape)
    except Exception as e:
        output_shape = "Unable to determine"
    
    summary = f"""
Model Summary:
=============
Architecture: {model.__class__.__name__}
Input shape: {input_shape}
Output shape: {output_shape}
Trainable parameters: {total_params:,}

Configuration:
"""
    
    # Add model-specific config if available
    if hasattr(model, 'get_config'):
        config = model.get_config()
        for key, value in config.items():
            summary += f"  {key}: {value}\n"
    
    return summary


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    strict: bool = True,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load model weights from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce key matching
        map_location: Device to map tensors to
        
    Returns:
        Checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=strict)
    
    # Return metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown'),
        'metrics': checkpoint.get('metrics', {}),
        'optimizer_state': 'optimizer_state_dict' in checkpoint
    }
    
    return metadata


def save_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None
):
    """
    Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        epoch: Training epoch
        loss: Training loss
        metrics: Training metrics
        optimizer: Optimizer state to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'metrics': metrics or {}
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Add model config if available
    if hasattr(model, 'get_config'):
        checkpoint['model_config'] = model.get_config()
    
    torch.save(checkpoint, checkpoint_path)
