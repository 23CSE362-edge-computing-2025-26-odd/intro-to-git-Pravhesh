"""
Feature extractor for ECG signals using CNN backbone.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class FeatureExtractor(nn.Module):
    """
    Feature extractor using ResNet18 backbone for spectrogram-based ECG analysis.
    
    This model takes spectrogram representations of ECG signals and extracts
    meaningful features using a pre-trained ResNet18 backbone, adapted for
    medical signal analysis.
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 512,
        pretrained: bool = True,
        finetune_last_block: bool = False,
        num_classes: Optional[int] = None
    ):
        """
        Initialize the feature extractor.
        
        Args:
            backbone: Name of the backbone architecture (currently supports 'resnet18')
            embedding_dim: Dimension of the output feature embeddings
            pretrained: Whether to use pre-trained weights
            finetune_last_block: Whether to allow gradients in the last residual block
            num_classes: Number of output classes for classification head (None for feature extraction only)
        """
        super().__init__()
        
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.finetune_last_block = finetune_last_block
        self.num_classes = num_classes
        
        # Initialize backbone
        if backbone == "resnet18":
            self.encoder = models.resnet18(pretrained=pretrained)
            # Modify first conv layer for single-channel spectrograms
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Get the feature dimension from the original classifier
            backbone_features = self.encoder.fc.in_features
            
            # Replace the classifier with identity or custom head
            if num_classes is not None:
                # Classification mode: backbone -> projection -> classifier
                self.encoder.fc = nn.Identity()
                self.projection = nn.Linear(backbone_features, embedding_dim)
                self.classifier = nn.Linear(embedding_dim, num_classes)
            else:
                # Feature extraction mode: backbone -> projection
                self.encoder.fc = nn.Linear(backbone_features, embedding_dim)
                self.projection = None
                self.classifier = None
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze layers except the last block if specified
        if not finetune_last_block and pretrained:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all parameters except the last residual block."""
        # Freeze all parameters first
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze the last residual block (layer4) and modified conv1/fc layers
        for param in self.encoder.layer4.parameters():
            param.requires_grad = True
        for param in self.encoder.conv1.parameters():
            param.requires_grad = True
        if hasattr(self.encoder, 'fc') and self.encoder.fc is not nn.Identity():
            for param in self.encoder.fc.parameters():
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width) - spectrogram
            
        Returns:
            Features tensor of shape (batch_size, embedding_dim) or
            Logits tensor of shape (batch_size, num_classes) if in classification mode
        """
        # Ensure input is the right shape for single-channel spectrogram
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        elif x.dim() == 4 and x.shape[1] != 1:
            # If multi-channel, take first channel or convert to grayscale
            if x.shape[1] == 3:  # RGB to grayscale conversion
                x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            else:
                x = x[:, 0:1]  # Take first channel
        
        # Extract features through backbone
        features = self.encoder(x)
        
        if self.num_classes is not None:
            # Classification mode
            embeddings = self.projection(features)
            logits = self.classifier(embeddings)
            return logits
        else:
            # Feature extraction mode
            return features
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (always returns embeddings, even in classification mode).
        
        Args:
            x: Input tensor
            
        Returns:
            Feature embeddings of shape (batch_size, embedding_dim)
        """
        # Temporarily switch to feature extraction mode
        was_classification = self.num_classes is not None
        if was_classification:
            # In classification mode, get features before classifier
            features = self.encoder(x)
            embeddings = self.projection(features)
            return embeddings
        else:
            return self.forward(x)
    
    def get_config(self) -> dict:
        """Get the configuration of this feature extractor."""
        return {
            'backbone': self.backbone,
            'embedding_dim': self.embedding_dim,
            'pretrained': self.pretrained,
            'finetune_last_block': self.finetune_last_block,
            'num_classes': self.num_classes
        }


class AdaptiveCNNExtractor(nn.Module):
    """
    Alternative lightweight CNN feature extractor designed specifically for ECG spectrograms.
    """
    
    def __init__(self, embedding_dim: int = 512, num_classes: Optional[int] = None):
        """
        Initialize the adaptive CNN extractor.
        
        Args:
            embedding_dim: Dimension of output embeddings
            num_classes: Number of classes for classification head
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Convolutional layers for spectrogram processing
        self.features = nn.Sequential(
            # Block 1: Extract low-level temporal patterns
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2: Capture frequency patterns
            nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: Higher-level feature combinations
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: Final feature extraction
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Feature projection
        self.projection = nn.Linear(512, embedding_dim)
        
        # Classification head (optional)
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN extractor."""
        # Ensure single channel
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Project to embedding space
        embeddings = self.projection(features)
        
        if self.classifier is not None:
            logits = self.classifier(embeddings)
            return logits
        else:
            return embeddings
