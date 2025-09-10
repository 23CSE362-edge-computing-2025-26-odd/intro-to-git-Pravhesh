"""
Tests for model components.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
import yaml

from src.ci.model import FeatureExtractor, AdaptiveCNNExtractor
from src.ci.model.model import (
    create_model, create_feature_extractor, create_classifier,
    ModelConfig, get_model_summary, save_model_checkpoint, load_model_checkpoint
)


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""
    
    def test_feature_extractor_init(self):
        """Test FeatureExtractor initialization."""
        model = FeatureExtractor(
            backbone="resnet18",
            embedding_dim=256,
            pretrained=False,  # Avoid downloading weights in tests
            num_classes=None
        )
        
        assert model.backbone == "resnet18"
        assert model.embedding_dim == 256
        assert model.num_classes is None
        assert not model.pretrained
    
    def test_feature_extractor_forward_feature_mode(self):
        """Test forward pass in feature extraction mode."""
        model = FeatureExtractor(
            embedding_dim=128,
            pretrained=False,
            num_classes=None
        )
        model.eval()
        
        # Test with different input shapes
        batch_size = 2
        
        # 3D input (add channel dimension)
        x = torch.randn(batch_size, 224, 224)
        output = model(x)
        assert output.shape == (batch_size, 128)
        
        # 4D input (proper format)
        x = torch.randn(batch_size, 1, 224, 224)
        output = model(x)
        assert output.shape == (batch_size, 128)
        
        # Multi-channel input (should convert to single channel)
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)
        assert output.shape == (batch_size, 128)
    
    def test_feature_extractor_forward_classification_mode(self):
        """Test forward pass in classification mode."""
        num_classes = 5
        model = FeatureExtractor(
            embedding_dim=128,
            pretrained=False,
            num_classes=num_classes
        )
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 1, 224, 224)
        output = model(x)
        
        assert output.shape == (batch_size, num_classes)
    
    def test_feature_extractor_extract_features(self):
        """Test extract_features method."""
        model = FeatureExtractor(
            embedding_dim=128,
            pretrained=False,
            num_classes=5  # Classification mode
        )
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 1, 224, 224)
        
        # Should return embeddings even in classification mode
        features = model.extract_features(x)
        assert features.shape == (batch_size, 128)
    
    def test_feature_extractor_unsupported_backbone(self):
        """Test error handling for unsupported backbone."""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            FeatureExtractor(backbone="unsupported_backbone", pretrained=False)
    
    def test_feature_extractor_config(self):
        """Test get_config method."""
        config = {
            'backbone': 'resnet18',
            'embedding_dim': 256,
            'pretrained': False,
            'finetune_last_block': True,
            'num_classes': 10
        }
        
        model = FeatureExtractor(**config)
        retrieved_config = model.get_config()
        
        assert retrieved_config == config


class TestAdaptiveCNNExtractor:
    """Test cases for AdaptiveCNNExtractor class."""
    
    def test_adaptive_cnn_init(self):
        """Test AdaptiveCNNExtractor initialization."""
        model = AdaptiveCNNExtractor(embedding_dim=256, num_classes=None)
        
        assert model.embedding_dim == 256
        assert model.num_classes is None
        assert model.classifier is None
    
    def test_adaptive_cnn_forward_feature_mode(self):
        """Test forward pass in feature extraction mode."""
        model = AdaptiveCNNExtractor(embedding_dim=128, num_classes=None)
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 1, 224, 224)
        output = model(x)
        
        assert output.shape == (batch_size, 128)
    
    def test_adaptive_cnn_forward_classification_mode(self):
        """Test forward pass in classification mode."""
        num_classes = 3
        model = AdaptiveCNNExtractor(embedding_dim=128, num_classes=num_classes)
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 1, 224, 224)
        output = model(x)
        
        assert output.shape == (batch_size, num_classes)


class TestModelFactory:
    """Test cases for model factory functions."""
    
    def test_create_model_from_dict(self):
        """Test creating model from configuration dictionary."""
        config = {
            'model': {
                'backbone': 'resnet18',
                'embedding_dim': 256,
                'pretrained': False,
                'finetune_last_block': True
            }
        }
        
        model = create_model(config, num_classes=None)
        
        assert isinstance(model, FeatureExtractor)
        assert model.embedding_dim == 256
        assert model.finetune_last_block == True
    
    def test_create_model_adaptive_cnn(self):
        """Test creating adaptive CNN model."""
        config = {
            'model': {
                'backbone': 'adaptive_cnn',
                'embedding_dim': 128
            }
        }
        
        model = create_model(config, num_classes=5)
        
        assert isinstance(model, AdaptiveCNNExtractor)
        assert model.embedding_dim == 128
        assert model.num_classes == 5
    
    def test_create_model_from_file(self):
        """Test creating model from configuration file."""
        config = {
            'model': {
                'backbone': 'resnet18',
                'embedding_dim': 512,
                'pretrained': False
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            model = create_model(temp_path, num_classes=None)
            assert isinstance(model, FeatureExtractor)
            assert model.embedding_dim == 512
        finally:
            Path(temp_path).unlink()
    
    def test_create_feature_extractor(self):
        """Test create_feature_extractor function."""
        config = {
            'model': {
                'backbone': 'resnet18',
                'embedding_dim': 256,
                'pretrained': False
            }
        }
        
        model = create_feature_extractor(config)
        
        assert isinstance(model, FeatureExtractor)
        assert model.num_classes is None
    
    def test_create_classifier(self):
        """Test create_classifier function."""
        config = {
            'model': {
                'backbone': 'resnet18',
                'embedding_dim': 256,
                'pretrained': False
            }
        }
        
        num_classes = 7
        model = create_classifier(config, num_classes)
        
        assert isinstance(model, FeatureExtractor)
        assert model.num_classes == num_classes
    
    def test_create_model_unsupported_backbone(self):
        """Test error handling for unsupported backbone."""
        config = {
            'model': {
                'backbone': 'unsupported_model',
                'embedding_dim': 256
            }
        }
        
        with pytest.raises(ValueError, match="Unsupported backbone"):
            create_model(config)


class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_model_config_init(self):
        """Test ModelConfig initialization."""
        config = ModelConfig(
            backbone='resnet18',
            embedding_dim=256,
            pretrained=True,
            finetune_last_block=False,
            num_classes=10
        )
        
        assert config.backbone == 'resnet18'
        assert config.embedding_dim == 256
        assert config.pretrained == True
        assert config.finetune_last_block == False
        assert config.num_classes == 10
    
    def test_model_config_validation(self):
        """Test ModelConfig parameter validation."""
        # Test invalid backbone
        with pytest.raises(ValueError, match="backbone must be one of"):
            ModelConfig(backbone='invalid_backbone')
        
        # Test invalid embedding_dim
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            ModelConfig(embedding_dim=0)
        
        # Test invalid num_classes
        with pytest.raises(ValueError, match="num_classes must be positive"):
            ModelConfig(num_classes=0)
    
    def test_model_config_to_dict(self):
        """Test ModelConfig to_dict method."""
        config = ModelConfig(
            backbone='adaptive_cnn',
            embedding_dim=128,
            pretrained=False,
            num_classes=5
        )
        
        config_dict = config.to_dict()
        expected = {
            'backbone': 'adaptive_cnn',
            'embedding_dim': 128,
            'pretrained': False,
            'finetune_last_block': False,
            'num_classes': 5
        }
        
        assert config_dict == expected
    
    def test_model_config_from_dict(self):
        """Test ModelConfig from_dict method."""
        config_dict = {
            'backbone': 'resnet18',
            'embedding_dim': 256,
            'pretrained': True,
            'finetune_last_block': True,
            'num_classes': None
        }
        
        config = ModelConfig.from_dict(config_dict)
        
        assert config.backbone == 'resnet18'
        assert config.embedding_dim == 256
        assert config.pretrained == True
        assert config.finetune_last_block == True
        assert config.num_classes is None
    
    def test_model_config_from_yaml(self):
        """Test ModelConfig from_yaml method."""
        config_data = {
            'model': {
                'backbone': 'adaptive_cnn',
                'embedding_dim': 512,
                'pretrained': False,
                'num_classes': 8
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = ModelConfig.from_yaml(temp_path)
            assert config.backbone == 'adaptive_cnn'
            assert config.embedding_dim == 512
            assert config.pretrained == False
            assert config.num_classes == 8
        finally:
            Path(temp_path).unlink()


class TestModelUtilities:
    """Test cases for model utility functions."""
    
    def test_get_model_summary(self):
        """Test get_model_summary function."""
        model = FeatureExtractor(
            embedding_dim=128,
            pretrained=False,
            num_classes=None
        )
        
        summary = get_model_summary(model, input_shape=(1, 1, 224, 224))
        
        assert "Model Summary:" in summary
        assert "FeatureExtractor" in summary
        assert "Trainable parameters:" in summary
        assert "Configuration:" in summary
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading model checkpoints."""
        model = FeatureExtractor(
            embedding_dim=128,
            pretrained=False,
            num_classes=5
        )
        
        # Create optimizer for testing
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # Save checkpoint
            save_model_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                epoch=10,
                loss=0.5,
                metrics={'accuracy': 0.85},
                optimizer=optimizer
            )
            
            # Create new model instance
            new_model = FeatureExtractor(
                embedding_dim=128,
                pretrained=False,
                num_classes=5
            )
            
            # Load checkpoint
            metadata = load_model_checkpoint(
                new_model, 
                checkpoint_path, 
                map_location='cpu'
            )
            
            assert metadata['epoch'] == 10
            assert metadata['loss'] == 0.5
            assert metadata['metrics']['accuracy'] == 0.85
            assert metadata['optimizer_state'] == True
            
        finally:
            Path(checkpoint_path).unlink()
    
    def test_load_checkpoint_different_formats(self):
        """Test loading checkpoints with different formats."""
        model = FeatureExtractor(embedding_dim=64, pretrained=False)
        
        # Test with state_dict format
        checkpoint = {'state_dict': model.state_dict()}
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
            torch.save(checkpoint, checkpoint_path)
        
        try:
            new_model = FeatureExtractor(embedding_dim=64, pretrained=False)
            metadata = load_model_checkpoint(new_model, checkpoint_path, map_location='cpu')
            
            assert metadata['epoch'] == 'unknown'
            assert metadata['optimizer_state'] == False
            
        finally:
            Path(checkpoint_path).unlink()


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_end_to_end_feature_extraction(self):
        """Test end-to-end feature extraction pipeline."""
        # Create config
        config = {
            'model': {
                'backbone': 'resnet18',
                'embedding_dim': 256,
                'pretrained': False,
                'finetune_last_block': True
            }
        }
        
        # Create feature extractor
        model = create_feature_extractor(config)
        model.eval()
        
        # Simulate spectrogram input
        batch_size = 4
        spectrograms = torch.randn(batch_size, 1, 224, 224)
        
        # Extract features
        with torch.no_grad():
            features = model(spectrograms)
        
        # Verify output
        assert features.shape == (batch_size, 256)
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()
    
    def test_end_to_end_classification(self):
        """Test end-to-end classification pipeline."""
        # Create config
        config = {
            'model': {
                'backbone': 'adaptive_cnn',
                'embedding_dim': 128
            }
        }
        
        # Create classifier
        num_classes = 4
        model = create_classifier(config, num_classes)
        model.eval()
        
        # Simulate spectrogram input
        batch_size = 3
        spectrograms = torch.randn(batch_size, 1, 224, 224)
        
        # Get predictions
        with torch.no_grad():
            logits = model(spectrograms)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # Verify output
        assert logits.shape == (batch_size, num_classes)
        assert probabilities.shape == (batch_size, num_classes)
        assert predictions.shape == (batch_size,)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size))
        assert all(0 <= pred < num_classes for pred in predictions)


if __name__ == "__main__":
    pytest.main([__file__])
