#!/usr/bin/env python3
"""
Model training script for ECG classification.

This script provides a complete training pipeline for ECG classification models
using the preprocessing and model components.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ci.model import create_classifier, create_feature_extractor
from src.ci.model.model import save_model_checkpoint, get_model_summary
from src.ci.preprocess import PreprocessingPipeline


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


class ECGDataset(torch.utils.data.Dataset):
    """
    Dummy ECG dataset for training demo.
    
    In a real implementation, this would load and preprocess actual ECG data.
    """
    
    def __init__(self, num_samples: int = 1000, num_classes: int = 4, image_size: int = 224):
        """
        Initialize dummy dataset.
        
        Args:
            num_samples: Number of samples to generate
            num_classes: Number of classes
            image_size: Size of spectrogram images
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Generate dummy data
        self.spectrograms = torch.randn(num_samples, 1, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]


class ModelTrainer:
    """
    Model trainer for ECG classification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Training parameters
        self.lr = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.epochs = config.get('epochs', 10)
        self.batch_size = config.get('batch_size', 32)
        self.val_split = config.get('val_split', 0.2)
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
    
    def create_data_loaders(self, num_samples: int = 1000, num_classes: int = 4) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation data loaders.
        
        Args:
            num_samples: Number of samples to generate
            num_classes: Number of classes
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create dummy dataset
        dataset = ECGDataset(num_samples, num_classes)
        
        # Split into train/val
        val_size = int(self.val_split * len(dataset))
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.debug(
                    f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}'
                )
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, save_dir: str = "checkpoints") -> Dict[str, Any]:
        """
        Run complete training loop.
        
        Args:
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history and metrics
        """
        self.logger.info("Starting training...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create data loaders
        num_classes = getattr(self.model, 'num_classes', 4)
        train_loader, val_loader = self.create_data_loaders(num_classes=num_classes)
        
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        self.logger.info(f"Number of classes: {num_classes}")
        
        # Training loop
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Log progress
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save checkpoint if best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                
                checkpoint_path = save_path / "best_model.pth"
                save_model_checkpoint(
                    model=self.model,
                    checkpoint_path=checkpoint_path,
                    epoch=epoch,
                    loss=val_loss,
                    metrics={'accuracy': val_acc, 'train_loss': train_loss},
                    optimizer=self.optimizer
                )
                self.logger.info(f"Saved best model to {checkpoint_path}")
            
            # Save latest checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = save_path / f"checkpoint_epoch_{epoch+1}.pth"
                save_model_checkpoint(
                    model=self.model,
                    checkpoint_path=checkpoint_path,
                    epoch=epoch,
                    loss=val_loss,
                    metrics={'accuracy': val_acc, 'train_loss': train_loss},
                    optimizer=self.optimizer
                )
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ECG classification model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, auto)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path (optional)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Override config with command line arguments
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'val_split': 0.2
    }
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create model
    try:
        model = create_classifier(config, num_classes=args.num_classes)
        logger.info("Created model successfully")
        
        # Print model summary
        summary = get_model_summary(model)
        logger.info(f"Model summary:\n{summary}")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        sys.exit(1)
    
    # Create trainer
    trainer = ModelTrainer(model, training_config, device)
    
    # Start training
    try:
        start_time = datetime.now()
        history = trainer.train(save_dir=args.save_dir)
        end_time = datetime.now()
        
        training_time = end_time - start_time
        logger.info(f"Total training time: {training_time}")
        
        # Save training history
        history_path = Path(args.save_dir) / "training_history.yaml"
        with open(history_path, 'w') as f:
            yaml.dump(history, f, default_flow_style=False)
        logger.info(f"Saved training history to {history_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
