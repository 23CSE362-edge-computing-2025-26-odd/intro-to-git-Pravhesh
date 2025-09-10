#!/usr/bin/env python3
"""
Extract ResNet18 embeddings from ECG spectrograms and save them for downstream tasks.

This script loads preprocessed ECG spectrograms, extracts 512-dimensional feature
embeddings using a ResNet18 backbone, and saves the embeddings along with metadata
for use in fuzzy logic systems and other downstream tasks.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ci.data.loader import ECGDataLoader
from ci.preprocess.pipeline import PreprocessingPipeline
from ci.model.feature_extractor import FeatureExtractor
from ci.model.model import create_feature_extractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extracts embeddings from ECG spectrograms using pre-trained or fine-tuned models."""
    
    def __init__(self, config_path: str, checkpoint_path: Optional[str] = None):
        """
        Initialize the embedding extractor.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Optional path to model checkpoint
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._create_model()
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessing pipeline
        self.preprocessor = PreprocessingPipeline(self.config)
        
    def _create_model(self) -> nn.Module:
        """Create feature extractor model from config."""
        model_config = self.config.get('model', {})
        
        # Create feature extractor (no classification head)
        model = FeatureExtractor(
            backbone=model_config.get('backbone', 'resnet18'),
            embedding_dim=model_config.get('embedding_dim', 512),
            pretrained=model_config.get('pretrained', True),
            finetune_last_block=model_config.get('finetune_last_block', False),
            num_classes=None  # Feature extraction mode
        )
        
        logger.info(f"Created {model_config.get('backbone', 'resnet18')} feature extractor")
        return model
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Using pretrained weights instead")
    
    def extract_from_data_loader(self, data_loader: ECGDataLoader, 
                                output_path: str, batch_size: int = 32) -> Dict:
        """
        Extract embeddings from data loader and save to file.
        
        Args:
            data_loader: ECG data loader instance
            output_path: Path to save embeddings
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with extraction statistics
        """
        # Get all files and their metadata
        all_files = []
        all_labels = []
        
        for dataset_name in data_loader.datasets:
            files = data_loader.get_dataset_files(dataset_name)
            for file_path in files:
                try:
                    metadata = data_loader.get_file_metadata(file_path)
                    all_files.append(file_path)
                    all_labels.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to get metadata for {file_path}: {e}")
        
        logger.info(f"Processing {len(all_files)} files")
        
        # Process files in batches
        embeddings_list = []
        metadata_list = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(all_files), batch_size), desc="Extracting embeddings"):
                batch_files = all_files[i:i+batch_size]
                batch_labels = all_labels[i:i+batch_size]
                
                batch_embeddings = []
                batch_metadata = []
                
                for file_path, metadata in zip(batch_files, batch_labels):
                    try:
                        # Load and preprocess ECG data
                        ecg_data = data_loader.load_ecg_file(file_path)
                        
                        # Get spectrograms from preprocessing pipeline
                        spectrograms = self.preprocessor.process_ecg(ecg_data, metadata['sampling_rate'])
                        
                        if len(spectrograms) == 0:
                            logger.warning(f"No spectrograms generated for {file_path}")
                            continue
                        
                        # Convert to tensor and extract embeddings
                        spectrogram_tensor = torch.from_numpy(np.array(spectrograms)).float()
                        spectrogram_tensor = spectrogram_tensor.to(self.device)
                        
                        # Extract embeddings for all windows
                        file_embeddings = self.model(spectrogram_tensor)
                        file_embeddings = file_embeddings.cpu().numpy()
                        
                        # Store embeddings and metadata for each window
                        for window_idx, embedding in enumerate(file_embeddings):
                            batch_embeddings.append(embedding)
                            window_metadata = metadata.copy()
                            window_metadata.update({
                                'file_path': str(file_path),
                                'window_index': window_idx,
                                'total_windows': len(file_embeddings),
                                'embedding_dim': embedding.shape[0]
                            })
                            batch_metadata.append(window_metadata)
                        
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                        continue
                
                if batch_embeddings:
                    embeddings_list.extend(batch_embeddings)
                    metadata_list.extend(batch_metadata)
        
        if not embeddings_list:
            raise ValueError("No embeddings extracted")
        
        # Convert to arrays
        embeddings_array = np.array(embeddings_list)
        logger.info(f"Extracted {len(embeddings_array)} embeddings of dimension {embeddings_array.shape[1]}")
        
        # Save to HDF5 file
        self._save_embeddings(embeddings_array, metadata_list, output_path)
        
        # Return statistics
        stats = {
            'num_embeddings': len(embeddings_array),
            'embedding_dim': embeddings_array.shape[1],
            'num_unique_files': len(set(m['file_path'] for m in metadata_list)),
            'label_distribution': self._get_label_distribution(metadata_list),
            'output_path': output_path
        }
        
        return stats
    
    def _save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict], output_path: str):
        """Save embeddings and metadata to HDF5 file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Save embeddings
            f.create_dataset('embeddings', data=embeddings, compression='gzip')
            
            # Save metadata as structured arrays
            if metadata:
                # Convert metadata to pandas DataFrame for easier handling
                df = pd.DataFrame(metadata)
                
                # Save each column separately
                metadata_group = f.create_group('metadata')
                for col in df.columns:
                    if col == 'file_path':
                        # Handle string data
                        str_data = [str(x).encode('utf-8') for x in df[col].values]
                        metadata_group.create_dataset(col, data=str_data)
                    elif col in ['label', 'diagnosis', 'patient_id']:
                        # Handle string labels
                        str_data = [str(x).encode('utf-8') for x in df[col].values]
                        metadata_group.create_dataset(col, data=str_data)
                    else:
                        # Handle numeric data
                        try:
                            numeric_data = pd.to_numeric(df[col], errors='coerce').fillna(0)
                            metadata_group.create_dataset(col, data=numeric_data.values)
                        except:
                            # Fallback to string
                            str_data = [str(x).encode('utf-8') for x in df[col].values]
                            metadata_group.create_dataset(col, data=str_data)
                
                # Save configuration
                config_group = f.create_group('config')
                config_str = yaml.dump(self.config).encode('utf-8')
                config_group.create_dataset('config_yaml', data=config_str)
        
        logger.info(f"Saved embeddings to {output_path}")
    
    def _get_label_distribution(self, metadata: List[Dict]) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        labels = [m.get('label', 'unknown') for m in metadata]
        from collections import Counter
        return dict(Counter(labels))


def load_embeddings(file_path: str) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    """
    Load embeddings and metadata from HDF5 file.
    
    Args:
        file_path: Path to embeddings file
        
    Returns:
        Tuple of (embeddings, metadata_df, config)
    """
    with h5py.File(file_path, 'r') as f:
        # Load embeddings
        embeddings = f['embeddings'][:]
        
        # Load metadata
        metadata_dict = {}
        metadata_group = f['metadata']
        for key in metadata_group.keys():
            data = metadata_group[key][:]
            if data.dtype.kind in ['S', 'U']:  # String data
                data = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in data]
            metadata_dict[key] = data
        
        metadata_df = pd.DataFrame(metadata_dict)
        
        # Load config
        config_str = f['config/config_yaml'][()].decode('utf-8')
        config = yaml.safe_load(config_str)
    
    return embeddings, metadata_df, config


def main():
    parser = argparse.ArgumentParser(description='Extract ResNet18 embeddings from ECG spectrograms')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--output', type=str, default='embeddings/ecg_embeddings.h5',
                       help='Output path for embeddings file')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for processing')
    parser.add_argument('--dataset', type=str, choices=['ptb', 'mitbih', 'all'], default='all',
                       help='Dataset to process')
    
    args = parser.parse_args()
    
    # Initialize extractor
    logger.info("Initializing embedding extractor...")
    extractor = EmbeddingExtractor(args.config, args.checkpoint)
    
    # Initialize data loader
    logger.info("Initializing data loader...")
    data_loader = ECGDataLoader(args.config)
    
    # Filter dataset if specified
    if args.dataset != 'all':
        # Filter to specific dataset
        datasets_to_use = [args.dataset] if args.dataset in data_loader.datasets else []
        if not datasets_to_use:
            logger.error(f"Dataset '{args.dataset}' not found. Available: {list(data_loader.datasets.keys())}")
            return
        data_loader.datasets = {k: v for k, v in data_loader.datasets.items() if k in datasets_to_use}
    
    # Extract embeddings
    logger.info("Starting embedding extraction...")
    stats = extractor.extract_from_data_loader(data_loader, args.output, args.batch_size)
    
    # Print statistics
    logger.info("Embedding extraction completed!")
    logger.info(f"Statistics: {stats}")
    
    # Verify saved file
    try:
        embeddings, metadata_df, config = load_embeddings(args.output)
        logger.info(f"Verification successful - loaded {embeddings.shape[0]} embeddings")
        logger.info(f"Label distribution: {dict(metadata_df['label'].value_counts())}")
    except Exception as e:
        logger.error(f"Failed to verify saved embeddings: {e}")


if __name__ == "__main__":
    main()
