#!/usr/bin/env python3
"""
ECG prediction script for single-file or batch inference.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
import yaml

import torch
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ci.model.model import create_model, load_model_checkpoint
from src.ci.evaluation.inference import ECGInferenceEngine, InferenceConfig
from src.ci.deployment.serve import ModelServer, create_server_from_checkpoint


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_signal_from_csv(file_path: str, value_column: str = 'value') -> np.ndarray:
    """Load ECG signal from CSV file."""
    df = pd.read_csv(file_path)
    
    if value_column not in df.columns:
        # Try to infer the value column
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 1:
            value_column = numeric_columns[0]
        elif 'ii' in df.columns:  # Common ECG lead name
            value_column = 'ii'
        elif 'value' in df.columns:
            value_column = 'value'
        else:
            raise ValueError(f"Could not find value column. Available columns: {list(df.columns)}")
    
    signal = df[value_column].values
    return signal.astype(np.float32)


def load_signal_from_txt(file_path: str) -> np.ndarray:
    """Load ECG signal from text file."""
    signal = np.loadtxt(file_path)
    return signal.astype(np.float32)


def load_signal_from_file(file_path: str, value_column: str = 'value') -> np.ndarray:
    """Load ECG signal from file (auto-detect format)."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() == '.csv':
        return load_signal_from_csv(str(file_path), value_column)
    elif file_path.suffix.lower() in ['.txt', '.dat']:
        return load_signal_from_txt(str(file_path))
    else:
        # Try to load as text first, then CSV
        try:
            return load_signal_from_txt(str(file_path))
        except:
            try:
                return load_signal_from_csv(str(file_path), value_column)
            except:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_prediction_results(results: dict, output_path: str):
    """Save prediction results to file."""
    output_path = Path(output_path)
    
    if output_path.suffix.lower() == '.json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif output_path.suffix.lower() in ['.yaml', '.yml']:
        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        # Default to JSON
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict ECG classification')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to model configuration')
    
    # Input arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory containing ECG data')
    parser.add_argument('--value-column', type=str, default='value',
                        help='Column name for ECG values (for CSV files)')
    parser.add_argument('--batch', action='store_true',
                        help='Process multiple files in directory')
    
    # Model parameters
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['Normal', 'Abnormal_1', 'Abnormal_2', 'Critical'],
                        help='List of class names')
    
    # Inference configuration
    parser.add_argument('--window-seconds', type=float, default=5.0,
                        help='Window size in seconds')
    parser.add_argument('--overlap-fraction', type=float, default=0.5,
                        help='Overlap fraction between windows')
    parser.add_argument('--sampling-rate', type=int, default=360,
                        help='Sampling rate in Hz')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: auto-generated)')
    parser.add_argument('--output-format', type=str, choices=['json', 'yaml'], default='json',
                        help='Output file format')
    
    # Device and logging
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, auto)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    # Server mode
    parser.add_argument('--server-mode', action='store_true',
                        help='Run in server mode for batch processing')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create inference configuration
    inference_config = InferenceConfig(
        window_seconds=args.window_seconds,
        overlap_fraction=args.overlap_fraction,
        sampling_rate_hz=args.sampling_rate
    )
    
    # Load model and create inference engine
    try:
        if args.server_mode:
            # Use server mode for better performance with batch processing
            from src.ci.model import FeatureExtractor
            
            server = create_server_from_checkpoint(
                checkpoint_path=args.checkpoint,
                model_class=FeatureExtractor,
                model_kwargs={'num_classes': args.num_classes},
                inference_config=inference_config,
                class_names=args.class_names,
                device=device
            )
            
            logger.info("Created model server")
            logger.info(f"Server health: {server.health_check()['status']}")
            
        else:
            # Direct inference engine
            model = create_model(config, num_classes=args.num_classes)
            load_model_checkpoint(model, args.checkpoint, map_location=device)
            
            inference_engine = ECGInferenceEngine(
                model=model,
                inference_config=inference_config,
                device=device,
                class_names=args.class_names
            )
            
            logger.info("Created inference engine")
        
    except Exception as e:
        logger.error(f"Failed to create inference engine: {e}")
        sys.exit(1)
    
    # Process input
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # Batch processing
        logger.info(f"Processing directory: {input_path}")
        
        if not input_path.is_dir():
            logger.error(f"Input path is not a directory: {input_path}")
            sys.exit(1)
        
        # Find all ECG files
        ecg_files = []
        for pattern in ['*.csv', '*.txt', '*.dat']:
            ecg_files.extend(input_path.glob(pattern))
        
        if not ecg_files:
            logger.error(f"No ECG files found in directory: {input_path}")
            sys.exit(1)
        
        logger.info(f"Found {len(ecg_files)} files to process")
        
        # Process each file
        batch_results = []
        
        for i, file_path in enumerate(ecg_files):
            logger.info(f"Processing file {i+1}/{len(ecg_files)}: {file_path.name}")
            
            try:
                # Load signal
                signal = load_signal_from_file(str(file_path), args.value_column)
                logger.debug(f"Loaded signal with {len(signal)} samples")
                
                # Make prediction
                if args.server_mode:
                    prediction_result = server.predict(signal)
                    prediction = prediction_result['prediction'] if prediction_result['success'] else {'error': prediction_result['error']}
                else:
                    prediction = inference_engine.predict_signal(signal)
                
                # Store result
                file_result = {
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'signal_length': len(signal),
                    'prediction': prediction,
                    'timestamp': datetime.now().isoformat()
                }
                
                batch_results.append(file_result)
                
                # Log prediction
                if 'error' not in prediction:
                    logger.info(f"  Prediction: {prediction['final_prediction']} (confidence: {max(prediction['final_probabilities']):.3f})")
                else:
                    logger.warning(f"  Prediction failed: {prediction['error']}")
            
            except Exception as e:
                logger.error(f"  Failed to process {file_path.name}: {str(e)}")
                batch_results.append({
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Prepare output
        results = {
            'batch_processing': True,
            'input_directory': str(input_path),
            'files_processed': len(ecg_files),
            'successful_predictions': sum(1 for r in batch_results if 'error' not in r and 'error' not in r.get('prediction', {})),
            'failed_predictions': len(batch_results) - sum(1 for r in batch_results if 'error' not in r and 'error' not in r.get('prediction', {})),
            'results': batch_results,
            'inference_config': {
                'window_seconds': args.window_seconds,
                'overlap_fraction': args.overlap_fraction,
                'sampling_rate_hz': args.sampling_rate,
                'num_classes': args.num_classes,
                'class_names': args.class_names
            },
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Add server statistics if in server mode
        if args.server_mode:
            results['server_stats'] = server.get_stats()
    
    else:
        # Single file processing
        logger.info(f"Processing single file: {input_path}")
        
        if not input_path.is_file():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        try:
            # Load signal
            signal = load_signal_from_file(str(input_path), args.value_column)
            logger.info(f"Loaded signal with {len(signal)} samples ({len(signal)/args.sampling_rate:.1f} seconds)")
            
            # Make prediction
            start_time = datetime.now()
            
            if args.server_mode:
                prediction_result = server.predict(signal)
                prediction = prediction_result['prediction'] if prediction_result['success'] else {'error': prediction_result['error']}
            else:
                prediction = inference_engine.predict_signal(signal)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Prepare results
            results = {
                'single_file_processing': True,
                'input_file': str(input_path),
                'signal_length': len(signal),
                'signal_duration_seconds': len(signal) / args.sampling_rate,
                'prediction': prediction,
                'processing_time_seconds': processing_time,
                'inference_config': {
                    'window_seconds': args.window_seconds,
                    'overlap_fraction': args.overlap_fraction,
                    'sampling_rate_hz': args.sampling_rate,
                    'num_classes': args.num_classes,
                    'class_names': args.class_names
                },
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Log prediction
            if 'error' not in prediction:
                logger.info(f"Prediction: {prediction['final_prediction']}")
                logger.info(f"Confidence: {max(prediction['final_probabilities']):.3f}")
                logger.info(f"Windows processed: {prediction['num_windows']}")
                logger.info(f"Processing time: {processing_time:.3f} seconds")
                
                # Print detailed results
                print("\n" + "="*60)
                print("PREDICTION RESULTS")
                print("="*60)
                print(f"File: {input_path.name}")
                print(f"Signal length: {len(signal)} samples ({len(signal)/args.sampling_rate:.1f}s)")
                print(f"Prediction: {prediction['final_prediction']}")
                print(f"Confidence: {max(prediction['final_probabilities']):.3f}")
                print(f"Windows: {prediction['num_windows']}")
                print("-" * 40)
                print("Class probabilities:")
                for i, prob in enumerate(prediction['final_probabilities']):
                    class_name = args.class_names[i] if i < len(args.class_names) else f'Class_{i}'
                    print(f"  {class_name}: {prob:.4f}")
                print("="*60)
                
            else:
                logger.error(f"Prediction failed: {prediction['error']}")
        
        except Exception as e:
            logger.error(f"Failed to process file: {str(e)}")
            results = {
                'single_file_processing': True,
                'input_file': str(input_path),
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.batch or input_path.is_dir():
            output_path = Path(f"batch_predictions_{timestamp}.{args.output_format}")
        else:
            output_path = Path(f"prediction_{input_path.stem}_{timestamp}.{args.output_format}")
    
    save_prediction_results(results, str(output_path))
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
