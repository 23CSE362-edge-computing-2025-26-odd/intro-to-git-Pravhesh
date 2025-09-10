#!/usr/bin/env python3
"""
Comprehensive model evaluation script for ECG classification.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ci.model.model import create_model, load_model_checkpoint
from src.ci.evaluation import ModelEvaluator
from scripts.train_model import ECGDataset  # Reuse dummy dataset for now


def setup_logging(log_level: str = "INFO", log_file: str = None):
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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_evaluation_dataloader(num_samples: int = 500, num_classes: int = 4, batch_size: int = 32) -> DataLoader:
    """Create DataLoader for evaluation (using dummy data for now)."""
    dataset = ECGDataset(num_samples, num_classes)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return dataloader


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate ECG classification model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to model configuration')
    
    # Evaluation arguments
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='Number of evaluation samples')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save individual predictions')
    
    # Device and logging
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, auto)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    # Cross-validation
    parser.add_argument('--cross-validate', action='store_true',
                        help='Perform cross-validation evaluation')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of cross-validation folds')
    
    # Class names
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['Normal', 'Abnormal_1', 'Abnormal_2', 'Critical'],
                        help='List of class names')
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.output_dir) / f"evaluation_log_{timestamp}.log"
    setup_logging(args.log_level, str(log_file))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Load model
    try:
        model = create_model(config, num_classes=args.num_classes)
        
        # Load checkpoint
        checkpoint_info = load_model_checkpoint(
            model, 
            args.checkpoint, 
            map_location=device
        )
        
        logger.info(f"Loaded model from checkpoint: {args.checkpoint}")
        logger.info(f"Checkpoint info: {checkpoint_info}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        class_names=args.class_names
    )
    
    logger.info(f"Created evaluator with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Perform evaluation
    if args.cross_validate:
        logger.info(f"Performing {args.num_folds}-fold cross-validation...")
        
        # Create multiple dataloaders for cross-validation
        dataloaders = []
        for fold in range(args.num_folds):
            dataloader = create_evaluation_dataloader(
                args.num_samples // args.num_folds,
                args.num_classes,
                args.batch_size
            )
            dataloaders.append(dataloader)
        
        # Perform cross-validation
        cv_results = evaluator.cross_validate(dataloaders)
        
        # Calculate aggregate metrics
        aggregate_metrics = evaluator.calculate_aggregate_metrics(cv_results)
        
        # Save cross-validation results
        cv_results_path = output_dir / f"cross_validation_results_{timestamp}.yaml"
        
        # Convert metrics to serializable format
        cv_results_serializable = {}
        for fold_name, metrics in cv_results.items():
            cv_results_serializable[fold_name] = {
                'accuracy': float(metrics.accuracy),
                'precision': float(metrics.precision),
                'recall': float(metrics.recall),
                'f1_score': float(metrics.f1_score),
                'roc_auc': float(metrics.roc_auc) if metrics.roc_auc else None
            }
        
        cv_output = {
            'cross_validation_results': cv_results_serializable,
            'aggregate_metrics': aggregate_metrics,
            'num_folds': args.num_folds,
            'evaluation_config': vars(args)
        }
        
        with open(cv_results_path, 'w') as f:
            yaml.dump(cv_output, f, default_flow_style=False)
        
        logger.info(f"Cross-validation results saved to: {cv_results_path}")
        
        # Print summary
        logger.info("Cross-validation summary:")
        for metric_name, stats in aggregate_metrics.items():
            if 'mean' in stats:
                logger.info(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    else:
        # Single evaluation
        logger.info("Performing single evaluation...")
        
        # Create evaluation dataloader
        eval_dataloader = create_evaluation_dataloader(
            args.num_samples,
            args.num_classes,
            args.batch_size
        )
        
        logger.info(f"Evaluating on {args.num_samples} samples...")
        
        # Perform evaluation
        metrics, predictions = evaluator.evaluate_dataloader(
            eval_dataloader,
            return_predictions=args.save_predictions,
            progress_bar=True
        )
        
        logger.info("Evaluation completed!")
        logger.info(f"Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"F1-Score: {metrics.f1_score:.4f}")
        logger.info(f"ROC-AUC: {metrics.roc_auc:.4f}" if metrics.roc_auc else "ROC-AUC: N/A")
        
        # Generate comprehensive report
        report_path = output_dir / f"evaluation_report_{timestamp}.txt"
        plot_dir = output_dir / f"plots_{timestamp}"
        
        report_text = evaluator.generate_evaluation_report(
            metrics,
            save_path=str(report_path),
            include_plots=True,
            plot_dir=str(plot_dir)
        )
        
        logger.info(f"Evaluation report saved to: {report_path}")
        logger.info(f"Evaluation plots saved to: {plot_dir}")
        
        # Save detailed results
        results_path = output_dir / f"evaluation_results_{timestamp}.yaml"
        evaluator.save_evaluation_results(metrics, predictions, str(results_path))
        
        logger.info(f"Detailed results saved to: {results_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {model.__class__.__name__}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Samples evaluated: {args.num_samples}")
        print(f"Classes: {args.num_classes}")
        print("-" * 40)
        print(f"Accuracy:     {metrics.accuracy:.4f}")
        print(f"Precision:    {metrics.precision:.4f}")
        print(f"Recall:       {metrics.recall:.4f}")
        print(f"F1-Score:     {metrics.f1_score:.4f}")
        if metrics.roc_auc:
            print(f"ROC-AUC:      {metrics.roc_auc:.4f}")
        print("-" * 40)
        print(f"Results saved in: {output_dir}")
        print("="*60)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
