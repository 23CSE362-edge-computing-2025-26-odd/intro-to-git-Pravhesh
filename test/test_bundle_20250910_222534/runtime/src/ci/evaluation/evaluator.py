"""
Model evaluation pipeline for ECG classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import yaml
import logging
from tqdm import tqdm

from .metrics import ClassificationMetrics, MetricsCalculator, calculate_metrics_from_tensors


class ModelEvaluator:
    """
    Comprehensive model evaluator for ECG classification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: PyTorch model to evaluate
            device: Device to run evaluation on
            class_names: List of class names for labeling
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.logger = logging.getLogger(__name__)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(class_names)
    
    def evaluate_dataloader(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
        progress_bar: bool = True
    ) -> Tuple[ClassificationMetrics, Optional[Dict]]:
        """
        Evaluate model on a DataLoader.
        
        Args:
            dataloader: DataLoader containing evaluation data
            return_predictions: Whether to return individual predictions
            progress_bar: Whether to show progress bar
            
        Returns:
            Tuple of (ClassificationMetrics, predictions dict if requested)
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_losses = []
        
        # Loss function for evaluation
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            iterator = tqdm(dataloader, desc="Evaluating") if progress_bar else dataloader
            
            for batch_idx, (data, targets) in enumerate(iterator):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_probabilities.append(probabilities.cpu())
                all_labels.append(targets.cpu())
                all_losses.append(loss.item())
                
                if progress_bar and isinstance(iterator, tqdm):
                    iterator.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Batch': f'{batch_idx + 1}/{len(dataloader)}'
                    })
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_probabilities = torch.cat(all_probabilities)
        all_labels = torch.cat(all_labels)
        avg_loss = np.mean(all_losses)
        
        # Calculate metrics
        metrics = calculate_metrics_from_tensors(
            all_labels,
            all_predictions,
            all_probabilities,
            self.class_names
        )
        
        # Prepare predictions dict if requested
        predictions_dict = None
        if return_predictions:
            predictions_dict = {
                'predictions': all_predictions.numpy(),
                'probabilities': all_probabilities.numpy(),
                'labels': all_labels.numpy(),
                'losses': all_losses,
                'average_loss': avg_loss
            }
        
        self.logger.info(f"Evaluation completed. Average loss: {avg_loss:.4f}, Accuracy: {metrics.accuracy:.4f}")
        
        return metrics, predictions_dict
    
    def evaluate_single_batch(
        self,
        data: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[ClassificationMetrics, Dict]:
        """
        Evaluate model on a single batch.
        
        Args:
            data: Input data tensor
            targets: Target labels tensor
            
        Returns:
            Tuple of (ClassificationMetrics, predictions dict)
        """
        data, targets = data.to(self.device), targets.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(data)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Calculate loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, targets)
        
        # Calculate metrics
        metrics = calculate_metrics_from_tensors(
            targets,
            predictions,
            probabilities,
            self.class_names
        )
        
        predictions_dict = {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'labels': targets.cpu().numpy(),
            'loss': loss.item()
        }
        
        return metrics, predictions_dict
    
    def cross_validate(
        self,
        dataloaders: List[DataLoader],
        fold_names: Optional[List[str]] = None
    ) -> Dict[str, ClassificationMetrics]:
        """
        Perform cross-validation evaluation.
        
        Args:
            dataloaders: List of DataLoaders for each fold
            fold_names: Names for each fold (optional)
            
        Returns:
            Dictionary mapping fold names to metrics
        """
        if fold_names is None:
            fold_names = [f"Fold_{i+1}" for i in range(len(dataloaders))]
        
        results = {}
        
        for i, (fold_name, dataloader) in enumerate(zip(fold_names, dataloaders)):
            self.logger.info(f"Evaluating {fold_name}...")
            
            metrics, _ = self.evaluate_dataloader(
                dataloader,
                return_predictions=False,
                progress_bar=True
            )
            
            results[fold_name] = metrics
            
            self.logger.info(f"{fold_name} - Accuracy: {metrics.accuracy:.4f}, F1: {metrics.f1_score:.4f}")
        
        return results
    
    def calculate_aggregate_metrics(
        self,
        cv_results: Dict[str, ClassificationMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate aggregate metrics across cross-validation folds.
        
        Args:
            cv_results: Results from cross_validate method
            
        Returns:
            Dictionary with mean and std statistics for each metric
        """
        metric_names = [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
            'macro_precision', 'macro_recall', 'macro_f1',
            'weighted_precision', 'weighted_recall', 'weighted_f1'
        ]
        
        aggregate_stats = {}
        
        for metric_name in metric_names:
            values = []
            for fold_metrics in cv_results.values():
                metric_value = getattr(fold_metrics, metric_name)
                if metric_value is not None:
                    values.append(metric_value)
            
            if values:
                aggregate_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        return aggregate_stats
    
    def generate_evaluation_report(
        self,
        metrics: ClassificationMetrics,
        save_path: Optional[str] = None,
        include_plots: bool = True,
        plot_dir: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: ClassificationMetrics object
            save_path: Path to save the report
            include_plots: Whether to generate and include plots
            plot_dir: Directory to save plots
            
        Returns:
            Report text
        """
        # Generate metrics summary
        report = self.metrics_calculator.create_metrics_summary(metrics)
        
        # Add model information
        model_info = f"""

Model Information:
================
- Architecture: {self.model.__class__.__name__}
- Device: {self.device}
- Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}
- Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}
"""
        report = model_info + report
        
        # Generate plots if requested
        if include_plots:
            if plot_dir is None:
                plot_dir = "evaluation_plots"
            
            plot_path = Path(plot_dir)
            plot_path.mkdir(exist_ok=True)
            
            plot_info = "\n\nGenerated Plots:\n===============\n"
            
            # Confusion matrix
            cm_path = plot_path / "confusion_matrix.png"
            self.metrics_calculator.plot_confusion_matrix(metrics, save_path=str(cm_path))
            plot_info += f"- Confusion Matrix: {cm_path}\n"
            
            # Normalized confusion matrix
            cm_norm_path = plot_path / "confusion_matrix_normalized.png"
            self.metrics_calculator.plot_confusion_matrix(metrics, normalize=True, save_path=str(cm_norm_path))
            plot_info += f"- Normalized Confusion Matrix: {cm_norm_path}\n"
            
            # ROC curves
            roc_path = plot_path / "roc_curves.png"
            roc_fig = self.metrics_calculator.plot_roc_curves(metrics, save_path=str(roc_path))
            if roc_fig is not None:
                plot_info += f"- ROC Curves: {roc_path}\n"
            
            # PR curves
            pr_path = plot_path / "precision_recall_curves.png"
            pr_fig = self.metrics_calculator.plot_precision_recall_curves(metrics, save_path=str(pr_path))
            if pr_fig is not None:
                plot_info += f"- Precision-Recall Curves: {pr_path}\n"
            
            report += plot_info
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Evaluation report saved to: {save_path}")
        
        return report
    
    def compare_models(
        self,
        other_evaluators: List['ModelEvaluator'],
        dataloader: DataLoader,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, ClassificationMetrics]:
        """
        Compare multiple models on the same dataset.
        
        Args:
            other_evaluators: List of other ModelEvaluator instances
            dataloader: DataLoader for comparison
            model_names: Names for each model (including this one)
            
        Returns:
            Dictionary mapping model names to metrics
        """
        all_evaluators = [self] + other_evaluators
        
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(all_evaluators))]
        
        comparison_results = {}
        
        for evaluator, model_name in zip(all_evaluators, model_names):
            self.logger.info(f"Evaluating {model_name}...")
            
            metrics, _ = evaluator.evaluate_dataloader(
                dataloader,
                return_predictions=False,
                progress_bar=True
            )
            
            comparison_results[model_name] = metrics
            
            self.logger.info(f"{model_name} - Accuracy: {metrics.accuracy:.4f}")
        
        return comparison_results
    
    def save_evaluation_results(
        self,
        metrics: ClassificationMetrics,
        predictions_dict: Optional[Dict],
        save_path: str
    ):
        """
        Save evaluation results to file.
        
        Args:
            metrics: ClassificationMetrics object
            predictions_dict: Predictions dictionary (optional)
            save_path: Path to save results
        """
        results = {
            'metrics': {
                'accuracy': float(metrics.accuracy),
                'precision': float(metrics.precision),
                'recall': float(metrics.recall),
                'f1_score': float(metrics.f1_score),
                'roc_auc': float(metrics.roc_auc) if metrics.roc_auc else None,
                'macro_precision': float(metrics.macro_precision),
                'macro_recall': float(metrics.macro_recall),
                'macro_f1': float(metrics.macro_f1),
                'weighted_precision': float(metrics.weighted_precision),
                'weighted_recall': float(metrics.weighted_recall),
                'weighted_f1': float(metrics.weighted_f1),
                'per_class_precision': metrics.per_class_precision.tolist(),
                'per_class_recall': metrics.per_class_recall.tolist(),
                'per_class_f1': metrics.per_class_f1.tolist(),
                'confusion_matrix': metrics.confusion_matrix.tolist(),
                'classification_report': metrics.classification_report
            },
            'model_info': {
                'architecture': self.model.__class__.__name__,
                'device': self.device,
                'class_names': self.class_names,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        if predictions_dict is not None:
            results['predictions'] = {
                'predictions': predictions_dict['predictions'].tolist(),
                'probabilities': predictions_dict['probabilities'].tolist(),
                'labels': predictions_dict['labels'].tolist(),
                'average_loss': predictions_dict.get('average_loss', None)
            }
        
        with open(save_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        self.logger.info(f"Evaluation results saved to: {save_path}")


def load_evaluation_results(file_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        Dictionary containing evaluation results
    """
    with open(file_path, 'r') as f:
        results = yaml.safe_load(f)
    
    return results
