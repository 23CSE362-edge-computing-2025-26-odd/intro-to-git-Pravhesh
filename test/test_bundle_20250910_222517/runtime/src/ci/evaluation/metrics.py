"""
Metrics calculation for ECG classification evaluation.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ClassificationMetrics:
    """
    Container for classification metrics results.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    confusion_matrix: np.ndarray
    classification_report: str
    
    # Per-class metrics
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    
    # Additional metrics
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    
    # ROC/PR curve data
    roc_curves: Optional[Dict] = None
    pr_curves: Optional[Dict] = None


class MetricsCalculator:
    """
    Calculator for comprehensive classification metrics.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names for labeling
        """
        self.class_names = class_names
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> ClassificationMetrics:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
            average: Averaging strategy for multi-class metrics
            
        Returns:
            ClassificationMetrics object with all computed metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Averaged metrics
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Different averaging strategies
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        target_names = self.class_names if self.class_names else None
        n_unique_classes = len(np.unique(np.concatenate([y_true, y_pred])))
        
        # Adjust target_names to match actual number of classes
        if target_names is not None and len(target_names) != n_unique_classes:
            target_names = target_names[:n_unique_classes] if len(target_names) > n_unique_classes else None
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        
        # ROC-AUC (if probabilities provided)
        roc_auc = None
        roc_curves = None
        pr_curves = None
        
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    # Binary classification
                    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    
                    # ROC curve
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                    roc_curves = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
                    
                    # PR curve
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
                    pr_auc = auc(recall_curve, precision_curve)
                    pr_curves = {'precision': precision_curve, 'recall': recall_curve, 'auc': pr_auc}
                    
                elif n_classes > 2:
                    # Multi-class classification
                    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    
                    # Per-class ROC curves
                    roc_curves = {}
                    pr_curves = {}
                    
                    for i in range(n_classes):
                        # Binary classification for class i vs rest
                        y_true_binary = (y_true == i).astype(int)
                        y_pred_binary = y_pred_proba[:, i]
                        
                        # ROC curve
                        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
                        class_roc_auc = auc(fpr, tpr)
                        
                        class_name = self.class_names[i] if self.class_names else f'Class_{i}'
                        roc_curves[class_name] = {'fpr': fpr, 'tpr': tpr, 'auc': class_roc_auc}
                        
                        # PR curve
                        precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_pred_binary)
                        pr_auc = auc(recall_curve, precision_curve)
                        pr_curves[class_name] = {'precision': precision_curve, 'recall': recall_curve, 'auc': pr_auc}
                        
            except ValueError as e:
                print(f"Warning: Could not calculate ROC-AUC: {e}")
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm,
            classification_report=report,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            weighted_precision=weighted_precision,
            weighted_recall=weighted_recall,
            weighted_f1=weighted_f1,
            roc_curves=roc_curves,
            pr_curves=pr_curves
        )
    
    def plot_confusion_matrix(
        self,
        metrics: ClassificationMetrics,
        normalize: bool = False,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            metrics: ClassificationMetrics object
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        cm = metrics.confusion_matrix
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.class_names or range(len(cm)),
            yticklabels=self.class_names or range(len(cm)),
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(
        self,
        metrics: ClassificationMetrics,
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot ROC curves.
        
        Args:
            metrics: ClassificationMetrics object
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure or None if no ROC data
        """
        if metrics.roc_curves is None:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if isinstance(metrics.roc_curves, dict) and 'fpr' in metrics.roc_curves:
            # Binary classification
            fpr = metrics.roc_curves['fpr']
            tpr = metrics.roc_curves['tpr']
            auc_score = metrics.roc_curves['auc']
            
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})', linewidth=2)
            
        else:
            # Multi-class classification
            for class_name, curve_data in metrics.roc_curves.items():
                fpr = curve_data['fpr']
                tpr = curve_data['tpr']
                auc_score = curve_data['auc']
                
                ax.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.2f})', linewidth=2)
        
        # Diagonal line for random classifier
        ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curves(
        self,
        metrics: ClassificationMetrics,
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot Precision-Recall curves.
        
        Args:
            metrics: ClassificationMetrics object
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure or None if no PR data
        """
        if metrics.pr_curves is None:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if isinstance(metrics.pr_curves, dict) and 'precision' in metrics.pr_curves:
            # Binary classification
            precision = metrics.pr_curves['precision']
            recall = metrics.pr_curves['recall']
            auc_score = metrics.pr_curves['auc']
            
            ax.plot(recall, precision, label=f'PR curve (AUC = {auc_score:.2f})', linewidth=2)
            
        else:
            # Multi-class classification
            for class_name, curve_data in metrics.pr_curves.items():
                precision = curve_data['precision']
                recall = curve_data['recall']
                auc_score = curve_data['auc']
                
                ax.plot(recall, precision, label=f'{class_name} (AUC = {auc_score:.2f})', linewidth=2)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_metrics_summary(self, metrics: ClassificationMetrics) -> str:
        """
        Create a formatted summary of all metrics.
        
        Args:
            metrics: ClassificationMetrics object
            
        Returns:
            Formatted metrics summary string
        """
        summary = f"""
ECG Classification Metrics Summary
================================

Overall Performance:
- Accuracy: {metrics.accuracy:.4f}
- Precision: {metrics.precision:.4f}
- Recall: {metrics.recall:.4f}
- F1-Score: {metrics.f1_score:.4f}
{f"- ROC-AUC: {metrics.roc_auc:.4f}" if metrics.roc_auc else ""}

Macro Averages:
- Macro Precision: {metrics.macro_precision:.4f}
- Macro Recall: {metrics.macro_recall:.4f}
- Macro F1-Score: {metrics.macro_f1:.4f}

Weighted Averages:
- Weighted Precision: {metrics.weighted_precision:.4f}
- Weighted Recall: {metrics.weighted_recall:.4f}
- Weighted F1-Score: {metrics.weighted_f1:.4f}

Per-Class Performance:
"""
        
        if self.class_names:
            for i, class_name in enumerate(self.class_names):
                if i < len(metrics.per_class_precision):
                    summary += f"- {class_name}:\n"
                    summary += f"  * Precision: {metrics.per_class_precision[i]:.4f}\n"
                    summary += f"  * Recall: {metrics.per_class_recall[i]:.4f}\n"
                    summary += f"  * F1-Score: {metrics.per_class_f1[i]:.4f}\n"
        else:
            for i in range(len(metrics.per_class_precision)):
                summary += f"- Class {i}:\n"
                summary += f"  * Precision: {metrics.per_class_precision[i]:.4f}\n"
                summary += f"  * Recall: {metrics.per_class_recall[i]:.4f}\n"
                summary += f"  * F1-Score: {metrics.per_class_f1[i]:.4f}\n"
        
        summary += f"\nDetailed Classification Report:\n{metrics.classification_report}"
        
        return summary


def calculate_metrics_from_tensors(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_pred_proba: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None
) -> ClassificationMetrics:
    """
    Convenience function to calculate metrics from PyTorch tensors.
    
    Args:
        y_true: True labels tensor
        y_pred: Predicted labels tensor
        y_pred_proba: Predicted probabilities tensor
        class_names: List of class names
        
    Returns:
        ClassificationMetrics object
    """
    # Convert tensors to numpy arrays
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    y_pred_proba_np = y_pred_proba.detach().cpu().numpy() if y_pred_proba is not None else None
    
    calculator = MetricsCalculator(class_names)
    return calculator.calculate_metrics(y_true_np, y_pred_np, y_pred_proba_np)
