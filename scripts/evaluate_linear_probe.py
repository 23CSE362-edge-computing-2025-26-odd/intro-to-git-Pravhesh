#!/usr/bin/env python3
"""
Linear probe baseline evaluation for ResNet18 embeddings.

This script loads extracted embeddings and trains/evaluates linear classifiers
on top of frozen ResNet18 features to establish baseline performance metrics.
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.extract_embeddings import load_embeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LinearProbeEvaluator:
    """Evaluates linear classifiers on extracted embeddings."""
    
    def __init__(self, embeddings_path: str, output_dir: str = "results/linear_probe"):
        """
        Initialize the linear probe evaluator.
        
        Args:
            embeddings_path: Path to embeddings HDF5 file
            output_dir: Directory to save results
        """
        self.embeddings_path = embeddings_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load embeddings and metadata
        logger.info(f"Loading embeddings from {embeddings_path}")
        self.embeddings, self.metadata_df, self.config = load_embeddings(embeddings_path)
        
        # Prepare data
        self.X, self.y, self.label_encoder = self._prepare_data()
        self.class_names = list(self.label_encoder.classes_)
        
        logger.info(f"Loaded {len(self.X)} samples with {self.X.shape[1]} features")
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Class distribution: {dict(pd.Series(self.y).value_counts())}")
    
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
        """Prepare embeddings and labels for classification."""
        # Get embeddings
        X = self.embeddings
        
        # Get labels
        labels = self.metadata_df['label'].values
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Save scaler for later use
        joblib.dump(scaler, self.output_dir / "feature_scaler.pkl")
        joblib.dump(label_encoder, self.output_dir / "label_encoder.pkl")
        
        return X, y, label_encoder
    
    def train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Split data into train/test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }
    
    def evaluate_classifier(self, classifier, X_train, X_test, y_train, y_test, name: str) -> Dict:
        """Evaluate a single classifier and return metrics."""
        logger.info(f"Training {name}...")
        
        # Train classifier
        classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = classifier.predict(X_test)
        y_pred_proba = None
        if hasattr(classifier, "predict_proba"):
            y_pred_proba = classifier.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=range(len(self.class_names))
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC (if binary or multiclass with probabilities)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                if len(self.class_names) == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC for {name}: {e}")
        
        # Save classifier
        joblib.dump(classifier, self.output_dir / f"{name.lower().replace(' ', '_')}_model.pkl")
        
        results = {
            'name': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
        }
        
        return results
    
    def cross_validate_classifier(self, classifier, name: str, cv_folds: int = 5) -> Dict:
        """Perform cross-validation evaluation."""
        logger.info(f"Cross-validating {name} with {cv_folds} folds...")
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(classifier, self.X, self.y, cv=skf, scoring='accuracy')
        cv_f1 = cross_val_score(classifier, self.X, self.y, cv=skf, scoring='f1_weighted')
        cv_precision = cross_val_score(classifier, self.X, self.y, cv=skf, scoring='precision_weighted')
        cv_recall = cross_val_score(classifier, self.X, self.y, cv=skf, scoring='recall_weighted')
        
        results = {
            'name': name,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'cv_precision_mean': cv_precision.mean(),
            'cv_precision_std': cv_precision.std(),
            'cv_recall_mean': cv_recall.mean(),
            'cv_recall_std': cv_recall.std(),
            'cv_scores': cv_scores.tolist(),
            'cv_f1_scores': cv_f1.tolist(),
            'cv_precision_scores': cv_precision.tolist(),
            'cv_recall_scores': cv_recall.tolist()
        }
        
        return results
    
    def run_evaluation(self, test_size: float = 0.2, cv_folds: int = 5) -> Dict:
        """Run complete linear probe evaluation."""
        logger.info("Starting linear probe evaluation...")
        
        # Split data
        data_split = self.train_test_split(test_size=test_size)
        
        # Define classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Linear SVM': SVC(kernel='linear', probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'RBF SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        # Train/test evaluation results
        train_test_results = []
        for name, classifier in classifiers.items():
            results = self.evaluate_classifier(
                classifier, 
                data_split['X_train'], data_split['X_test'],
                data_split['y_train'], data_split['y_test'],
                name
            )
            train_test_results.append(results)
        
        # Cross-validation results
        cv_results = []
        for name, classifier in classifiers.items():
            cv_result = self.cross_validate_classifier(classifier, name, cv_folds)
            cv_results.append(cv_result)
        
        # Combine results
        all_results = {
            'embeddings_path': self.embeddings_path,
            'dataset_info': {
                'num_samples': len(self.X),
                'num_features': self.X.shape[1],
                'num_classes': len(self.class_names),
                'class_names': self.class_names,
                'class_distribution': dict(pd.Series(self.y).value_counts())
            },
            'train_test_results': train_test_results,
            'cv_results': cv_results,
            'config': self.config
        }
        
        # Save results
        import json
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        
        # Generate plots and summary
        self._generate_plots(all_results)
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _generate_plots(self, results: Dict):
        """Generate evaluation plots."""
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Accuracy comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Train/test accuracy
        names = [r['name'] for r in results['train_test_results']]
        accuracies = [r['accuracy'] for r in results['train_test_results']]
        
        bars1 = ax1.bar(names, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Test Set Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Cross-validation accuracy with error bars
        cv_names = [r['name'] for r in results['cv_results']]
        cv_means = [r['cv_accuracy_mean'] for r in results['cv_results']]
        cv_stds = [r['cv_accuracy_std'] for r in results['cv_results']]
        
        bars2 = ax2.bar(cv_names, cv_means, yerr=cv_stds, capsize=5, 
                       color='lightcoral', alpha=0.7)
        ax2.set_title('Cross-Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars2, cv_means, cv_stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrices
        n_classifiers = len(results['train_test_results'])
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, result in enumerate(results['train_test_results']):
            cm = np.array(result['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], 
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       cmap='Blues')
            axes[i].set_title(f"{result['name']} - Confusion Matrix", fontweight='bold')
            axes[i].set_xlabel('Predicted', fontsize=10)
            axes[i].set_ylabel('True', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Metrics comparison radar chart (if we have multiple metrics)
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        classifier_names = [r['name'] for r in results['train_test_results']]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(classifier_names)))
        
        for i, result in enumerate(results['train_test_results']):
            values = [result[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=result['name'], color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Comparison', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated evaluation plots")
    
    def _generate_summary_report(self, results: Dict):
        """Generate a summary report."""
        report = []
        report.append("# Linear Probe Evaluation Report")
        report.append("")
        report.append(f"**Embeddings File:** {results['embeddings_path']}")
        report.append(f"**Dataset:** {results['dataset_info']['num_samples']} samples, "
                     f"{results['dataset_info']['num_features']} features, "
                     f"{results['dataset_info']['num_classes']} classes")
        report.append("")
        
        # Class distribution
        report.append("## Class Distribution")
        for class_name, count in zip(results['dataset_info']['class_names'], 
                                   results['dataset_info']['class_distribution'].values()):
            report.append(f"- {class_name}: {count}")
        report.append("")
        
        # Train/Test Results
        report.append("## Train/Test Results")
        report.append("| Classifier | Accuracy | Precision | Recall | F1-Score | ROC AUC |")
        report.append("|------------|----------|-----------|---------|----------|---------|")
        
        for result in results['train_test_results']:
            roc_auc_str = f"{result['roc_auc']:.3f}" if result['roc_auc'] is not None else "N/A"
            report.append(f"| {result['name']} | {result['accuracy']:.3f} | "
                         f"{result['precision']:.3f} | {result['recall']:.3f} | "
                         f"{result['f1']:.3f} | {roc_auc_str} |")
        report.append("")
        
        # Cross-Validation Results
        report.append("## Cross-Validation Results")
        report.append("| Classifier | Accuracy (μ±σ) | Precision (μ±σ) | Recall (μ±σ) | F1-Score (μ±σ) |")
        report.append("|------------|----------------|-----------------|--------------|----------------|")
        
        for result in results['cv_results']:
            report.append(f"| {result['name']} | "
                         f"{result['cv_accuracy_mean']:.3f}±{result['cv_accuracy_std']:.3f} | "
                         f"{result['cv_precision_mean']:.3f}±{result['cv_precision_std']:.3f} | "
                         f"{result['cv_recall_mean']:.3f}±{result['cv_recall_std']:.3f} | "
                         f"{result['cv_f1_mean']:.3f}±{result['cv_f1_std']:.3f} |")
        report.append("")
        
        # Best performer
        best_accuracy = max(results['train_test_results'], key=lambda x: x['accuracy'])
        best_cv = max(results['cv_results'], key=lambda x: x['cv_accuracy_mean'])
        
        report.append("## Summary")
        report.append(f"- **Best Test Accuracy:** {best_accuracy['name']} ({best_accuracy['accuracy']:.3f})")
        report.append(f"- **Best CV Accuracy:** {best_cv['name']} ({best_cv['cv_accuracy_mean']:.3f}±{best_cv['cv_accuracy_std']:.3f})")
        report.append("")
        report.append("See generated plots for detailed visualizations.")
        
        # Save report
        report_file = self.output_dir / "evaluation_summary.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Generated summary report: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate linear probe baseline on extracted embeddings')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings HDF5 file')
    parser.add_argument('--output-dir', type=str, default='results/linear_probe',
                       help='Output directory for results')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0.0-1.0)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    logger.info("Initializing linear probe evaluator...")
    evaluator = LinearProbeEvaluator(args.embeddings, args.output_dir)
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.run_evaluation(test_size=args.test_size, cv_folds=args.cv_folds)
    
    # Print summary
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    
    # Print best results
    best_test = max(results['train_test_results'], key=lambda x: x['accuracy'])
    best_cv = max(results['cv_results'], key=lambda x: x['cv_accuracy_mean'])
    
    print(f"\n🏆 Best Test Performance: {best_test['name']} - {best_test['accuracy']:.3f} accuracy")
    print(f"🏆 Best CV Performance: {best_cv['name']} - {best_cv['cv_accuracy_mean']:.3f}±{best_cv['cv_accuracy_std']:.3f} accuracy")


if __name__ == "__main__":
    main()
