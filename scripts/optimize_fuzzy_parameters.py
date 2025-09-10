#!/usr/bin/env python3
"""
Subject-wise k-fold optimization of fuzzy parameters using PSO.

This script loads extracted embeddings, performs subject-wise cross-validation
optimization of fuzzy logic parameters, and saves the best parameters.
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
from sklearn.model_selection import StratifiedGroupKFold
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ci.fuzzy.engine import FuzzyConfig, FuzzyDecisionEngine, create_default_config
from ci.fuzzy.optimize import PSOOptimizer, FuzzyPSOConfig, OptimizationResult
from scripts.extract_embeddings import load_embeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SubjectWiseOptimizer:
    """Subject-wise k-fold optimization of fuzzy parameters."""
    
    def __init__(self, output_dir: str = "results/fuzzy_optimization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, embeddings_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load embeddings and prepare for subject-wise optimization.
        
        Returns:
            embeddings: Feature embeddings
            labels: Target labels (encoded)
            subjects: Subject/patient identifiers
        """
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings, metadata_df, config = load_embeddings(embeddings_path)
        
        # Encode labels
        unique_labels = sorted(metadata_df['label'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_to_idx[label] for label in metadata_df['label']])
        
        # Get subject identifiers
        if 'patient_id' in metadata_df.columns:
            subjects = metadata_df['patient_id'].values
        elif 'file_path' in metadata_df.columns:
            # Use file path as proxy for subject (assuming one file per subject)
            subjects = metadata_df['file_path'].apply(lambda x: Path(x).stem).values
        else:
            # Create artificial subjects based on indices
            logger.warning("No subject information found, creating artificial subject groups")
            subjects = np.arange(len(embeddings)) // 10  # Group every 10 samples
        
        logger.info(f"Loaded {len(embeddings)} samples with {embeddings.shape[1]} features")
        logger.info(f"Found {len(unique_labels)} classes: {unique_labels}")
        logger.info(f"Found {len(np.unique(subjects))} unique subjects")
        
        return embeddings, labels, subjects
    
    def optimize_subject_wise(self, embeddings: np.ndarray, labels: np.ndarray, 
                            subjects: np.ndarray, fuzzy_config: FuzzyConfig,
                            pso_config: FuzzyPSOConfig, n_splits: int = 5) -> Dict:
        """
        Perform subject-wise k-fold optimization.
        
        Args:
            embeddings: Input embeddings
            labels: Target labels
            subjects: Subject identifiers for grouping
            fuzzy_config: Base fuzzy configuration
            pso_config: PSO configuration
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with optimization results for each fold and best overall
        """
        logger.info(f"Starting subject-wise {n_splits}-fold optimization")
        
        # Use StratifiedGroupKFold to ensure subjects don't appear in both train/test
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_results = []
        all_best_params = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(embeddings, labels, subjects)):
            logger.info(f"\n=== FOLD {fold_idx + 1}/{n_splits} ===")
            
            X_train, X_val = embeddings[train_idx], embeddings[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            subjects_train = subjects[train_idx]
            subjects_val = subjects[val_idx]
            
            logger.info(f"Train: {len(X_train)} samples, {len(np.unique(subjects_train))} subjects")
            logger.info(f"Val: {len(X_val)} samples, {len(np.unique(subjects_val))} subjects")
            
            # Optimize on training set
            optimizer = PSOOptimizer(pso_config)
            
            try:
                optimization_result = optimizer.optimize(fuzzy_config, X_train, y_train)
                
                # Evaluate on validation set
                val_scores = self._evaluate_on_validation(
                    optimization_result.best_config, X_val, y_val
                )
                
                fold_result = {
                    'fold': fold_idx,
                    'train_subjects': len(np.unique(subjects_train)),
                    'val_subjects': len(np.unique(subjects_val)),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'best_parameters': optimization_result.best_parameters,
                    'train_fitness': optimization_result.best_fitness,
                    'validation_scores': val_scores,
                    'convergence_iteration': optimization_result.convergence_iteration,
                    'fitness_history': optimization_result.fitness_history
                }
                
                fold_results.append(fold_result)
                all_best_params.append(optimization_result.best_parameters)
                
                # Save fold result
                fold_path = self.output_dir / f"fold_{fold_idx}_results.json"
                with open(fold_path, 'w') as f:
                    json.dump(fold_result, f, indent=2)
                
                logger.info(f"Fold {fold_idx} - Train fitness: {optimization_result.best_fitness:.4f}")
                logger.info(f"Fold {fold_idx} - Val accuracy: {val_scores['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold_idx}: {e}")
                # Add empty result to maintain fold indexing
                fold_results.append({
                    'fold': fold_idx,
                    'error': str(e),
                    'train_fitness': 0.0,
                    'validation_scores': {'accuracy': 0.0, 'f1_weighted': 0.0}
                })
        
        # Aggregate results and find best parameters
        results_summary = self._aggregate_fold_results(fold_results, all_best_params, fuzzy_config)
        
        return results_summary
    
    def _evaluate_on_validation(self, config: FuzzyConfig, 
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate optimized config on validation set."""
        try:
            engine = FuzzyDecisionEngine(config)
            
            predictions = []
            confidences = []
            
            for i in range(len(X_val)):
                result = engine.diagnose_from_embeddings(X_val[i])
                
                # Convert risk level to prediction
                if result.risk_level == 'low':
                    pred = 0
                elif result.risk_level == 'moderate':
                    pred = 1
                else:  # high
                    pred = 2
                
                predictions.append(pred)
                confidences.append(result.confidence)
            
            from sklearn.metrics import accuracy_score, f1_score
            
            accuracy = accuracy_score(y_val, predictions)
            f1_weighted = f1_score(y_val, predictions, average='weighted', zero_division=0)
            f1_macro = f1_score(y_val, predictions, average='macro', zero_division=0)
            
            return {
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'mean_confidence': np.mean(confidences)
            }
            
        except Exception as e:
            logger.error(f"Error in validation evaluation: {e}")
            return {
                'accuracy': 0.0,
                'f1_weighted': 0.0,
                'f1_macro': 0.0,
                'mean_confidence': 0.0
            }
    
    def _aggregate_fold_results(self, fold_results: List[Dict], 
                              all_best_params: List[Dict],
                              base_config: FuzzyConfig) -> Dict:
        """Aggregate results from all folds and determine best parameters."""
        logger.info("Aggregating fold results...")
        
        # Filter successful folds
        successful_folds = [r for r in fold_results if 'error' not in r]
        
        if not successful_folds:
            logger.error("No successful folds!")
            return {'error': 'All folds failed'}
        
        # Calculate statistics
        train_fitnesses = [r['train_fitness'] for r in successful_folds]
        val_accuracies = [r['validation_scores']['accuracy'] for r in successful_folds]
        val_f1_scores = [r['validation_scores']['f1_weighted'] for r in successful_folds]
        
        # Find best fold based on validation accuracy
        best_fold_idx = np.argmax(val_accuracies)
        best_fold = successful_folds[best_fold_idx]
        
        # Ensemble parameters (average across folds)
        if len(all_best_params) > 1:
            ensemble_params = self._ensemble_parameters(all_best_params)
        else:
            ensemble_params = all_best_params[0] if all_best_params else {}
        
        # Create optimized configs
        best_config = self._parameters_to_config(best_fold['best_parameters'], base_config)
        ensemble_config = self._parameters_to_config(ensemble_params, base_config)
        
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'num_folds': len(fold_results),
            'successful_folds': len(successful_folds),
            'failed_folds': len(fold_results) - len(successful_folds),
            
            # Aggregate statistics
            'mean_train_fitness': np.mean(train_fitnesses),
            'std_train_fitness': np.std(train_fitnesses),
            'mean_val_accuracy': np.mean(val_accuracies),
            'std_val_accuracy': np.std(val_accuracies),
            'mean_val_f1': np.mean(val_f1_scores),
            'std_val_f1': np.std(val_f1_scores),
            
            # Best fold results
            'best_fold_idx': best_fold_idx,
            'best_fold_val_accuracy': val_accuracies[best_fold_idx],
            'best_fold_parameters': best_fold['best_parameters'],
            'best_fold_config': best_config.to_dict(),
            
            # Ensemble results
            'ensemble_parameters': ensemble_params,
            'ensemble_config': ensemble_config.to_dict(),
            
            # All fold results
            'fold_results': fold_results
        }
        
        return results_summary
    
    def _ensemble_parameters(self, param_list: List[Dict]) -> Dict:
        """Create ensemble parameters by averaging."""
        if not param_list:
            return {}
        
        # Get all parameter names
        all_keys = set()
        for params in param_list:
            all_keys.update(params.keys())
        
        # Average parameters
        ensemble = {}
        for key in all_keys:
            values = [params.get(key, 0.0) for params in param_list if key in params]
            if values:
                ensemble[key] = np.mean(values)
        
        return ensemble
    
    def _parameters_to_config(self, parameters: Dict[str, float], 
                            base_config: FuzzyConfig) -> FuzzyConfig:
        """Convert parameters to FuzzyConfig (simplified version)."""
        import copy
        new_config = copy.deepcopy(base_config)
        
        # Apply parameter updates (simplified)
        if 'heart_rate_normal_a' in parameters:
            # Update risk thresholds based on optimized parameters
            low_thresh = parameters.get('heart_rate_normal_a', 50) / 200.0
            mod_thresh = parameters.get('heart_rate_normal_b', 100) / 200.0 
            high_thresh = parameters.get('heart_rate_tachy_a', 120) / 200.0
            
            new_config.risk_thresholds = {
                'low': (0.0, min(0.4, low_thresh + 0.1)),
                'moderate': (max(0.3, low_thresh), min(0.7, mod_thresh + 0.1)),
                'high': (max(0.6, mod_thresh), 1.0)
            }
        
        return new_config
    
    def save_results(self, results: Dict, output_name: str = "optimization_results"):
        """Save final optimization results."""
        # Save full results
        results_path = self.output_dir / f"{output_name}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save best configuration
        if 'best_fold_config' in results:
            config_path = self.output_dir / f"{output_name}_best_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump({'fuzzy': results['best_fold_config']}, f, 
                         default_flow_style=False, indent=2)
        
        # Save ensemble configuration
        if 'ensemble_config' in results:
            ensemble_config_path = self.output_dir / f"{output_name}_ensemble_config.yaml"
            with open(ensemble_config_path, 'w') as f:
                yaml.dump({'fuzzy': results['ensemble_config']}, f, 
                         default_flow_style=False, indent=2)
        
        # Create summary report
        self._create_summary_report(results, output_name)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _create_summary_report(self, results: Dict, output_name: str):
        """Create a markdown summary report."""
        report = []
        report.append("# Fuzzy Parameter Optimization Results")
        report.append("")
        report.append(f"**Timestamp:** {results.get('timestamp', 'N/A')}")
        report.append(f"**Folds:** {results.get('num_folds', 'N/A')} "
                     f"({results.get('successful_folds', 0)} successful)")
        report.append("")
        
        # Overall performance
        report.append("## Overall Performance")
        report.append(f"- **Mean Validation Accuracy:** "
                     f"{results.get('mean_val_accuracy', 0):.3f} ± "
                     f"{results.get('std_val_accuracy', 0):.3f}")
        report.append(f"- **Mean Validation F1:** "
                     f"{results.get('mean_val_f1', 0):.3f} ± "
                     f"{results.get('std_val_f1', 0):.3f}")
        report.append(f"- **Mean Training Fitness:** "
                     f"{results.get('mean_train_fitness', 0):.3f} ± "
                     f"{results.get('std_train_fitness', 0):.3f}")
        report.append("")
        
        # Best fold
        report.append("## Best Fold Performance")
        best_idx = results.get('best_fold_idx', 0)
        report.append(f"- **Best Fold:** {best_idx + 1}")
        report.append(f"- **Validation Accuracy:** "
                     f"{results.get('best_fold_val_accuracy', 0):.3f}")
        report.append("")
        
        # Fold-by-fold results
        report.append("## Fold-by-Fold Results")
        report.append("| Fold | Train Fitness | Val Accuracy | Val F1 | Status |")
        report.append("|------|---------------|--------------|--------|--------|")
        
        for fold_result in results.get('fold_results', []):
            fold_num = fold_result.get('fold', 0) + 1
            if 'error' in fold_result:
                report.append(f"| {fold_num} | - | - | - | Failed |")
            else:
                fitness = fold_result.get('train_fitness', 0)
                accuracy = fold_result.get('validation_scores', {}).get('accuracy', 0)
                f1 = fold_result.get('validation_scores', {}).get('f1_weighted', 0)
                report.append(f"| {fold_num} | {fitness:.3f} | {accuracy:.3f} | {f1:.3f} | Success |")
        
        report.append("")
        report.append("## Files Generated")
        report.append(f"- `{output_name}.json` - Complete results")
        report.append(f"- `{output_name}_best_config.yaml` - Best fold configuration")
        report.append(f"- `{output_name}_ensemble_config.yaml` - Ensemble configuration")
        report.append("- `fold_*_results.json` - Individual fold results")
        
        # Save report
        report_path = self.output_dir / f"{output_name}_summary.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description='Subject-wise k-fold optimization of fuzzy parameters')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings HDF5 file')
    parser.add_argument('--output-dir', type=str, default='results/fuzzy_optimization',
                       help='Output directory for results')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--fuzzy-config', type=str, default=None,
                       help='Path to base fuzzy config YAML (optional)')
    
    # PSO parameters
    parser.add_argument('--particles', type=int, default=20,
                       help='Number of PSO particles')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Maximum PSO iterations')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    logger.info("Initializing subject-wise optimizer...")
    optimizer = SubjectWiseOptimizer(args.output_dir)
    
    # Load data
    embeddings, labels, subjects = optimizer.load_data(args.embeddings)
    
    # Load or create fuzzy config
    if args.fuzzy_config:
        fuzzy_config = FuzzyConfig.from_yaml(args.fuzzy_config)
        logger.info(f"Loaded fuzzy config from {args.fuzzy_config}")
    else:
        fuzzy_config = create_default_config()
        logger.info("Using default fuzzy config")
    
    # Create PSO config
    pso_config = FuzzyPSOConfig(
        num_particles=args.particles,
        max_iterations=args.iterations,
        patience=args.patience,
        cv_folds=3  # Inner CV for PSO evaluation
    )
    
    # Run optimization
    logger.info("Starting subject-wise optimization...")
    results = optimizer.optimize_subject_wise(
        embeddings, labels, subjects, fuzzy_config, pso_config, args.n_folds
    )
    
    # Save results
    output_name = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    optimizer.save_results(results, output_name)
    
    # Print summary
    logger.info("Optimization completed!")
    if 'error' not in results:
        logger.info(f"Mean validation accuracy: {results['mean_val_accuracy']:.3f} ± {results['std_val_accuracy']:.3f}")
        logger.info(f"Best fold accuracy: {results['best_fold_val_accuracy']:.3f}")
        logger.info(f"Results saved to: {args.output_dir}")
    else:
        logger.error(f"Optimization failed: {results['error']}")


if __name__ == "__main__":
    main()
