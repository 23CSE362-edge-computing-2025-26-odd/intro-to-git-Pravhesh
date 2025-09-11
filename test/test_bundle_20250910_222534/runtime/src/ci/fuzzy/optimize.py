"""
Particle Swarm Optimization (PSO) for optimizing fuzzy system parameters.

This module provides PSO-based optimization of fuzzy membership function parameters
and rule weights for improved ECG classification performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from .engine import FuzzyDecisionEngine, FuzzyConfig
from .membership import FuzzyVariable, MembershipFunctionConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of PSO optimization."""
    
    # Best solution
    best_parameters: Dict[str, Any]
    best_fitness: float
    best_config: FuzzyConfig
    
    # Optimization history
    fitness_history: List[float]
    parameter_history: List[Dict[str, Any]]
    
    # Validation results
    validation_scores: Dict[str, float]
    fold_scores: List[float]
    
    # Metadata
    num_iterations: int
    num_particles: int
    convergence_iteration: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'best_parameters': self.best_parameters,
            'best_fitness': self.best_fitness,
            'best_config': self.best_config.to_dict(),
            'fitness_history': self.fitness_history,
            'validation_scores': self.validation_scores,
            'fold_scores': self.fold_scores,
            'num_iterations': self.num_iterations,
            'num_particles': self.num_particles,
            'convergence_iteration': self.convergence_iteration
        }
    
    def save(self, output_path: Union[str, Path]):
        """Save optimization result to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved optimization result to {output_path}")


@dataclass
class FuzzyPSOConfig:
    """Configuration for fuzzy PSO optimization."""
    
    # PSO parameters
    num_particles: int = 30
    max_iterations: int = 100
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.4
    social_coeff: float = 1.4
    
    # Convergence criteria
    tolerance: float = 1e-6
    patience: int = 10  # Iterations without improvement
    
    # Parameter bounds
    membership_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'triangular_a': (0.0, 1.0),
        'triangular_b': (0.0, 1.0), 
        'triangular_c': (0.0, 1.0),
        'trapezoidal_a': (0.0, 1.0),
        'trapezoidal_b': (0.0, 1.0),
        'trapezoidal_c': (0.0, 1.0),
        'trapezoidal_d': (0.0, 1.0),
        'gaussian_center': (0.0, 1.0),
        'gaussian_sigma': (0.01, 0.5)
    })
    
    rule_weight_bounds: Tuple[float, float] = (0.1, 1.0)
    
    # Optimization targets
    optimize_membership_functions: bool = True
    optimize_rule_weights: bool = True
    optimize_defuzzification: bool = False
    
    # Cross-validation
    cv_folds: int = 5
    validation_metric: str = 'f1_weighted'  # 'accuracy', 'f1_weighted', 'f1_macro'
    
    # Objective function weights
    performance_weight: float = 0.8
    complexity_weight: float = 0.2  # Penalize overly complex solutions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'num_particles': self.num_particles,
            'max_iterations': self.max_iterations,
            'inertia_weight': self.inertia_weight,
            'cognitive_coeff': self.cognitive_coeff,
            'social_coeff': self.social_coeff,
            'tolerance': self.tolerance,
            'patience': self.patience,
            'membership_bounds': self.membership_bounds,
            'rule_weight_bounds': self.rule_weight_bounds,
            'optimize_membership_functions': self.optimize_membership_functions,
            'optimize_rule_weights': self.optimize_rule_weights,
            'optimize_defuzzification': self.optimize_defuzzification,
            'cv_folds': self.cv_folds,
            'validation_metric': self.validation_metric,
            'performance_weight': self.performance_weight,
            'complexity_weight': self.complexity_weight
        }


class Particle:
    """A particle in the PSO algorithm."""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        self.parameter_bounds = parameter_bounds
        self.num_dimensions = len(parameter_bounds)
        
        # Initialize position and velocity randomly
        self.position = self._random_position()
        self.velocity = self._random_velocity()
        
        # Track personal best
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')
        
        # Current fitness
        self.fitness = float('-inf')
    
    def _random_position(self) -> np.ndarray:
        """Generate random initial position within bounds."""
        position = np.zeros(self.num_dimensions)
        for i, (param_name, (low, high)) in enumerate(self.parameter_bounds.items()):
            position[i] = np.random.uniform(low, high)
        return position
    
    def _random_velocity(self) -> np.ndarray:
        """Generate random initial velocity."""
        velocity = np.zeros(self.num_dimensions)
        for i, (param_name, (low, high)) in enumerate(self.parameter_bounds.items()):
            velocity[i] = np.random.uniform(-(high-low)*0.1, (high-low)*0.1)
        return velocity
    
    def update_velocity(self, global_best_position: np.ndarray, 
                       inertia: float, cognitive: float, social: float):
        """Update particle velocity."""
        r1 = np.random.random(self.num_dimensions)
        r2 = np.random.random(self.num_dimensions)
        
        # PSO velocity update equation
        self.velocity = (inertia * self.velocity + 
                        cognitive * r1 * (self.best_position - self.position) +
                        social * r2 * (global_best_position - self.position))
    
    def update_position(self):
        """Update particle position and apply bounds."""
        self.position += self.velocity
        
        # Apply bounds
        for i, (param_name, (low, high)) in enumerate(self.parameter_bounds.items()):
            self.position[i] = np.clip(self.position[i], low, high)
    
    def update_best(self):
        """Update personal best if current fitness is better."""
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()
            return True
        return False
    
    def get_parameters(self) -> Dict[str, float]:
        """Convert position vector to parameter dictionary."""
        parameters = {}
        for i, param_name in enumerate(self.parameter_bounds.keys()):
            parameters[param_name] = self.position[i]
        return parameters


class PSOOptimizer:
    """Particle Swarm Optimization for fuzzy system parameters."""
    
    def __init__(self, config: FuzzyPSOConfig):
        self.config = config
        self.swarm: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness: float = float('-inf')
        
        # Optimization history
        self.fitness_history: List[float] = []
        self.parameter_history: List[Dict[str, Any]] = []
        
    def _create_parameter_bounds(self, fuzzy_config: FuzzyConfig) -> Dict[str, Tuple[float, float]]:
        """Create parameter bounds for optimization."""
        bounds = {}
        
        if self.config.optimize_membership_functions:
            # Add membership function parameters
            # This is a simplified approach - in practice you'd extract actual MF parameters
            bounds.update({
                'heart_rate_normal_a': (50, 70),
                'heart_rate_normal_b': (60, 80),
                'heart_rate_normal_c': (80, 120),
                'heart_rate_tachy_a': (100, 120),
                'heart_rate_tachy_b': (120, 150),
                'qrs_normal_a': (80, 90),
                'qrs_normal_b': (90, 110),
                'qrs_normal_c': (110, 120),
                'st_normal_center': (-0.5, 0.5),
                'st_normal_sigma': (0.1, 0.8)
            })
        
        if self.config.optimize_rule_weights:
            # Add rule weight parameters (assuming 5 rules from the rule base)
            for i in range(5):  # Number of rules in the default rule base
                bounds[f'rule_{i}_weight'] = self.config.rule_weight_bounds
        
        return bounds
    
    def _parameters_to_config(self, parameters: Dict[str, float], 
                             base_config: FuzzyConfig) -> FuzzyConfig:
        """Convert parameter dictionary to fuzzy configuration."""
        # Create a copy of the base config
        new_config = copy.deepcopy(base_config)
        
        # Apply optimized parameters
        # This is simplified - in practice you'd update the actual MF parameters
        # For now, we'll just modify some basic thresholds
        
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
    
    def _evaluate_fitness(self, parameters: Dict[str, float], 
                         fuzzy_config: FuzzyConfig, 
                         X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate fitness of a parameter set using cross-validation."""
        try:
            # Convert parameters to config
            config = self._parameters_to_config(parameters, fuzzy_config)
            
            # Create fuzzy engine
            engine = FuzzyDecisionEngine(config)
            
            # Perform cross-validation
            kfold = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kfold.split(X, y):
                X_val, y_val = X[val_idx], y[val_idx]
                
                # Generate predictions
                predictions = []
                for i in range(len(X_val)):
                    # Simulate diagnosis (this would use actual embeddings in practice)
                    result = engine.diagnose_from_embeddings(X_val[i])
                    
                    # Convert risk level to class prediction
                    if result.risk_level == 'low':
                        pred = 0
                    elif result.risk_level == 'moderate':
                        pred = 1
                    else:  # high
                        pred = 2
                    
                    predictions.append(pred)
                
                # Calculate score
                if self.config.validation_metric == 'accuracy':
                    score = accuracy_score(y_val, predictions)
                else:  # f1_weighted or f1_macro
                    average = 'weighted' if self.config.validation_metric == 'f1_weighted' else 'macro'
                    score = f1_score(y_val, predictions, average=average, zero_division=0)
                
                scores.append(score)
            
            # Calculate mean performance
            performance = np.mean(scores)
            
            # Add complexity penalty (simpler models preferred)
            complexity_penalty = 0
            if self.config.complexity_weight > 0:
                # Penalize extreme parameter values
                param_complexity = sum(abs(p - 0.5) for p in parameters.values()) / len(parameters)
                complexity_penalty = self.config.complexity_weight * param_complexity
            
            # Combined fitness
            fitness = self.config.performance_weight * performance - complexity_penalty
            
            return fitness
            
        except Exception as e:
            logger.warning(f"Error evaluating fitness: {e}")
            return float('-inf')
    
    def optimize(self, fuzzy_config: FuzzyConfig, 
                X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """
        Optimize fuzzy parameters using PSO.
        
        Args:
            fuzzy_config: Base fuzzy configuration to optimize
            X: Input embeddings or features
            y: Target labels
            
        Returns:
            OptimizationResult with best parameters and performance
        """
        logger.info(f"Starting PSO optimization with {self.config.num_particles} particles, "
                   f"{self.config.max_iterations} iterations")
        
        # Create parameter bounds
        parameter_bounds = self._create_parameter_bounds(fuzzy_config)
        logger.info(f"Optimizing {len(parameter_bounds)} parameters")
        
        # Initialize swarm
        self.swarm = [Particle(parameter_bounds) for _ in range(self.config.num_particles)]
        
        # Optimization loop
        convergence_counter = 0
        convergence_iteration = None
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"PSO Iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Evaluate fitness for all particles
            for particle in self.swarm:
                parameters = particle.get_parameters()
                fitness = self._evaluate_fitness(parameters, fuzzy_config, X, y)
                particle.fitness = fitness
                
                # Update personal best
                particle.update_best()
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                    convergence_counter = 0  # Reset convergence counter
                else:
                    convergence_counter += 1
            
            # Store history
            self.fitness_history.append(self.global_best_fitness)
            if self.global_best_position is not None:
                best_particle = min(self.swarm, key=lambda p: 
                                  np.linalg.norm(p.position - self.global_best_position))
                self.parameter_history.append(best_particle.get_parameters())
            
            logger.info(f"Best fitness: {self.global_best_fitness:.4f}")
            
            # Check convergence
            if convergence_counter >= self.config.patience:
                convergence_iteration = iteration
                logger.info(f"Converged at iteration {iteration}")
                break
            
            # Update particles
            for particle in self.swarm:
                particle.update_velocity(
                    self.global_best_position,
                    self.config.inertia_weight,
                    self.config.cognitive_coeff,
                    self.config.social_coeff
                )
                particle.update_position()
        
        # Get best parameters and create optimized config
        best_particle = min(self.swarm, key=lambda p: 
                          np.linalg.norm(p.position - self.global_best_position))
        best_parameters = best_particle.get_parameters()
        best_config = self._parameters_to_config(best_parameters, fuzzy_config)
        
        # Validate on full dataset
        validation_scores = self._final_validation(best_config, X, y)
        
        result = OptimizationResult(
            best_parameters=best_parameters,
            best_fitness=self.global_best_fitness,
            best_config=best_config,
            fitness_history=self.fitness_history,
            parameter_history=self.parameter_history,
            validation_scores=validation_scores,
            fold_scores=[self.global_best_fitness],  # Simplified
            num_iterations=len(self.fitness_history),
            num_particles=self.config.num_particles,
            convergence_iteration=convergence_iteration
        )
        
        logger.info(f"PSO optimization completed. Best fitness: {self.global_best_fitness:.4f}")
        return result
    
    def _final_validation(self, config: FuzzyConfig, 
                         X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform final validation with best configuration."""
        try:
            engine = FuzzyDecisionEngine(config)
            
            # Generate predictions
            predictions = []
            confidences = []
            
            for i in range(len(X)):
                result = engine.diagnose_from_embeddings(X[i])
                
                # Convert to prediction
                if result.risk_level == 'low':
                    pred = 0
                elif result.risk_level == 'moderate':
                    pred = 1
                else:
                    pred = 2
                
                predictions.append(pred)
                confidences.append(result.confidence)
            
            # Calculate metrics
            accuracy = accuracy_score(y, predictions)
            f1_weighted = f1_score(y, predictions, average='weighted', zero_division=0)
            f1_macro = f1_score(y, predictions, average='macro', zero_division=0)
            
            return {
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'mean_confidence': np.mean(confidences)
            }
            
        except Exception as e:
            logger.error(f"Error in final validation: {e}")
            return {
                'accuracy': 0.0,
                'f1_weighted': 0.0,
                'f1_macro': 0.0,
                'mean_confidence': 0.0
            }


def optimize_fuzzy_parameters(fuzzy_config: FuzzyConfig,
                            embeddings: np.ndarray,
                            labels: np.ndarray,
                            pso_config: Optional[FuzzyPSOConfig] = None) -> OptimizationResult:
    """
    Convenience function to optimize fuzzy parameters.
    
    Args:
        fuzzy_config: Base fuzzy configuration
        embeddings: Input embeddings for training
        labels: Target labels
        pso_config: PSO configuration (uses default if None)
        
    Returns:
        OptimizationResult with best parameters
    """
    if pso_config is None:
        pso_config = FuzzyPSOConfig()
    
    optimizer = PSOOptimizer(pso_config)
    return optimizer.optimize(fuzzy_config, embeddings, labels)
