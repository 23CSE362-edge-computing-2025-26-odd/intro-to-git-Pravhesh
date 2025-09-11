"""
Fuzzy Decision Engine for ECG Classification

This module provides the main decision engine that combines ResNet18 features
with fuzzy logic inference for cardiac risk assessment and diagnosis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import logging
import json

from .membership import FuzzyVariable, create_ecg_risk_variable, create_ecg_feature_variables
from .rules import FuzzyRuleBase, RuleEngine, create_ecg_rule_base

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisResult:
    """Result of fuzzy diagnosis with confidence and explanations."""
    
    # Main results
    risk_score: float  # Continuous risk score [0, 1]
    risk_level: str    # Categorical risk level (low, moderate, high)
    confidence: float  # Confidence in the diagnosis [0, 1]
    
    # Supporting information
    fired_rules: List[Dict[str, Any]]  # Rules that fired and their activations
    feature_memberships: Dict[str, Dict[str, float]]  # Feature membership degrees
    input_features: Dict[str, float]  # Input feature values
    
    # Metadata
    patient_id: Optional[str] = None
    timestamp: Optional[str] = None
    model_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'confidence': self.confidence,
            'fired_rules': self.fired_rules,
            'feature_memberships': self.feature_memberships,
            'input_features': self.input_features,
            'patient_id': self.patient_id,
            'timestamp': self.timestamp,
            'model_version': self.model_version
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class FuzzyConfig:
    """Configuration for fuzzy decision engine."""
    
    # Rule base configuration
    rule_base_name: str = "ecg_diagnosis"
    inference_method: str = "mamdani"  # 'mamdani' or 'sugeno'
    defuzzification_method: str = "centroid"  # 'centroid', 'maximum', 'mean_of_maxima'
    
    # Aggregation parameters
    rule_aggregation: str = "maximum"  # 'maximum', 'sum', 'probor'
    min_rule_activation: float = 0.01  # Minimum activation for a rule to fire
    
    # Feature processing
    feature_normalization: bool = True
    feature_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'heart_rate': (40, 200),
        'qrs_duration': (60, 200), 
        'st_elevation': (-2, 5)
    })
    
    # Confidence calculation
    confidence_method: str = "activation_weighted"  # 'activation_weighted', 'rule_count'
    min_confidence: float = 0.1
    
    # Output configuration
    risk_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'low': (0.0, 0.4),
        'moderate': (0.3, 0.7),
        'high': (0.6, 1.0)
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'rule_base_name': self.rule_base_name,
            'inference_method': self.inference_method,
            'defuzzification_method': self.defuzzification_method,
            'rule_aggregation': self.rule_aggregation,
            'min_rule_activation': self.min_rule_activation,
            'feature_normalization': self.feature_normalization,
            'feature_bounds': self.feature_bounds,
            'confidence_method': self.confidence_method,
            'min_confidence': self.min_confidence,
            'risk_thresholds': self.risk_thresholds
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FuzzyConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'FuzzyConfig':
        """Create config from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        fuzzy_config = config.get('fuzzy', {})
        return cls.from_dict(fuzzy_config)


class FeatureProcessor:
    """Processes ResNet features and clinical features for fuzzy inference."""
    
    def __init__(self, config: FuzzyConfig):
        self.config = config
        
    def extract_clinical_features(self, ecg_data: np.ndarray, 
                                sampling_rate: float) -> Dict[str, float]:
        """
        Extract clinical features from ECG data.
        
        This is a simplified implementation. In practice, you would use
        sophisticated ECG analysis algorithms.
        """
        # Simple feature extraction (placeholder implementation)
        features = {}
        
        # Estimate heart rate from R-R intervals
        # This is a very basic implementation
        if len(ecg_data) > 0:
            duration = len(ecg_data) / sampling_rate
            # Rough heart rate estimation (this should be much more sophisticated)
            features['heart_rate'] = 60.0 + 20 * np.sin(np.mean(ecg_data))
            
            # QRS duration estimation (placeholder)
            features['qrs_duration'] = 80.0 + 15 * np.abs(np.std(ecg_data))
            
            # ST elevation estimation (placeholder)
            features['st_elevation'] = np.mean(ecg_data) * 0.1
        else:
            # Default values if no data
            features['heart_rate'] = 72.0
            features['qrs_duration'] = 90.0
            features['st_elevation'] = 0.0
        
        return features
    
    def process_resnet_features(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Process ResNet embeddings to extract relevant clinical indicators.
        
        This maps the 512-d embedding space to clinical features that can
        be used by the fuzzy rule system.
        """
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Use PCA-like projections or learned mappings
        # For now, use simple statistics as feature extractors
        features = {}
        
        # Map embedding statistics to clinical features
        mean_emb = np.mean(embeddings, axis=1)[0]
        std_emb = np.std(embeddings, axis=1)[0]
        skew_emb = self._skewness(embeddings[0])
        
        # Map to heart rate (example mapping)
        features['heart_rate'] = np.clip(72 + 30 * mean_emb, 40, 200)
        
        # Map to QRS duration
        features['qrs_duration'] = np.clip(90 + 40 * std_emb, 60, 200)
        
        # Map to ST elevation
        features['st_elevation'] = np.clip(skew_emb * 2, -2, 5)
        
        return features
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features to expected ranges."""
        if not self.config.feature_normalization:
            return features
        
        normalized = {}
        for feature_name, value in features.items():
            if feature_name in self.config.feature_bounds:
                min_val, max_val = self.config.feature_bounds[feature_name]
                # Clip to bounds
                normalized_value = np.clip(value, min_val, max_val)
                normalized[feature_name] = normalized_value
            else:
                normalized[feature_name] = value
        
        return normalized
    
    def combine_features(self, resnet_features: Optional[Dict[str, float]] = None,
                        clinical_features: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Combine features from different sources with optional weighting."""
        combined = {}
        
        if resnet_features:
            for key, value in resnet_features.items():
                combined[f"resnet_{key}"] = value
        
        if clinical_features:
            for key, value in clinical_features.items():
                combined[f"clinical_{key}"] = value
        
        # If we have both, create weighted combinations
        if resnet_features and clinical_features:
            for key in resnet_features.keys():
                if key in clinical_features:
                    # Weighted average (can be made configurable)
                    resnet_weight = 0.7  # ResNet features have higher weight
                    clinical_weight = 0.3
                    combined[key] = (resnet_weight * resnet_features[key] + 
                                   clinical_weight * clinical_features[key])
        elif resnet_features:
            # Use ResNet features as primary
            combined.update(resnet_features)
        elif clinical_features:
            # Use clinical features as primary
            combined.update(clinical_features)
        
        return self.normalize_features(combined)


class FuzzyDecisionEngine:
    """Main fuzzy decision engine for ECG diagnosis."""
    
    def __init__(self, config: Optional[FuzzyConfig] = None):
        """Initialize the fuzzy decision engine."""
        self.config = config or FuzzyConfig()
        
        # Initialize components
        self.feature_processor = FeatureProcessor(self.config)
        self.risk_variable = create_ecg_risk_variable()
        self.feature_variables = create_ecg_feature_variables()
        self.rule_base = create_ecg_rule_base()
        self.rule_engine = RuleEngine(self.rule_base)
        
        logger.info(f"Initialized fuzzy decision engine with {len(self.rule_base.rules)} rules")
    
    def diagnose_from_embeddings(self, embeddings: np.ndarray, 
                                patient_id: Optional[str] = None,
                                timestamp: Optional[str] = None) -> DiagnosisResult:
        """
        Diagnose from ResNet embeddings.
        
        Args:
            embeddings: ResNet feature embeddings (512-d vector or batch)
            patient_id: Optional patient identifier
            timestamp: Optional timestamp
            
        Returns:
            DiagnosisResult with risk assessment
        """
        # Process embeddings to clinical features
        resnet_features = self.feature_processor.process_resnet_features(embeddings)
        combined_features = self.feature_processor.combine_features(
            resnet_features=resnet_features
        )
        
        return self._diagnose_from_features(combined_features, patient_id, timestamp)
    
    def diagnose_from_ecg(self, ecg_data: np.ndarray, sampling_rate: float,
                         embeddings: Optional[np.ndarray] = None,
                         patient_id: Optional[str] = None,
                         timestamp: Optional[str] = None) -> DiagnosisResult:
        """
        Diagnose from raw ECG data, optionally combined with embeddings.
        
        Args:
            ecg_data: Raw ECG signal
            sampling_rate: Sampling rate of ECG data
            embeddings: Optional ResNet embeddings
            patient_id: Optional patient identifier
            timestamp: Optional timestamp
            
        Returns:
            DiagnosisResult with risk assessment
        """
        # Extract clinical features
        clinical_features = self.feature_processor.extract_clinical_features(
            ecg_data, sampling_rate
        )
        
        # Process embeddings if provided
        resnet_features = None
        if embeddings is not None:
            resnet_features = self.feature_processor.process_resnet_features(embeddings)
        
        # Combine features
        combined_features = self.feature_processor.combine_features(
            resnet_features=resnet_features,
            clinical_features=clinical_features
        )
        
        return self._diagnose_from_features(combined_features, patient_id, timestamp)
    
    def _diagnose_from_features(self, features: Dict[str, float],
                               patient_id: Optional[str] = None,
                               timestamp: Optional[str] = None) -> DiagnosisResult:
        """Internal method to perform diagnosis from processed features."""
        
        # Perform fuzzy inference
        fuzzy_output = self.rule_engine.infer(features)
        
        # Get activated rules for explanation
        fired_rules = []
        active_rules = self.rule_base.get_active_rules(features, self.config.min_rule_activation)
        
        for rule, activation in active_rules:
            fired_rules.append({
                'name': rule.name,
                'description': rule.description,
                'activation': activation,
                'weight': rule.weight,
                'consequent': f"{rule.consequent.variable_name} IS {rule.consequent.fuzzy_set_name}"
            })
        
        # Calculate feature memberships for explanation
        feature_memberships = {}
        for var_name, variable in self.feature_variables.items():
            if var_name in features:
                memberships = variable.get_membership_degrees(features[var_name])
                feature_memberships[var_name] = memberships
        
        # Defuzzify to get crisp risk score
        risk_score = 0.5  # Default middle risk
        if 'risk' in fuzzy_output and fuzzy_output['risk']:
            risk_score = self.rule_engine.defuzzify(
                fuzzy_output['risk'], 
                self.risk_variable, 
                method=self.config.defuzzification_method
            )
        
        # Determine risk level
        risk_level = self._categorize_risk(risk_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(fired_rules, risk_score)
        
        return DiagnosisResult(
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            fired_rules=fired_rules,
            feature_memberships=feature_memberships,
            input_features=features,
            patient_id=patient_id,
            timestamp=timestamp,
            model_version="1.0"
        )
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize continuous risk score into discrete risk level."""
        for risk_level, (low_thresh, high_thresh) in self.config.risk_thresholds.items():
            if low_thresh <= risk_score <= high_thresh:
                # If overlapping ranges, choose the one where score is more central
                center = (low_thresh + high_thresh) / 2
                if not hasattr(self, '_best_risk_match'):
                    self._best_risk_match = (risk_level, abs(risk_score - center))
                else:
                    current_distance = abs(risk_score - center)
                    if current_distance < self._best_risk_match[1]:
                        self._best_risk_match = (risk_level, current_distance)
        
        if hasattr(self, '_best_risk_match'):
            result = self._best_risk_match[0]
            delattr(self, '_best_risk_match')
            return result
        else:
            # Fallback
            if risk_score < 0.4:
                return 'low'
            elif risk_score < 0.7:
                return 'moderate'
            else:
                return 'high'
    
    def _calculate_confidence(self, fired_rules: List[Dict], risk_score: float) -> float:
        """Calculate confidence in the diagnosis."""
        if not fired_rules:
            return self.config.min_confidence
        
        if self.config.confidence_method == "activation_weighted":
            # Weight by activation levels
            total_activation = sum(rule['activation'] for rule in fired_rules)
            max_possible = len(fired_rules)  # If all rules fired at 1.0
            confidence = total_activation / max_possible if max_possible > 0 else 0
            
        elif self.config.confidence_method == "rule_count":
            # Simple rule count method
            confidence = min(len(fired_rules) / 3.0, 1.0)  # Assume 3+ rules = high confidence
            
        else:
            confidence = 0.5
        
        return max(confidence, self.config.min_confidence)
    
    def batch_diagnose(self, embeddings_batch: np.ndarray,
                      patient_ids: Optional[List[str]] = None) -> List[DiagnosisResult]:
        """Diagnose a batch of embeddings."""
        if len(embeddings_batch.shape) == 1:
            embeddings_batch = embeddings_batch.reshape(1, -1)
        
        results = []
        for i in range(embeddings_batch.shape[0]):
            patient_id = patient_ids[i] if patient_ids else None
            result = self.diagnose_from_embeddings(
                embeddings_batch[i], 
                patient_id=patient_id
            )
            results.append(result)
        
        return results
    
    def save_config(self, config_path: Union[str, Path]):
        """Save current configuration to file."""
        config_dict = {'fuzzy': self.config.to_dict()}
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved fuzzy config to {config_path}")
    
    def load_config(self, config_path: Union[str, Path]):
        """Load configuration from file."""
        self.config = FuzzyConfig.from_yaml(config_path)
        
        # Reinitialize components with new config
        self.feature_processor = FeatureProcessor(self.config)
        
        logger.info(f"Loaded fuzzy config from {config_path}")
    
    def get_explanation(self, result: DiagnosisResult) -> str:
        """Generate human-readable explanation of diagnosis."""
        explanation = []
        
        explanation.append(f"Risk Assessment: {result.risk_level.upper()} ({result.risk_score:.2f})")
        explanation.append(f"Confidence: {result.confidence:.2f}")
        explanation.append("")
        
        explanation.append("Activated Rules:")
        for rule in result.fired_rules:
            explanation.append(f"  • {rule['name']}: {rule['description']} "
                             f"(activation: {rule['activation']:.3f})")
        
        explanation.append("")
        explanation.append("Feature Analysis:")
        for feature, memberships in result.feature_memberships.items():
            dominant = max(memberships, key=memberships.get)
            explanation.append(f"  • {feature}: {result.input_features[feature]:.1f} "
                             f"-> {dominant} ({memberships[dominant]:.3f})")
        
        return "\n".join(explanation)


def create_default_config() -> FuzzyConfig:
    """Create a default fuzzy configuration."""
    return FuzzyConfig()


def load_decision_engine(config_path: Optional[Union[str, Path]] = None) -> FuzzyDecisionEngine:
    """Load a fuzzy decision engine with optional config."""
    if config_path:
        config = FuzzyConfig.from_yaml(config_path)
    else:
        config = create_default_config()
    
    return FuzzyDecisionEngine(config)
