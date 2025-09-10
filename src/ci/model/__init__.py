"""
Model components for CI Medical Diagnosis.

This module provides neural network architectures and feature extractors
for ECG signal classification and analysis.
"""

from .feature_extractor import FeatureExtractor
from .model import create_model

__all__ = ['FeatureExtractor', 'create_model']
