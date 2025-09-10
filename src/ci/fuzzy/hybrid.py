"""
Hybrid model that combines ResNet18 embeddings with the fuzzy decision engine.

This is a minimal implementation to satisfy imports and allow integration tests.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from .engine import FuzzyDecisionEngine, FuzzyConfig, DiagnosisResult


@dataclass
class HybridConfig:
    """Configuration for the hybrid ECG model."""
    fuzzy: FuzzyConfig = FuzzyConfig()


class HybridECGModel:
    """Minimal hybrid model wrapper around the fuzzy decision engine."""

    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        self.engine = FuzzyDecisionEngine(self.config.fuzzy)

    def diagnose_from_embeddings(self, embeddings: np.ndarray, **kwargs) -> DiagnosisResult:
        return self.engine.diagnose_from_embeddings(embeddings, **kwargs)

    def diagnose_from_ecg(self, ecg_data: np.ndarray, sampling_rate: float, **kwargs) -> DiagnosisResult:
        return self.engine.diagnose_from_ecg(ecg_data, sampling_rate, **kwargs)

