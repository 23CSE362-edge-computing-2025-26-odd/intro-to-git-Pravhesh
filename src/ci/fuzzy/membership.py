"""
Fuzzy Membership Functions for ECG Classification

This module provides various membership function types for fuzzy logic operations
including triangular, trapezoidal, and Gaussian membership functions.
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MembershipFunctionConfig:
    """Configuration for membership functions"""
    name: str
    mf_type: str  # 'triangular', 'trapezoidal', 'gaussian'
    parameters: Dict[str, float]


class BaseMembershipFunction:
    """Base class for all membership functions"""
    
    def __init__(self, name: str, parameters: Dict[str, float]):
        self.name = name
        self.parameters = parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate membership function parameters"""
        pass
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate membership degree for input x"""
        raise NotImplementedError("Subclasses must implement membership method")
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Allow calling the membership function directly"""
        return self.membership(x)


class TriangularMF(BaseMembershipFunction):
    """Triangular membership function"""
    
    def _validate_parameters(self):
        required = ['a', 'b', 'c']
        for param in required:
            if param not in self.parameters:
                raise ValueError(f"Triangular MF requires parameter '{param}'")
        
        a, b, c = self.parameters['a'], self.parameters['b'], self.parameters['c']
        if not (a <= b <= c):
            raise ValueError("Triangular MF requires a <= b <= c")
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate triangular membership degree"""
        a, b, c = self.parameters['a'], self.parameters['b'], self.parameters['c']
        x = np.asarray(x)
        
        # Initialize result array
        result = np.zeros_like(x, dtype=float)
        
        # Left slope
        mask1 = (x >= a) & (x <= b) & (b != a)
        result[mask1] = (x[mask1] - a) / (b - a)
        
        # Right slope
        mask2 = (x >= b) & (x <= c) & (c != b)
        result[mask2] = (c - x[mask2]) / (c - b)
        
        # Peak
        result[x == b] = 1.0
        
        return float(result) if np.isscalar(x) else result


class TrapezoidalMF(BaseMembershipFunction):
    """Trapezoidal membership function"""
    
    def _validate_parameters(self):
        required = ['a', 'b', 'c', 'd']
        for param in required:
            if param not in self.parameters:
                raise ValueError(f"Trapezoidal MF requires parameter '{param}'")
        
        a, b, c, d = [self.parameters[p] for p in required]
        if not (a <= b <= c <= d):
            raise ValueError("Trapezoidal MF requires a <= b <= c <= d")
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate trapezoidal membership degree"""
        a, b, c, d = [self.parameters[p] for p in ['a', 'b', 'c', 'd']]
        x = np.asarray(x)
        
        result = np.zeros_like(x, dtype=float)
        
        # Left slope
        mask1 = (x >= a) & (x <= b) & (b != a)
        result[mask1] = (x[mask1] - a) / (b - a)
        
        # Flat top
        mask2 = (x >= b) & (x <= c)
        result[mask2] = 1.0
        
        # Right slope
        mask3 = (x >= c) & (x <= d) & (d != c)
        result[mask3] = (d - x[mask3]) / (d - c)
        
        return float(result) if np.isscalar(x) else result


class GaussianMF(BaseMembershipFunction):
    """Gaussian membership function"""
    
    def _validate_parameters(self):
        required = ['center', 'sigma']
        for param in required:
            if param not in self.parameters:
                raise ValueError(f"Gaussian MF requires parameter '{param}'")
        
        if self.parameters['sigma'] <= 0:
            raise ValueError("Gaussian MF requires sigma > 0")
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate Gaussian membership degree"""
        center = self.parameters['center']
        sigma = self.parameters['sigma']
        x = np.asarray(x)
        
        result = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        return float(result) if np.isscalar(x) else result


class FuzzySet:
    """Represents a fuzzy set with a membership function"""
    
    def __init__(self, name: str, membership_function: BaseMembershipFunction,
                 universe: Optional[np.ndarray] = None):
        self.name = name
        self.membership_function = membership_function
        self.universe = universe
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Get membership degree for value(s) x"""
        return self.membership_function.membership(x)
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Allow calling the fuzzy set directly"""
        return self.membership(x)


class FuzzyVariable:
    """Represents a fuzzy variable with multiple fuzzy sets"""
    
    def __init__(self, name: str, universe: np.ndarray):
        self.name = name
        self.universe = universe
        self.fuzzy_sets: Dict[str, FuzzySet] = {}
    
    def add_fuzzy_set(self, fuzzy_set: FuzzySet):
        """Add a fuzzy set to this variable"""
        self.fuzzy_sets[fuzzy_set.name] = fuzzy_set
        fuzzy_set.universe = self.universe
        logger.debug(f"Added fuzzy set '{fuzzy_set.name}' to variable '{self.name}'")
    
    def create_triangular_set(self, name: str, a: float, b: float, c: float):
        """Create and add a triangular fuzzy set"""
        mf = TriangularMF(name, {'a': a, 'b': b, 'c': c})
        fuzzy_set = FuzzySet(name, mf, self.universe)
        self.add_fuzzy_set(fuzzy_set)
        return fuzzy_set
    
    def create_trapezoidal_set(self, name: str, a: float, b: float, c: float, d: float):
        """Create and add a trapezoidal fuzzy set"""
        mf = TrapezoidalMF(name, {'a': a, 'b': b, 'c': c, 'd': d})
        fuzzy_set = FuzzySet(name, mf, self.universe)
        self.add_fuzzy_set(fuzzy_set)
        return fuzzy_set
    
    def create_gaussian_set(self, name: str, center: float, sigma: float):
        """Create and add a Gaussian fuzzy set"""
        mf = GaussianMF(name, {'center': center, 'sigma': sigma})
        fuzzy_set = FuzzySet(name, mf, self.universe)
        self.add_fuzzy_set(fuzzy_set)
        return fuzzy_set
    
    def get_membership_degrees(self, value: float) -> Dict[str, float]:
        """Get membership degrees for all sets for a given value"""
        degrees = {}
        for set_name, fuzzy_set in self.fuzzy_sets.items():
            degrees[set_name] = fuzzy_set.membership(value)
        return degrees
    
    def get_dominant_set(self, value: float) -> str:
        """Get the name of the fuzzy set with highest membership for given value"""
        degrees = self.get_membership_degrees(value)
        return max(degrees, key=degrees.get)


def create_ecg_risk_variable() -> FuzzyVariable:
    """Create a fuzzy variable for ECG risk assessment"""
    universe = np.linspace(0, 1, 1000)
    risk_var = FuzzyVariable("risk", universe)
    
    # Define risk levels with fuzzy sets
    risk_var.create_trapezoidal_set("low", 0.0, 0.0, 0.2, 0.4)
    risk_var.create_triangular_set("moderate", 0.3, 0.5, 0.7)
    risk_var.create_trapezoidal_set("high", 0.6, 0.8, 1.0, 1.0)
    
    return risk_var


def create_ecg_feature_variables() -> Dict[str, FuzzyVariable]:
    """Create fuzzy variables for ECG features"""
    variables = {}
    
    # Heart rate variable (bpm)
    hr_universe = np.linspace(40, 200, 1000)
    hr_var = FuzzyVariable("heart_rate", hr_universe)
    hr_var.create_trapezoidal_set("bradycardia", 40, 40, 50, 60)
    hr_var.create_trapezoidal_set("normal", 50, 60, 100, 110)
    hr_var.create_trapezoidal_set("tachycardia", 100, 120, 200, 200)
    variables["heart_rate"] = hr_var
    
    # QRS duration variable (ms)
    qrs_universe = np.linspace(60, 200, 1000)
    qrs_var = FuzzyVariable("qrs_duration", qrs_universe)
    qrs_var.create_trapezoidal_set("narrow", 60, 60, 80, 100)
    qrs_var.create_triangular_set("normal", 80, 100, 120)
    qrs_var.create_trapezoidal_set("wide", 100, 120, 200, 200)
    variables["qrs_duration"] = qrs_var
    
    # ST elevation variable (mm)
    st_universe = np.linspace(-2, 5, 1000)
    st_var = FuzzyVariable("st_elevation", st_universe)
    st_var.create_trapezoidal_set("depression", -2, -2, -0.5, 0)
    st_var.create_triangular_set("normal", -0.5, 0, 0.5)
    st_var.create_trapezoidal_set("elevation", 0, 1, 5, 5)
    variables["st_elevation"] = st_var
    
    return variables


def create_membership_function(config: MembershipFunctionConfig) -> BaseMembershipFunction:
    """Factory function to create membership functions from config"""
    mf_type = config.mf_type.lower()
    
    if mf_type == 'triangular':
        return TriangularMF(config.name, config.parameters)
    elif mf_type == 'trapezoidal':
        return TrapezoidalMF(config.name, config.parameters)
    elif mf_type == 'gaussian':
        return GaussianMF(config.name, config.parameters)
    else:
        raise ValueError(f"Unknown membership function type: {config.mf_type}")
