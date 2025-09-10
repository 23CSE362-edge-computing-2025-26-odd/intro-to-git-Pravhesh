"""
Fuzzy Rule Engine for ECG Classification

This module provides fuzzy rule-based inference for cardiac diagnosis including
rule definition, rule firing, and inference mechanisms.
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .membership import FuzzyVariable, FuzzySet

logger = logging.getLogger(__name__)


class FuzzyOperator(Enum):
    """Fuzzy operators for rule antecedents and consequents"""
    AND = "and"
    OR = "or"
    NOT = "not"


class InferenceMethod(Enum):
    """Methods for fuzzy inference"""
    MAMDANI = "mamdani"
    SUGENO = "sugeno"


@dataclass
class FuzzyCondition:
    """Represents a single fuzzy condition (variable IS fuzzy_set)"""
    variable_name: str
    fuzzy_set_name: str
    negated: bool = False
    
    def evaluate(self, variables: Dict[str, FuzzyVariable], 
                 input_values: Dict[str, float]) -> float:
        """Evaluate the condition with given input values"""
        if self.variable_name not in variables:
            raise ValueError(f"Variable '{self.variable_name}' not found")
        
        if self.variable_name not in input_values:
            raise ValueError(f"Input value for '{self.variable_name}' not provided")
        
        variable = variables[self.variable_name]
        if self.fuzzy_set_name not in variable.fuzzy_sets:
            raise ValueError(f"Fuzzy set '{self.fuzzy_set_name}' not found in variable '{self.variable_name}'")
        
        membership = variable.fuzzy_sets[self.fuzzy_set_name].membership(
            input_values[self.variable_name]
        )
        
        return (1.0 - membership) if self.negated else membership


@dataclass
class FuzzyAntecedent:
    """Represents the antecedent (IF part) of a fuzzy rule"""
    conditions: List[FuzzyCondition]
    operators: List[FuzzyOperator] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate that operators list has correct length"""
        if len(self.conditions) > 1 and len(self.operators) != len(self.conditions) - 1:
            raise ValueError("Number of operators must be one less than number of conditions")
    
    def evaluate(self, variables: Dict[str, FuzzyVariable], 
                 input_values: Dict[str, float]) -> float:
        """Evaluate the antecedent with given input values"""
        if not self.conditions:
            return 1.0
        
        # Evaluate first condition
        result = self.conditions[0].evaluate(variables, input_values)
        
        # Apply operators sequentially
        for i, operator in enumerate(self.operators):
            next_result = self.conditions[i + 1].evaluate(variables, input_values)
            
            if operator == FuzzyOperator.AND:
                result = min(result, next_result)
            elif operator == FuzzyOperator.OR:
                result = max(result, next_result)
        
        return result


@dataclass
class FuzzyConsequent:
    """Represents the consequent (THEN part) of a fuzzy rule"""
    variable_name: str
    fuzzy_set_name: str
    certainty_factor: float = 1.0  # Confidence in this rule
    
    def __post_init__(self):
        """Validate certainty factor"""
        if not (0.0 <= self.certainty_factor <= 1.0):
            raise ValueError("Certainty factor must be between 0 and 1")


@dataclass
class FuzzyRule:
    """Represents a complete fuzzy rule (IF ... THEN ...)"""
    name: str
    antecedent: FuzzyAntecedent
    consequent: FuzzyConsequent
    weight: float = 1.0
    description: str = ""
    
    def __post_init__(self):
        """Validate rule weight"""
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError("Rule weight must be between 0 and 1")
    
    def fire(self, variables: Dict[str, FuzzyVariable], 
             input_values: Dict[str, float]) -> Tuple[str, str, float]:
        """Fire the rule and return (output_var, output_set, activation_level)"""
        activation = self.antecedent.evaluate(variables, input_values)
        activation *= self.weight * self.consequent.certainty_factor
        
        return (
            self.consequent.variable_name,
            self.consequent.fuzzy_set_name,
            activation
        )


class FuzzyRuleBase:
    """Collection of fuzzy rules for inference"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.rules: List[FuzzyRule] = []
        self.variables: Dict[str, FuzzyVariable] = {}
    
    def add_rule(self, rule: FuzzyRule):
        """Add a rule to the rule base"""
        self.rules.append(rule)
        logger.debug(f"Added rule '{rule.name}' to rule base '{self.name}'")
    
    def add_variable(self, variable: FuzzyVariable):
        """Add a variable to the rule base"""
        self.variables[variable.name] = variable
        logger.debug(f"Added variable '{variable.name}' to rule base '{self.name}'")
    
    def create_rule(self, name: str, conditions: List[Tuple[str, str, bool]], 
                   operators: List[str], consequent: Tuple[str, str], 
                   weight: float = 1.0, certainty: float = 1.0, 
                   description: str = "") -> FuzzyRule:
        """Create and add a rule from simplified format"""
        
        # Create conditions
        fuzzy_conditions = []
        for var_name, set_name, negated in conditions:
            fuzzy_conditions.append(FuzzyCondition(var_name, set_name, negated))
        
        # Create operators
        fuzzy_operators = []
        for op_str in operators:
            if op_str.upper() == "AND":
                fuzzy_operators.append(FuzzyOperator.AND)
            elif op_str.upper() == "OR":
                fuzzy_operators.append(FuzzyOperator.OR)
            else:
                raise ValueError(f"Unknown operator: {op_str}")
        
        # Create antecedent and consequent
        antecedent = FuzzyAntecedent(fuzzy_conditions, fuzzy_operators)
        consequent_obj = FuzzyConsequent(consequent[0], consequent[1], certainty)
        
        # Create rule
        rule = FuzzyRule(name, antecedent, consequent_obj, weight, description)
        self.add_rule(rule)
        return rule
    
    def get_active_rules(self, input_values: Dict[str, float], 
                        threshold: float = 0.0) -> List[Tuple[FuzzyRule, float]]:
        """Get all rules that fire above threshold with their activation levels"""
        active_rules = []
        
        for rule in self.rules:
            try:
                _, _, activation = rule.fire(self.variables, input_values)
                if activation > threshold:
                    active_rules.append((rule, activation))
            except Exception as e:
                logger.warning(f"Error firing rule '{rule.name}': {e}")
        
        return sorted(active_rules, key=lambda x: x[1], reverse=True)


class RuleEngine:
    """Main fuzzy rule engine for inference"""
    
    def __init__(self, rule_base: FuzzyRuleBase, 
                 inference_method: InferenceMethod = InferenceMethod.MAMDANI):
        self.rule_base = rule_base
        self.inference_method = inference_method
    
    def infer(self, input_values: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Perform fuzzy inference and return aggregated results"""
        # Get all active rules
        active_rules = self.rule_base.get_active_rules(input_values)
        
        # Aggregate results by output variable and fuzzy set
        aggregated = {}
        for rule, activation in active_rules:
            _, _, result = rule.fire(self.rule_base.variables, input_values)
            var_name = rule.consequent.variable_name
            set_name = rule.consequent.fuzzy_set_name
            
            if var_name not in aggregated:
                aggregated[var_name] = {}
            
            # Use maximum aggregation (can be changed to sum or other methods)
            if set_name not in aggregated[var_name]:
                aggregated[var_name][set_name] = result
            else:
                aggregated[var_name][set_name] = max(aggregated[var_name][set_name], result)
        
        return aggregated
    
    def defuzzify(self, fuzzy_output: Dict[str, float], 
                  output_variable: FuzzyVariable, 
                  method: str = "centroid") -> float:
        """Defuzzify fuzzy output to crisp value"""
        if method == "centroid":
            return self._centroid_defuzzification(fuzzy_output, output_variable)
        elif method == "maximum":
            return self._maximum_defuzzification(fuzzy_output, output_variable)
        elif method == "mean_of_maxima":
            return self._mean_of_maxima_defuzzification(fuzzy_output, output_variable)
        else:
            raise ValueError(f"Unknown defuzzification method: {method}")
    
    def _centroid_defuzzification(self, fuzzy_output: Dict[str, float], 
                                 output_variable: FuzzyVariable) -> float:
        """Centroid (center of gravity) defuzzification"""
        universe = output_variable.universe
        numerator = 0.0
        denominator = 0.0
        
        for x in universe:
            membership = 0.0
            for set_name, activation in fuzzy_output.items():
                if set_name in output_variable.fuzzy_sets:
                    set_membership = output_variable.fuzzy_sets[set_name].membership(x)
                    # Apply clipping (Mamdani inference)
                    membership = max(membership, min(activation, set_membership))
            
            numerator += x * membership
            denominator += membership
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _maximum_defuzzification(self, fuzzy_output: Dict[str, float], 
                                output_variable: FuzzyVariable) -> float:
        """Maximum defuzzification (pick value with highest membership)"""
        max_membership = 0.0
        result_value = 0.0
        
        for x in output_variable.universe:
            membership = 0.0
            for set_name, activation in fuzzy_output.items():
                if set_name in output_variable.fuzzy_sets:
                    set_membership = output_variable.fuzzy_sets[set_name].membership(x)
                    membership = max(membership, min(activation, set_membership))
            
            if membership > max_membership:
                max_membership = membership
                result_value = x
        
        return result_value
    
    def _mean_of_maxima_defuzzification(self, fuzzy_output: Dict[str, float], 
                                       output_variable: FuzzyVariable) -> float:
        """Mean of maxima defuzzification"""
        max_membership = 0.0
        max_values = []
        
        # Find maximum membership level
        for x in output_variable.universe:
            membership = 0.0
            for set_name, activation in fuzzy_output.items():
                if set_name in output_variable.fuzzy_sets:
                    set_membership = output_variable.fuzzy_sets[set_name].membership(x)
                    membership = max(membership, min(activation, set_membership))
            
            if membership > max_membership:
                max_membership = membership
                max_values = [x]
            elif membership == max_membership and membership > 0:
                max_values.append(x)
        
        return np.mean(max_values) if max_values else 0.0


def create_ecg_rule_base() -> FuzzyRuleBase:
    """Create a sample rule base for ECG diagnosis"""
    rule_base = FuzzyRuleBase("ecg_diagnosis")
    
    # Add variables (will be added when rules are created)
    from .membership import create_ecg_risk_variable, create_ecg_feature_variables
    
    risk_var = create_ecg_risk_variable()
    feature_vars = create_ecg_feature_variables()
    
    rule_base.add_variable(risk_var)
    for var in feature_vars.values():
        rule_base.add_variable(var)
    
    # Rule 1: Normal heart rate + normal QRS + normal ST = low risk
    rule_base.create_rule(
        name="normal_ecg",
        conditions=[
            ("heart_rate", "normal", False),
            ("qrs_duration", "normal", False),
            ("st_elevation", "normal", False)
        ],
        operators=["AND", "AND"],
        consequent=("risk", "low"),
        weight=1.0,
        description="Normal ECG parameters indicate low risk"
    )
    
    # Rule 2: Tachycardia + wide QRS = high risk
    rule_base.create_rule(
        name="tachy_wide_qrs",
        conditions=[
            ("heart_rate", "tachycardia", False),
            ("qrs_duration", "wide", False)
        ],
        operators=["AND"],
        consequent=("risk", "high"),
        weight=0.9,
        description="Tachycardia with wide QRS suggests serious arrhythmia"
    )
    
    # Rule 3: ST elevation = high risk
    rule_base.create_rule(
        name="st_elevation",
        conditions=[("st_elevation", "elevation", False)],
        operators=[],
        consequent=("risk", "high"),
        weight=1.0,
        description="ST elevation indicates potential myocardial infarction"
    )
    
    # Rule 4: Bradycardia alone = moderate risk
    rule_base.create_rule(
        name="bradycardia",
        conditions=[("heart_rate", "bradycardia", False)],
        operators=[],
        consequent=("risk", "moderate"),
        weight=0.7,
        description="Bradycardia may indicate conduction issues"
    )
    
    # Rule 5: ST depression = moderate risk
    rule_base.create_rule(
        name="st_depression",
        conditions=[("st_elevation", "depression", False)],
        operators=[],
        consequent=("risk", "moderate"),
        weight=0.8,
        description="ST depression may indicate ischemia"
    )
    
    return rule_base
