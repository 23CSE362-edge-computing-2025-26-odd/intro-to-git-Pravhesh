"""
Unit tests for fuzzy logic modules: membership functions, rules, decision engine, and PSO optimizer (smoke tests).
"""

import numpy as np
import pytest

from src.ci.fuzzy.membership import (
    TriangularMF, TrapezoidalMF, GaussianMF, FuzzyVariable,
    create_ecg_risk_variable, create_ecg_feature_variables
)
from src.ci.fuzzy.rules import (
    FuzzyCondition, FuzzyAntecedent, FuzzyConsequent, FuzzyRule, FuzzyRuleBase,
    RuleEngine, InferenceMethod, create_ecg_rule_base
)
from src.ci.fuzzy.engine import FuzzyDecisionEngine, FuzzyConfig
from src.ci.fuzzy.optimize import FuzzyPSOConfig, PSOOptimizer


class TestMembershipFunctions:
    def test_triangular_membership(self):
        mf = TriangularMF("test", {"a": 0, "b": 5, "c": 10})
        x = np.array([0, 2.5, 5, 7.5, 10])
        y = mf.membership(x)
        assert np.allclose(y, [0.0, 0.5, 1.0, 0.5, 0.0])

    def test_trapezoidal_membership(self):
        mf = TrapezoidalMF("test", {"a": 0, "b": 2, "c": 8, "d": 10})
        x = np.array([0, 1, 3, 8, 9, 10])
        y = mf.membership(x)
        assert np.isclose(y[0], 0.0)
        assert y[2] == 1.0
        assert y[-1] == 0.0

    def test_gaussian_membership(self):
        mf = GaussianMF("test", {"center": 0, "sigma": 1})
        assert pytest.approx(mf.membership(0.0), 1e-6) == 1.0
        assert mf.membership(3.0) < 0.1


class TestRuleEngine:
    def test_rule_firing(self):
        variables = create_ecg_feature_variables()
        risk_var = create_ecg_risk_variable()

        rule_base = FuzzyRuleBase("test")
        rule_base.add_variable(risk_var)
        for var in variables.values():
            rule_base.add_variable(var)

        # If HR normal AND QRS normal THEN risk low
        conditions = [
            ("heart_rate", "normal", False),
            ("qrs_duration", "normal", False)
        ]
        rule = rule_base.create_rule(
            name="normal_low_risk",
            conditions=conditions,
            operators=["AND"],
            consequent=("risk", "low"),
            weight=1.0
        )

        engine = RuleEngine(rule_base)
        inputs = {"heart_rate": 70, "qrs_duration": 100, "st_elevation": 0}
        aggregated = engine.infer(inputs)

        assert "risk" in aggregated
        assert aggregated["risk"]["low"] > 0

    def test_defuzzification_bounds(self):
        risk_var = create_ecg_risk_variable()
        engine = RuleEngine(FuzzyRuleBase())
        fuzzy_output = {"low": 0.8, "high": 0.3}
        value = engine.defuzzify(fuzzy_output, risk_var, method="centroid")
        assert 0.0 <= value <= 1.0


class TestDecisionEngine:
    def test_diagnose_from_embeddings(self):
        engine = FuzzyDecisionEngine(FuzzyConfig())
        embedding = np.random.randn(512)
        result = engine.diagnose_from_embeddings(embedding)
        assert 0.0 <= result.risk_score <= 1.0
        assert result.risk_level in {"low", "moderate", "high"}
        assert 0.0 <= result.confidence <= 1.0

    def test_diagnose_from_ecg(self):
        engine = FuzzyDecisionEngine(FuzzyConfig())
        ecg = np.random.randn(3600)  # 10s at 360Hz
        result = engine.diagnose_from_ecg(ecg, sampling_rate=360)
        assert 0.0 <= result.risk_score <= 1.0
        assert result.risk_level in {"low", "moderate", "high"}


class TestPSOOptimizer:
    def test_pso_smoke(self):
        # Minimal smoke test: small dataset with 3 classes (low/moderate/high)
        np.random.seed(0)
        X = np.random.randn(60, 512)
        y = np.repeat([0, 1, 2], 20)

        pso_config = FuzzyPSOConfig(num_particles=5, max_iterations=3, cv_folds=2, patience=2)
        optimizer = PSOOptimizer(pso_config)
        config = FuzzyConfig()

        result = optimizer.optimize(config, X, y)
        assert isinstance(result.best_fitness, float)
        assert len(result.fitness_history) >= 1
        assert 'accuracy' in result.validation_scores

