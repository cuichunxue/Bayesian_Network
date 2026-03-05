"""Tests for BayesianAnalyzer core functionality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_network.config import BNConfig
from bayesian_network.analyzer import (
    BayesianAnalyzer,
    ModelNotTrainedError,
)


class TestAnalyzerInit:
    def test_empty_dataframe_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            BayesianAnalyzer(pd.DataFrame())

    def test_single_column_raises(self) -> None:
        with pytest.raises(ValueError, match="At least 2 columns"):
            BayesianAnalyzer(pd.DataFrame({"A": [1, 2, 3]}))

    def test_properties_before_training(self, sample_df: pd.DataFrame) -> None:
        a = BayesianAnalyzer(sample_df)
        assert not a.is_trained
        assert a.edges == []
        assert a.model is None
        assert a.training_time == 0.0
        assert set(a.columns) == {"A", "B", "C", "D"}


class TestDataPreparation:
    def test_inf_replaced(self) -> None:
        df = pd.DataFrame({"X": [1, np.inf, 3], "Y": [4, 5, -np.inf]})
        a = BayesianAnalyzer(df)
        data = a.data
        assert "__MISSING__" in data["X"].astype(str).values
        assert "__MISSING__" in data["Y"].astype(str).values

    def test_nan_replaced(self) -> None:
        df = pd.DataFrame({"X": [1, np.nan, 3], "Y": [4, 5, 6]})
        a = BayesianAnalyzer(df)
        data = a.data
        assert "__MISSING__" in data["X"].astype(str).values

    def test_custom_na_token(self) -> None:
        df = pd.DataFrame({"X": [1, np.nan], "Y": [3, 4]})
        cfg = BNConfig(na_token="NONE")
        a = BayesianAnalyzer(df, cfg)
        assert "NONE" in a.data["X"].astype(str).values

    def test_discretize_numeric(self) -> None:
        rng = np.random.RandomState(0)
        df = pd.DataFrame({
            "X": rng.randn(200),
            "Y": rng.randint(0, 2, size=200),
        })
        cfg = BNConfig(discretize_numeric=True, discretize_bins=4, max_states_per_col=10)
        a = BayesianAnalyzer(df, cfg)
        assert a.data["X"].nunique() <= 10

    def test_data_returns_copy(self, sample_df: pd.DataFrame) -> None:
        a = BayesianAnalyzer(sample_df)
        d1 = a.data
        d2 = a.data
        assert d1 is not d2


class TestTraining:
    def test_train_returns_self(self, sample_df: pd.DataFrame, default_config: BNConfig) -> None:
        a = BayesianAnalyzer(sample_df, default_config)
        result = a.train_model()
        assert result is a

    def test_is_trained_after_fit(self, trained_analyzer: BayesianAnalyzer) -> None:
        assert trained_analyzer.is_trained
        assert trained_analyzer.model is not None
        assert trained_analyzer.training_time > 0

    def test_edges_are_tuples(self, trained_analyzer: BayesianAnalyzer) -> None:
        for edge in trained_analyzer.edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 2

    def test_train_clears_mi_cache(self, sample_df: pd.DataFrame, default_config: BNConfig) -> None:
        a = BayesianAnalyzer(sample_df, default_config)
        a.train_model()
        a.compute_sensitivity("C")
        a.train_model()
        # After retraining, cache should be empty (fresh computation)
        assert a._mi_cache == {}


class TestInference:
    def test_query_before_training_raises(self, sample_df: pd.DataFrame) -> None:
        a = BayesianAnalyzer(sample_df)
        with pytest.raises(ModelNotTrainedError):
            a.query("C")

    def test_query_unknown_target_raises(self, trained_analyzer: BayesianAnalyzer) -> None:
        with pytest.raises(KeyError, match="NONEXISTENT"):
            trained_analyzer.query("NONEXISTENT")

    def test_query_unknown_evidence_raises(self, trained_analyzer: BayesianAnalyzer) -> None:
        with pytest.raises(KeyError, match="Unknown evidence"):
            trained_analyzer.query("C", {"ZZZ": "1"})

    def test_query_returns_factor(self, trained_analyzer: BayesianAnalyzer) -> None:
        result = trained_analyzer.query("C")
        assert hasattr(result, "values")
        # Probabilities sum to 1
        assert abs(result.values.sum() - 1.0) < 1e-6

    def test_query_with_evidence(self, trained_analyzer: BayesianAnalyzer) -> None:
        result = trained_analyzer.query("C", {"A": "0"})
        assert abs(result.values.sum() - 1.0) < 1e-6


class TestSensitivity:
    def test_unknown_target_raises(self, trained_analyzer: BayesianAnalyzer) -> None:
        with pytest.raises(KeyError):
            trained_analyzer.compute_sensitivity("NONEXISTENT")

    def test_returns_dict(self, trained_analyzer: BayesianAnalyzer) -> None:
        mi = trained_analyzer.compute_sensitivity("C")
        assert isinstance(mi, dict)
        assert "C" not in mi  # target excluded
        assert all(isinstance(v, float) for v in mi.values())
        assert all(v >= 0 for v in mi.values())

    def test_sorted_descending(self, trained_analyzer: BayesianAnalyzer) -> None:
        mi = trained_analyzer.compute_sensitivity("C")
        values = list(mi.values())
        assert values == sorted(values, reverse=True)

    def test_cache_hit(self, trained_analyzer: BayesianAnalyzer) -> None:
        mi1 = trained_analyzer.compute_sensitivity("C")
        mi2 = trained_analyzer.compute_sensitivity("C")
        assert mi1 == mi2

    def test_invalidate_cache(self, trained_analyzer: BayesianAnalyzer) -> None:
        trained_analyzer.compute_sensitivity("C")
        trained_analyzer.invalidate_cache()
        assert trained_analyzer._mi_cache == {}


class TestSummary:
    def test_summary_keys(self, trained_analyzer: BayesianAnalyzer) -> None:
        s = trained_analyzer.summary()
        expected_keys = {
            "is_trained", "n_rows", "n_columns", "columns",
            "n_edges", "edges", "training_time_sec", "config",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_before_training(self, sample_df: pd.DataFrame) -> None:
        a = BayesianAnalyzer(sample_df)
        s = a.summary()
        assert s["is_trained"] is False
        assert s["n_edges"] == 0
