"""Tests for BNConfig validation and defaults."""

from __future__ import annotations

import pytest

from bayesian_network.config import BNConfig, ConfigValidationError


class TestBNConfigDefaults:
    def test_default_creation(self) -> None:
        cfg = BNConfig()
        assert cfg.scoring_method == "bic-d"
        assert cfg.max_indegree == 3
        assert cfg.prior_type == "BDeu"
        assert cfg.equivalent_sample_size == 5.0

    def test_immutable(self) -> None:
        cfg = BNConfig()
        with pytest.raises(AttributeError):
            cfg.scoring_method = "k2"  # type: ignore[misc]


class TestBNConfigValidation:
    def test_invalid_scoring_method(self) -> None:
        with pytest.raises(ConfigValidationError, match="scoring_method"):
            BNConfig(scoring_method="invalid")

    def test_invalid_prior_type(self) -> None:
        with pytest.raises(ConfigValidationError, match="prior_type"):
            BNConfig(prior_type="flat")

    def test_max_indegree_too_low(self) -> None:
        with pytest.raises(ConfigValidationError, match="max_indegree"):
            BNConfig(max_indegree=0)

    def test_negative_tabu_length(self) -> None:
        with pytest.raises(ConfigValidationError, match="tabu_length"):
            BNConfig(tabu_length=-1)

    def test_zero_equivalent_sample_size(self) -> None:
        with pytest.raises(ConfigValidationError, match="equivalent_sample_size"):
            BNConfig(equivalent_sample_size=0)

    def test_node_size_max_leq_min(self) -> None:
        with pytest.raises(ConfigValidationError, match="node_size_max"):
            BNConfig(node_size_min=50.0, node_size_max=20.0)

    def test_discretize_bins_too_small(self) -> None:
        with pytest.raises(ConfigValidationError, match="discretize_bins"):
            BNConfig(discretize_bins=1)

    def test_valid_alternative_scoring(self) -> None:
        cfg = BNConfig(scoring_method="k2")
        assert cfg.scoring_method == "k2"
