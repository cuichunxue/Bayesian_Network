"""Shared fixtures for Bayesian Network tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_network.config import BNConfig
from bayesian_network.analyzer import BayesianAnalyzer


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """DataFrame with clear statistical dependencies for testing."""
    rng = np.random.RandomState(42)
    n = 500
    a = rng.randint(0, 3, size=n)
    b = rng.randint(0, 2, size=n)
    c = ((a + b) >= 2).astype(int)
    d = rng.randint(0, 2, size=n)
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d})


@pytest.fixture()
def default_config() -> BNConfig:
    return BNConfig(show_progress=False)


@pytest.fixture()
def trained_analyzer(sample_df: pd.DataFrame, default_config: BNConfig) -> BayesianAnalyzer:
    """Return an analyzer that has already been trained."""
    return BayesianAnalyzer(sample_df, default_config).train_model()
