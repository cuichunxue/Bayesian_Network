"""Configuration for Bayesian Network analysis.

All tunable parameters are centralized here with validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

VALID_SCORING_METHODS = frozenset({"bic-d", "bic", "k2", "bdeu", "aic-d", "aic"})
VALID_PRIOR_TYPES = frozenset({"BDeu", "K2", "dirichlet"})


class ConfigValidationError(ValueError):
    """Raised when a configuration value is invalid."""


@dataclass(frozen=True, slots=True)
class BNConfig:
    """Immutable configuration for :class:`BayesianAnalyzer`.

    Parameters
    ----------
    scoring_method : str
        Scoring metric for structure learning.
        One of ``"bic-d"`` (default), ``"bic"``, ``"k2"``, ``"bdeu"``,
        ``"aic-d"``, ``"aic"``.
    max_indegree : int
        Maximum number of parents per node (default 3).
    tabu_length : int
        Tabu search memory length. 0 disables tabu (default).
    show_progress : bool
        Show progress bar during structure learning.
    prior_type : str
        Prior distribution for parameter learning (``"BDeu"``, ``"K2"``,
        ``"dirichlet"``).
    equivalent_sample_size : float
        Pseudo-count parameter for BDeu prior.
    na_token : str
        Sentinel string used to replace ``NaN`` / ``inf`` values.
    max_states_per_col : int
        Columns exceeding this unique-value count are forced to ``str``.
    discretize_numeric : bool
        If ``True``, automatically bin numeric columns.
    discretize_bins : int
        Number of quantile bins when discretizing.
    node_size_min : float
        Minimum marker size in the network graph.
    node_size_max : float
        Maximum marker size in the network graph.
    default_node_size : float
        Marker size when MI information is not available.
    layout_seed : int
        Random seed for reproducible graph layouts.
    layout_k : float
        Optimal node distance for spring-layout algorithm.
    color_scheme : str
        Plotly-compatible color scale name (e.g. ``"Viridis"``).
    """

    # --- Structure learning ---
    scoring_method: str = "bic-d"
    max_indegree: int = 3
    tabu_length: int = 0
    show_progress: bool = True

    # --- Parameter learning ---
    prior_type: str = "BDeu"
    equivalent_sample_size: float = 5.0

    # --- Data handling ---
    na_token: str = "__MISSING__"
    max_states_per_col: int = 30
    discretize_numeric: bool = False
    discretize_bins: int = 5

    # --- Visualization ---
    node_size_min: float = 18.0
    node_size_max: float = 45.0
    default_node_size: float = 28.0
    layout_seed: int = 42
    layout_k: float = 1.8
    color_scheme: str = "Viridis"

    def __post_init__(self) -> None:
        errors: list[str] = []

        if self.scoring_method not in VALID_SCORING_METHODS:
            errors.append(
                f"scoring_method must be one of {sorted(VALID_SCORING_METHODS)}, "
                f"got '{self.scoring_method}'"
            )
        if self.prior_type not in VALID_PRIOR_TYPES:
            errors.append(
                f"prior_type must be one of {sorted(VALID_PRIOR_TYPES)}, "
                f"got '{self.prior_type}'"
            )
        if self.max_indegree < 1:
            errors.append(f"max_indegree must be >= 1, got {self.max_indegree}")
        if self.tabu_length < 0:
            errors.append(f"tabu_length must be >= 0, got {self.tabu_length}")
        if self.equivalent_sample_size <= 0:
            errors.append(
                f"equivalent_sample_size must be > 0, got {self.equivalent_sample_size}"
            )
        if self.max_states_per_col < 2:
            errors.append(
                f"max_states_per_col must be >= 2, got {self.max_states_per_col}"
            )
        if self.discretize_bins < 2:
            errors.append(f"discretize_bins must be >= 2, got {self.discretize_bins}")
        if self.node_size_min < 0:
            errors.append(f"node_size_min must be >= 0, got {self.node_size_min}")
        if self.node_size_max <= self.node_size_min:
            errors.append(
                f"node_size_max ({self.node_size_max}) must be > "
                f"node_size_min ({self.node_size_min})"
            )

        if errors:
            raise ConfigValidationError(
                "Invalid BNConfig:\n  - " + "\n  - ".join(errors)
            )
