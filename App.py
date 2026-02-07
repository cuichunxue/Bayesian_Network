#!/usr/bin/env python3
"""Bayesian Network Analyzer — Production Demo.

This script demonstrates the full capabilities of the bayesian_network package:
  1. Synthetic data generation with known causal structure
  2. Structure learning with HillClimbSearch
  3. Parameter learning with BDeu prior
  4. Probabilistic inference (variable elimination)
  5. Sensitivity analysis (mutual information)
  6. Interactive visualisation (network, sensitivity, CPD heatmap, dashboard)

All outputs are exported to the ``output/`` directory as interactive HTML files.

Usage::

    python App.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_network import BNConfig, BayesianAnalyzer, NetworkVisualizer
from bayesian_network.logging_config import setup_logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Synthetic data with known causal structure
# ---------------------------------------------------------------------------
def generate_demo_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data with a known causal graph.

    Ground-truth structure::

        Weather ──► Traffic ──► Commute
           │                       ▲
           └──► Accident ──────────┘
        DayOfWeek (independent noise)

    Parameters
    ----------
    n : int
        Number of samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.RandomState(seed)

    weather = rng.choice(["Sunny", "Rainy", "Cloudy"], size=n, p=[0.5, 0.25, 0.25])
    day_of_week = rng.choice(["Weekday", "Weekend"], size=n, p=[0.7, 0.3])

    # Traffic depends on Weather
    traffic = np.where(
        weather == "Rainy",
        rng.choice(["Heavy", "Moderate", "Light"], size=n, p=[0.6, 0.3, 0.1]),
        np.where(
            weather == "Cloudy",
            rng.choice(["Heavy", "Moderate", "Light"], size=n, p=[0.3, 0.5, 0.2]),
            rng.choice(["Heavy", "Moderate", "Light"], size=n, p=[0.1, 0.3, 0.6]),
        ),
    )

    # Accident depends on Weather
    accident_prob = np.where(weather == "Rainy", 0.3, np.where(weather == "Cloudy", 0.15, 0.05))
    accident = np.array(["Yes" if rng.random() < p else "No" for p in accident_prob])

    # Commute depends on Traffic and Accident
    commute = []
    for t, a in zip(traffic, accident):
        if t == "Heavy" and a == "Yes":
            commute.append(rng.choice(["VeryLong", "Long", "Normal"], p=[0.7, 0.25, 0.05]))
        elif t == "Heavy" or a == "Yes":
            commute.append(rng.choice(["VeryLong", "Long", "Normal"], p=[0.2, 0.6, 0.2]))
        elif t == "Moderate":
            commute.append(rng.choice(["VeryLong", "Long", "Normal"], p=[0.05, 0.3, 0.65]))
        else:
            commute.append(rng.choice(["VeryLong", "Long", "Normal"], p=[0.02, 0.08, 0.9]))

    return pd.DataFrame({
        "Weather": weather,
        "DayOfWeek": day_of_week,
        "Traffic": traffic,
        "Accident": accident,
        "Commute": commute,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("Bayesian Network Analyzer — Production Demo")
    logger.info("=" * 60)

    # -- Data --
    df = generate_demo_data(n=2000, seed=42)
    logger.info("Generated %d rows x %d cols", df.shape[0], df.shape[1])
    logger.info("Columns: %s", list(df.columns))
    logger.info("Sample:\n%s", df.head(5).to_string())

    # -- Config --
    config = BNConfig(
        scoring_method="bic-d",
        max_indegree=3,
        prior_type="BDeu",
        equivalent_sample_size=5.0,
        show_progress=True,
        layout_k=2.0,
    )

    # -- Analyse --
    analyzer = BayesianAnalyzer(df, config)
    analyzer.train_model()

    logger.info("-" * 40)
    logger.info("Model summary:")
    for k, v in analyzer.summary().items():
        logger.info("  %s: %s", k, v)

    # -- Inference examples --
    logger.info("-" * 40)
    logger.info("Inference examples:")

    result = analyzer.query("Commute", {"Weather": "Rainy", "Accident": "Yes"})
    logger.info("P(Commute | Weather=Rainy, Accident=Yes):\n%s", result)

    result2 = analyzer.query("Traffic", {"Weather": "Sunny"})
    logger.info("P(Traffic | Weather=Sunny):\n%s", result2)

    # -- Sensitivity --
    logger.info("-" * 40)
    target = "Commute"
    mi = analyzer.compute_sensitivity(target)
    logger.info("MI scores for target '%s':", target)
    for name, score in mi.items():
        logger.info("  %-15s  %.6f", name, score)

    # -- Visualisation --
    logger.info("-" * 40)
    logger.info("Generating visualisations in %s/", OUTPUT_DIR)

    viz = NetworkVisualizer(analyzer)

    viz.plot_network(
        target_node=target,
        html_path=str(OUTPUT_DIR / "network.html"),
    )

    viz.plot_sensitivity(
        target_node=target,
        top_k=10,
        html_path=str(OUTPUT_DIR / "sensitivity.html"),
    )

    # CPD heatmap for a node that has parents
    edges = analyzer.edges
    if edges:
        child_node = edges[0][1]
        viz.plot_cpd_heatmap(
            node=child_node,
            html_path=str(OUTPUT_DIR / "cpd_heatmap.html"),
        )

    viz.plot_dashboard(
        target_node=target,
        top_k=10,
        html_path=str(OUTPUT_DIR / "dashboard.html"),
    )

    logger.info("=" * 60)
    logger.info("All outputs saved to %s/", OUTPUT_DIR)
    logger.info("Open the HTML files in a browser to view interactive charts.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
