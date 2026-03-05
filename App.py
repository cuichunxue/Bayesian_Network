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

    # Standard batch mode (all charts)
    python App.py

    # Interactive dashboard with evidence
    python App.py --interactive
    python App.py --interactive --evidence Weather=Rainy Accident=Yes
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_network import BNConfig, BayesianAnalyzer, NetworkVisualizer
from bayesian_network.demo_data import generate_demo_data
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
#    (delegated to bayesian_network.demo_data)
# ---------------------------------------------------------------------------


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian Network Analyzer")
    parser.add_argument(
        "--interactive", action="store_true",
        help="Generate interactive dashboard HTML with evidence support",
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Launch Flask web application for interactive analysis",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Web server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Web server port (default: 5000)",
    )
    parser.add_argument(
        "--evidence", nargs="*", default=[],
        help="Evidence as KEY=VALUE pairs (e.g. Weather=Rainy Accident=Yes)",
    )
    parser.add_argument(
        "--target", type=str, default="Commute",
        help="Target node for the dashboard (default: Commute)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for dashboard HTML (default: output/dashboard_interactive.html)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.web:
        from bayesian_network.webapp import main as web_main
        web_main(host=args.host, port=args.port)
    elif args.interactive:
        # Parse evidence
        evidence = {}
        for item in args.evidence:
            if "=" in item:
                k, v = item.split("=", 1)
                evidence[k] = v

        df = generate_demo_data(n=2000, seed=42)
        config = BNConfig(
            scoring_method="bic-d",
            max_indegree=3,
            prior_type="BDeu",
            equivalent_sample_size=5.0,
            show_progress=True,
            layout_k=2.0,
        )
        analyzer = BayesianAnalyzer(df, config)
        analyzer.train_model()

        from bayesian_network.interactive import save_dashboard
        out = args.output or str(OUTPUT_DIR / "dashboard_interactive.html")
        path = save_dashboard(
            analyzer, target_node=args.target,
            evidence=evidence, html_path=out,
        )
        logger.info("Dashboard saved: %s", path)
    else:
        main()
