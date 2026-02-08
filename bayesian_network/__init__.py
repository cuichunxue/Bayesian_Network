"""Production-grade Bayesian Network analysis toolkit.

Provides structure learning, parameter learning, probabilistic inference,
sensitivity analysis, and interactive visualization for discrete Bayesian Networks.
"""

from bayesian_network.config import BNConfig
from bayesian_network.analyzer import BayesianAnalyzer
from bayesian_network.visualizer import NetworkVisualizer

__all__ = ["BNConfig", "BayesianAnalyzer", "NetworkVisualizer"]
__version__ = "1.1.0"
