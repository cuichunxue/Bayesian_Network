"""Interactive visualisation for Bayesian Networks.

Provides network graph, sensitivity bar chart, CPD heatmap, and
a combined dashboard — all built with Plotly for interactivity and
HTML export.

Delegates to :mod:`bayesian_network.charts` for all chart construction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import plotly.graph_objects as go

from bayesian_network.charts import (
    build_cpd_heatmap_figure,
    build_dashboard_figure,
    build_network_figure,
    build_sensitivity_figure,
    safe_write_html,
)
from bayesian_network.config import BNConfig

logger = logging.getLogger(__name__)


class NetworkVisualizer:
    """Visualisation toolkit bound to a :class:`BayesianAnalyzer` instance.

    Usage::

        from bayesian_network import BayesianAnalyzer, NetworkVisualizer

        analyzer = BayesianAnalyzer(df).train_model()
        viz = NetworkVisualizer(analyzer)
        viz.plot_network(target_node="Target", html_path="network.html")
        viz.plot_sensitivity("Target", html_path="sensitivity.html")
        viz.plot_dashboard("Target", html_path="dashboard.html")
    """

    def __init__(self, analyzer: Any) -> None:
        self._analyzer = analyzer
        self._cfg: BNConfig = analyzer.config

    # -----------------------------------------------------------------
    # 1. Network graph
    # -----------------------------------------------------------------
    def plot_network(
        self,
        target_node: Optional[str] = None,
        show_mi: bool = True,
        html_path: Optional[str] = None,
        width: int = 1000,
        height: int = 700,
    ) -> go.Figure:
        """Draw the Bayesian Network as a directed graph.

        Nodes are coloured and sized by their mutual information with
        *target_node*.  Edges are drawn as arrows.

        Parameters
        ----------
        target_node : str, optional
            When given, MI values colour/size the nodes.
        show_mi : bool
            Whether to map MI values to visual attributes.
        html_path : str, optional
            If provided, export the figure to this HTML file.
        width, height : int
            Figure dimensions in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        a = self._analyzer
        if not a.is_trained:
            raise RuntimeError("Model not trained. Call train_model() first.")

        if not a.columns:
            raise RuntimeError("No nodes to visualise.")

        mi_map: Dict[str, float] = {}
        if show_mi and target_node and target_node in a.columns:
            mi_map = a.compute_sensitivity(target_node)
            max_mi = max(mi_map.values()) if mi_map else 0.0
            mi_map[target_node] = max_mi * 1.2 if max_mi > 0 else 1.0

        fig = build_network_figure(
            columns=a.columns,
            edges=a.edges,
            layout_k=self._cfg.layout_k,
            layout_seed=self._cfg.layout_seed,
            target_node=target_node,
            mi_map=mi_map,
            node_size_min=self._cfg.node_size_min,
            node_size_max=self._cfg.node_size_max,
            default_node_size=self._cfg.default_node_size,
            width=width,
            height=height,
        )

        if html_path:
            safe_write_html(fig, html_path)

        return fig

    # -----------------------------------------------------------------
    # 2. Sensitivity bar chart
    # -----------------------------------------------------------------
    def plot_sensitivity(
        self,
        target_node: str,
        top_k: int = 20,
        html_path: Optional[str] = None,
        width: int = 900,
        height: int = 500,
    ) -> go.Figure:
        """Horizontal bar chart of mutual-information scores.

        Parameters
        ----------
        target_node : str
            Variable whose sensitivity is analysed.
        top_k : int
            Number of top variables to show.
        html_path : str, optional
            Export path.
        width, height : int
            Figure dimensions.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        mi = self._analyzer.compute_sensitivity(target_node)

        fig = build_sensitivity_figure(
            mi_scores=mi,
            target_node=target_node,
            top_k=top_k,
            width=width,
            height=height,
        )

        if html_path:
            safe_write_html(fig, html_path)

        return fig

    # -----------------------------------------------------------------
    # 3. CPD heatmap
    # -----------------------------------------------------------------
    def plot_cpd_heatmap(
        self,
        node: str,
        html_path: Optional[str] = None,
        width: int = 800,
        height: int = 500,
    ) -> go.Figure:
        """Plot the conditional probability distribution of *node* as a heatmap.

        Parameters
        ----------
        node : str
            Variable whose CPD is visualised.
        html_path : str, optional
            Export path.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        a = self._analyzer
        if not a.is_trained:
            raise RuntimeError("Model not trained.")

        cpd = a.model.get_cpds(node)

        fig = build_cpd_heatmap_figure(cpd=cpd, node=node, width=width, height=height)

        if html_path:
            safe_write_html(fig, html_path)

        return fig

    # -----------------------------------------------------------------
    # 4. Combined dashboard
    # -----------------------------------------------------------------
    def plot_dashboard(
        self,
        target_node: str,
        top_k: int = 15,
        html_path: Optional[str] = None,
        width: int = 1400,
        height: int = 750,
    ) -> go.Figure:
        """Two-panel dashboard: network graph (left) + sensitivity bars (right).

        Parameters
        ----------
        target_node : str
            Variable of interest.
        top_k : int
            Number of bars in the sensitivity panel.
        html_path : str, optional
            Export path.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        a = self._analyzer
        if not a.is_trained:
            raise RuntimeError("Model not trained.")

        mi = a.compute_sensitivity(target_node)

        fig = build_dashboard_figure(
            columns=a.columns,
            edges=a.edges,
            mi_scores=mi,
            target_node=target_node,
            layout_k=self._cfg.layout_k,
            layout_seed=self._cfg.layout_seed,
            top_k=top_k,
            node_size_min=self._cfg.node_size_min,
            node_size_max=self._cfg.node_size_max,
            default_node_size=self._cfg.default_node_size,
            width=width,
            height=height,
        )

        if html_path:
            safe_write_html(fig, html_path)

        return fig
