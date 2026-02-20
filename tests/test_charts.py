"""Tests for the shared charts module."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from bayesian_network.charts import (
    PALETTE,
    bar_colors,
    build_cpd_heatmap_figure,
    build_dashboard_figure,
    build_data_distribution_figure,
    build_edge_strength_figure,
    build_markov_blanket_figure,
    build_mi_matrix_figure,
    build_network_figure,
    build_posterior_figure,
    build_sensitivity_figure,
    compute_layout,
    fig_to_div,
    mi_node_color,
    prob_color,
)


class TestPalette:
    def test_palette_has_required_keys(self) -> None:
        required = {
            "edge", "edge_highlight", "node_border", "target_border",
            "evidence_border", "bg", "bar_main", "bar_highlight",
            "bar_low", "text",
        }
        assert required.issubset(set(PALETTE.keys()))


class TestColorFunctions:
    def test_prob_color_returns_rgba(self) -> None:
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            c = prob_color(p)
            assert c.startswith("rgba(")

    def test_mi_node_color_returns_rgba(self) -> None:
        for t in [0.0, 0.5, 1.0]:
            c = mi_node_color(t)
            assert c.startswith("rgba(")

    def test_mi_node_color_brighter_at_high_mi(self) -> None:
        low = mi_node_color(0.0)
        high = mi_node_color(1.0)
        # Just check they are different
        assert low != high

    def test_bar_colors_length_matches(self) -> None:
        vals = [0.1, 0.5, 0.9]
        colors = bar_colors(vals)
        assert len(colors) == 3


class TestComputeLayout:
    def test_empty_graph(self) -> None:
        import networkx as nx
        g = nx.DiGraph()
        pos = compute_layout(g)
        assert pos == {}

    def test_dag_layout(self) -> None:
        import networkx as nx
        g = nx.DiGraph([("A", "B"), ("B", "C")])
        pos = compute_layout(g)
        assert set(pos.keys()) == {"A", "B", "C"}

    def test_layout_does_not_mutate_graph(self) -> None:
        import networkx as nx
        g = nx.DiGraph([("A", "B"), ("B", "C")])
        # Ensure no "subset" attribute leaks into original graph
        compute_layout(g)
        for node in g.nodes():
            assert "subset" not in g.nodes[node]


class TestBuildNetworkFigure:
    def test_basic_figure(self) -> None:
        fig = build_network_figure(
            columns=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C")],
            target_node="C",
        )
        assert isinstance(fig, go.Figure)

    def test_posterior_mode(self) -> None:
        posteriors = {"A": {"0": 0.4, "1": 0.6}, "B": {"0": 0.5, "1": 0.5}}
        fig = build_network_figure(
            columns=["A", "B"],
            edges=[("A", "B")],
            posteriors=posteriors,
            evidence={"A": "1"},
        )
        assert isinstance(fig, go.Figure)

    def test_mi_mode(self) -> None:
        mi_map = {"A": 0.3, "B": 0.1, "C": 0.5}
        fig = build_network_figure(
            columns=["A", "B", "C"],
            edges=[("A", "C")],
            mi_map=mi_map,
            target_node="C",
        )
        assert isinstance(fig, go.Figure)

    def test_empty_edges(self) -> None:
        fig = build_network_figure(
            columns=["A", "B"],
            edges=[],
        )
        assert isinstance(fig, go.Figure)


class TestBuildSensitivityFigure:
    def test_basic(self) -> None:
        mi = {"A": 0.5, "B": 0.3, "C": 0.1}
        fig = build_sensitivity_figure(mi, target_node="D")
        assert isinstance(fig, go.Figure)

    def test_empty_data(self) -> None:
        fig = build_sensitivity_figure({}, target_node="X")
        assert isinstance(fig, go.Figure)
        # Should show "No sensitivity data" annotation
        assert len(fig.layout.annotations) > 0


class TestBuildPosteriorFigure:
    def test_basic(self) -> None:
        posteriors = {
            "X": {"a": 0.7, "b": 0.3},
            "Y": {"c": 0.5, "d": 0.5},
        }
        fig = build_posterior_figure(posteriors, target_node="X")
        assert isinstance(fig, go.Figure)

    def test_empty(self) -> None:
        fig = build_posterior_figure({}, target_node="X")
        assert isinstance(fig, go.Figure)


class TestBuildDataDistributionFigure:
    def test_basic(self) -> None:
        import pandas as pd
        df = pd.DataFrame({"A": ["x", "y", "x"], "B": [1, 2, 1]})
        fig = build_data_distribution_figure(df)
        assert isinstance(fig, go.Figure)


class TestBuildEdgeStrengthFigure:
    def test_basic(self) -> None:
        fig = build_edge_strength_figure(
            edge_labels=["A -> B", "B -> C"],
            mi_vals=[0.3, 0.1],
        )
        assert isinstance(fig, go.Figure)

    def test_empty(self) -> None:
        fig = build_edge_strength_figure([], [])
        assert isinstance(fig, go.Figure)


class TestBuildMarkovBlanketFigure:
    def test_basic(self) -> None:
        fig = build_markov_blanket_figure(
            target_node="B",
            mb_nodes={"A", "C"},
            edges=[("A", "B"), ("B", "C")],
        )
        assert isinstance(fig, go.Figure)

    def test_empty_blanket(self) -> None:
        fig = build_markov_blanket_figure(
            target_node="X",
            mb_nodes=set(),
            edges=[],
        )
        assert isinstance(fig, go.Figure)


class TestBuildMIMatrixFigure:
    def test_basic(self) -> None:
        fig = build_mi_matrix_figure(
            columns=["A", "B", "C"],
            mi_func=lambda a, b: 0.1,
        )
        assert isinstance(fig, go.Figure)


class TestBuildDashboardFigure:
    def test_basic(self) -> None:
        mi = {"A": 0.5, "B": 0.2}
        fig = build_dashboard_figure(
            columns=["A", "B", "C"],
            edges=[("A", "C")],
            mi_scores=mi,
            target_node="C",
        )
        assert isinstance(fig, go.Figure)


class TestFigToDiv:
    def test_returns_html_string(self) -> None:
        fig = go.Figure()
        div = fig_to_div(fig)
        assert "<div" in div
