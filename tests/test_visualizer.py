"""Tests for NetworkVisualizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import plotly.graph_objects as go

from bayesian_network.analyzer import BayesianAnalyzer
from bayesian_network.visualizer import NetworkVisualizer


@pytest.fixture()
def viz(trained_analyzer: BayesianAnalyzer) -> NetworkVisualizer:
    return NetworkVisualizer(trained_analyzer)


class TestPlotNetwork:
    def test_returns_figure(self, viz: NetworkVisualizer) -> None:
        fig = viz.plot_network()
        assert isinstance(fig, go.Figure)

    def test_with_target(self, viz: NetworkVisualizer) -> None:
        fig = viz.plot_network(target_node="C")
        assert isinstance(fig, go.Figure)

    def test_html_export(self, viz: NetworkVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "net.html")
            viz.plot_network(html_path=path)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 100

    def test_untrained_raises(self, sample_df) -> None:
        a = BayesianAnalyzer(sample_df)
        v = NetworkVisualizer(a)
        with pytest.raises(RuntimeError, match="not trained"):
            v.plot_network()


class TestPlotSensitivity:
    def test_returns_figure(self, viz: NetworkVisualizer) -> None:
        fig = viz.plot_sensitivity("C")
        assert isinstance(fig, go.Figure)

    def test_top_k(self, viz: NetworkVisualizer) -> None:
        fig = viz.plot_sensitivity("C", top_k=2)
        assert isinstance(fig, go.Figure)

    def test_html_export(self, viz: NetworkVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sens.html")
            viz.plot_sensitivity("C", html_path=path)
            assert Path(path).exists()


class TestPlotCPD:
    def test_returns_figure(self, viz: NetworkVisualizer, trained_analyzer: BayesianAnalyzer) -> None:
        edges = trained_analyzer.edges
        if edges:
            node = edges[0][1]
            fig = viz.plot_cpd_heatmap(node)
            assert isinstance(fig, go.Figure)

    def test_html_export(self, viz: NetworkVisualizer, trained_analyzer: BayesianAnalyzer) -> None:
        edges = trained_analyzer.edges
        if edges:
            node = edges[0][1]
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "cpd.html")
                viz.plot_cpd_heatmap(node, html_path=path)
                assert Path(path).exists()


class TestPlotDashboard:
    def test_returns_figure(self, viz: NetworkVisualizer) -> None:
        fig = viz.plot_dashboard("C")
        assert isinstance(fig, go.Figure)

    def test_html_export(self, viz: NetworkVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "dash.html")
            viz.plot_dashboard("C", html_path=path)
            assert Path(path).exists()
