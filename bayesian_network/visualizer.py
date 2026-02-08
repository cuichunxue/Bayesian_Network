"""Interactive visualisation for Bayesian Networks.

Provides network graph, sensitivity bar chart, CPD heatmap, and
a combined dashboard — all built with Plotly for interactivity and
HTML export.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bayesian_network.config import BNConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
_PALETTE = {
    "edge": "rgba(150, 160, 175, 0.55)",
    "edge_highlight": "rgba(55, 90, 180, 0.80)",
    "node_border": "rgba(50, 60, 80, 0.9)",
    "target_border": "rgba(220, 50, 50, 0.95)",
    "bg": "#FAFBFC",
    "grid": "rgba(0,0,0,0.04)",
    "bar_main": "#4C78A8",
    "bar_highlight": "#E45756",
    "bar_low": "#72B7B2",
    "text": "#2E3440",
    "text_sub": "#6B7280",
    "heatmap": "Blues",
}


def _safe_write_html(fig: go.Figure, path: str) -> None:
    """Write figure to HTML, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(p), include_plotlyjs="cdn")
    logger.info("Saved HTML: %s", p)


# =========================================================================
# Public API
# =========================================================================
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

        g = nx.DiGraph()
        g.add_nodes_from(a.columns)
        g.add_edges_from(a.edges)

        if g.number_of_nodes() == 0:
            raise RuntimeError("No nodes to visualise.")

        pos = self._compute_layout(g)

        # MI mapping
        mi_map: Dict[str, float] = {}
        if show_mi and target_node and target_node in a.columns:
            mi_map = a.compute_sensitivity(target_node)
            max_mi = max(mi_map.values()) if mi_map else 0.0
            mi_map[target_node] = max_mi * 1.2 if max_mi > 0 else 1.0

        fig = go.Figure()

        # --- Edges ---
        self._add_edge_traces(fig, g, pos, mi_map, target_node)

        # --- Nodes ---
        self._add_node_trace(fig, g, pos, mi_map, target_node)

        # --- Arrow annotations ---
        annotations = self._build_arrow_annotations(g, pos, mi_map, target_node)

        title = "Bayesian Network Structure"
        if not a.edges:
            title += "  (empty graph — no edges learned)"
        elif target_node:
            title += f"  — target: {target_node}"

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color=_PALETTE["text"])),
            showlegend=False,
            annotations=annotations,
            template="plotly_white",
            plot_bgcolor=_PALETTE["bg"],
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=65, b=20),
            width=width,
            height=height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        if html_path:
            _safe_write_html(fig, html_path)

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
        items = list(mi.items())[:top_k]

        if not items:
            logger.warning("No sensitivity data to plot.")
            return go.Figure()

        names = [k for k, _ in items][::-1]
        values = [v for _, v in items][::-1]

        max_val = max(values) if values else 1.0
        colours = [
            _PALETTE["bar_highlight"] if v >= max_val * 0.8
            else _PALETTE["bar_main"] if v >= max_val * 0.3
            else _PALETTE["bar_low"]
            for v in values
        ]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker=dict(
                    color=colours,
                    line=dict(width=0.5, color="rgba(0,0,0,0.15)"),
                ),
                hovertemplate="%{y}: MI = %{x:.6f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Sensitivity Analysis (MI) — target: {target_node}",
                font=dict(size=15, color=_PALETTE["text"]),
            ),
            template="plotly_white",
            plot_bgcolor=_PALETTE["bg"],
            paper_bgcolor="white",
            xaxis=dict(
                title="Mutual Information",
                gridcolor=_PALETTE["grid"],
                zeroline=True,
                zerolinecolor=_PALETTE["grid"],
            ),
            yaxis=dict(title=""),
            margin=dict(l=140, r=30, t=65, b=50),
            width=width,
            height=max(height, len(names) * 28 + 120),
        )

        if html_path:
            _safe_write_html(fig, html_path)

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

        model = a.model
        cpd = model.get_cpds(node)

        values = cpd.get_values()
        state_names = cpd.state_names
        var = cpd.variable

        row_labels = [str(s) for s in state_names[var]]

        # Build column labels from parent combinations
        parents = cpd.variables[1:]
        if parents:
            import itertools
            parent_states = [state_names[p] for p in parents]
            combos = list(itertools.product(*parent_states))
            col_labels = [
                " | ".join(f"{p}={s}" for p, s in zip(parents, combo))
                for combo in combos
            ]
        else:
            col_labels = ["(no parents)"]

        # Ensure shape matches
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        fig = go.Figure(
            go.Heatmap(
                z=values,
                x=col_labels,
                y=row_labels,
                colorscale=_PALETTE["heatmap"],
                hovertemplate=(
                    "P(%{y} | %{x}) = %{z:.4f}<extra></extra>"
                ),
                colorbar=dict(title="P", thickness=15),
            )
        )

        fig.update_layout(
            title=dict(
                text=f"CPD: P({node} | parents)",
                font=dict(size=15, color=_PALETTE["text"]),
            ),
            template="plotly_white",
            plot_bgcolor=_PALETTE["bg"],
            paper_bgcolor="white",
            xaxis=dict(title="Parent States", tickangle=-45),
            yaxis=dict(title=f"{node} States", autorange="reversed"),
            margin=dict(l=80, r=30, t=65, b=120),
            width=width,
            height=height,
        )

        if html_path:
            _safe_write_html(fig, html_path)

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
        items = list(mi.items())[:top_k]

        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.58, 0.42],
            subplot_titles=[
                f"Network Structure — target: {target_node}",
                f"Sensitivity (MI) — top {min(top_k, len(items))}",
            ],
            horizontal_spacing=0.08,
        )

        # ---- Left: network ----
        g = nx.DiGraph()
        g.add_nodes_from(a.columns)
        g.add_edges_from(a.edges)
        pos = self._compute_layout(g)

        mi_map = dict(mi)
        max_mi = max(mi_map.values()) if mi_map else 0.0
        mi_map[target_node] = max_mi * 1.2 if max_mi > 0 else 1.0

        # edges
        edge_x, edge_y = [], []
        for u, v in g.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(width=1.0, color=_PALETTE["edge"]),
                hoverinfo="none", showlegend=False,
            ),
            row=1, col=1,
        )

        # nodes
        nodes = list(g.nodes())
        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        sizes, colours, hovers = self._node_visual_attrs(nodes, mi_map, target_node)

        fig.add_trace(
            go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=nodes,
                textposition="top center",
                textfont=dict(size=10, color=_PALETTE["text"]),
                hovertext=hovers,
                hoverinfo="text",
                marker=dict(
                    size=sizes,
                    color=colours,
                    line=dict(width=1.5, color=_PALETTE["node_border"]),
                ),
                showlegend=False,
            ),
            row=1, col=1,
        )

        # ---- Right: bars ----
        names = [k for k, _ in items][::-1]
        values = [v for _, v in items][::-1]
        max_val = max(values) if values else 1.0
        bar_colours = [
            _PALETTE["bar_highlight"] if v >= max_val * 0.8
            else _PALETTE["bar_main"] if v >= max_val * 0.3
            else _PALETTE["bar_low"]
            for v in values
        ]

        fig.add_trace(
            go.Bar(
                x=values, y=names,
                orientation="h",
                marker=dict(color=bar_colours, line=dict(width=0.5, color="rgba(0,0,0,0.12)")),
                hovertemplate="%{y}: MI = %{x:.6f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

        # Layout
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor=_PALETTE["bg"],
            paper_bgcolor="white",
            margin=dict(l=20, r=30, t=70, b=30),
            width=width,
            height=max(height, len(names) * 24 + 200),
            title=dict(
                text="Bayesian Network — Analysis Dashboard",
                font=dict(size=17, color=_PALETTE["text"]),
            ),
        )

        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        fig.update_xaxes(title_text="Mutual Information", gridcolor=_PALETTE["grid"], row=1, col=2)
        fig.update_yaxes(title_text="", row=1, col=2)

        if html_path:
            _safe_write_html(fig, html_path)

        return fig

    # =================================================================
    # Internal helpers
    # =================================================================
    def _compute_layout(self, g: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Compute node positions using the best available layout.

        For DAGs, tries a layered (hierarchical) layout first;
        falls back to spring layout.
        """
        if g.number_of_nodes() == 0:
            return {}

        # Prefer hierarchical layout for DAGs
        if nx.is_directed_acyclic_graph(g) and g.number_of_edges() > 0:
            try:
                # Multipartite layout based on topological generations
                for layer, nodes in enumerate(nx.topological_generations(g)):
                    for node in nodes:
                        g.nodes[node]["subset"] = layer
                pos = nx.multipartite_layout(g, subset_key="subset", align="horizontal")
                # Rotate 90 degrees so flow goes top-to-bottom
                pos = {n: (y, -x) for n, (x, y) in pos.items()}
                return pos
            except Exception:
                pass

        return nx.spring_layout(
            g,
            k=self._cfg.layout_k,
            seed=self._cfg.layout_seed,
            iterations=80,
        )

    def _node_visual_attrs(
        self,
        nodes: List[str],
        mi_map: Dict[str, float],
        target_node: Optional[str],
    ) -> Tuple[List[float], List[str], List[str]]:
        """Return (sizes, colours, hover_texts) for each node."""
        cfg = self._cfg

        if not mi_map:
            size = cfg.default_node_size
            return (
                [size] * len(nodes),
                [_PALETTE["bar_main"]] * len(nodes),
                [str(n) for n in nodes],
            )

        raw = np.array([mi_map.get(n, 0.0) for n in nodes], dtype=float)
        raw_max = float(raw.max()) if raw.size > 0 else 0.0

        # Sizes
        if raw_max > 0:
            normed = raw / raw_max
            sizes = (cfg.node_size_min + normed * (cfg.node_size_max - cfg.node_size_min)).tolist()
        else:
            sizes = [cfg.default_node_size] * len(nodes)

        # Colours — map MI to a continuous colour scale
        if raw_max > 0:
            normed_01 = raw / raw_max
        else:
            normed_01 = np.zeros(len(nodes))

        colours: list[str] = []
        for i, n in enumerate(nodes):
            if n == target_node:
                colours.append("rgba(220, 50, 50, 0.90)")
            else:
                t = normed_01[i]
                r = int(76 + t * (68 - 76))
                g_val = int(120 + t * (1 - 120))
                b = int(168 + t * (84 - 168))
                colours.append(f"rgba({r}, {g_val}, {b}, 0.85)")

        # Hover
        hovers: list[str] = []
        for n in nodes:
            parts = [f"<b>{n}</b>"]
            if n in mi_map:
                parts.append(f"MI = {mi_map[n]:.6f}")
            if n == target_node:
                parts.append("(TARGET)")
            hovers.append("<br>".join(parts))

        return sizes, colours, hovers

    def _add_edge_traces(
        self,
        fig: go.Figure,
        g: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]],
        mi_map: Dict[str, float],
        target_node: Optional[str],
    ) -> None:
        """Add edge lines with optional highlighting for target-connected edges."""
        normal_x, normal_y = [], []
        highlight_x, highlight_y = [], []

        for u, v in g.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            segment = [x0, x1, None], [y0, y1, None]

            if target_node and (u == target_node or v == target_node):
                highlight_x += segment[0]
                highlight_y += segment[1]
            else:
                normal_x += segment[0]
                normal_y += segment[1]

        if normal_x:
            fig.add_trace(go.Scatter(
                x=normal_x, y=normal_y,
                mode="lines",
                line=dict(width=1.0, color=_PALETTE["edge"]),
                hoverinfo="none", showlegend=False,
            ))
        if highlight_x:
            fig.add_trace(go.Scatter(
                x=highlight_x, y=highlight_y,
                mode="lines",
                line=dict(width=2.0, color=_PALETTE["edge_highlight"]),
                hoverinfo="none", showlegend=False,
            ))

    def _add_node_trace(
        self,
        fig: go.Figure,
        g: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]],
        mi_map: Dict[str, float],
        target_node: Optional[str],
    ) -> None:
        """Add a single scatter trace for all nodes."""
        nodes = list(g.nodes())
        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        sizes, colours, hovers = self._node_visual_attrs(nodes, mi_map, target_node)

        border_colours = [
            _PALETTE["target_border"] if n == target_node else _PALETTE["node_border"]
            for n in nodes
        ]
        border_widths = [2.5 if n == target_node else 1.5 for n in nodes]

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            textfont=dict(size=11, color=_PALETTE["text"]),
            hovertext=hovers,
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color=colours,
                line=dict(width=border_widths, color=border_colours),
            ),
            showlegend=False,
        ))

    def _build_arrow_annotations(
        self,
        g: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]],
        mi_map: Dict[str, float],
        target_node: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Create arrow annotations for directed edges."""
        annotations = []
        for u, v in g.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            # Shorten arrow to avoid overlapping with node marker
            dx, dy = x1 - x0, y1 - y0
            dist = math.hypot(dx, dy)
            if dist > 0:
                shrink = 0.06
                x1 -= dx / dist * shrink
                y1 -= dy / dist * shrink

            is_highlight = target_node and (u == target_node or v == target_node)

            annotations.append(dict(
                ax=x0, ay=y0,
                x=x1, y=y1,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=1.8 if is_highlight else 1.2,
                arrowcolor=_PALETTE["edge_highlight"] if is_highlight else _PALETTE["edge"],
                opacity=0.9 if is_highlight else 0.6,
            ))
        return annotations
