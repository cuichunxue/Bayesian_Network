"""Shared chart-building functions for Bayesian Network visualisation.

This module is the single source of truth for all Plotly chart construction.
``visualizer.py``, ``interactive.py``, and ``webapp.py`` all delegate here
to avoid code duplication.
"""

from __future__ import annotations

import html as html_mod
import itertools
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (single source of truth)
# ---------------------------------------------------------------------------
PALETTE = {
    "edge": "rgba(150, 160, 175, 0.55)",
    "edge_highlight": "rgba(55, 90, 180, 0.80)",
    "node_border": "rgba(50, 60, 80, 0.9)",
    "target_border": "rgba(220, 50, 50, 0.95)",
    "evidence_border": "rgba(34, 139, 34, 0.95)",
    "bg": "#FAFBFC",
    "grid": "rgba(0,0,0,0.04)",
    "bar_main": "#4C78A8",
    "bar_highlight": "#E45756",
    "bar_low": "#72B7B2",
    "text": "#2E3440",
    "text_sub": "#6B7280",
    "heatmap": "Blues",
    "success": "#54A24B",
    "warning": "#EECA3B",
    "muted": "rgba(180, 180, 190, 0.7)",
}

_DISTRIBUTION_PALETTE = [
    "#4C78A8", "#F58518", "#E45756", "#72B7B2",
    "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(text: Any) -> str:
    """HTML-escape arbitrary text for safe embedding in hover labels."""
    return html_mod.escape(str(text))


def prob_color(p: float) -> str:
    """Map probability [0, 1] to a blue → white → red colour string."""
    if p < 0.5:
        t = p / 0.5
        r = int(76 + t * 144)
        g = int(120 + t * 100)
        b = int(200 - t * 32)
    else:
        t = (p - 0.5) / 0.5
        r = int(220 + t * 35)
        g = int(220 - t * 140)
        b = int(168 - t * 100)
    return f"rgba({r},{g},{b},0.88)"


def mi_node_color(t: float) -> str:
    """Map normalised MI [0, 1] to a light-blue → vivid-blue gradient.

    Higher MI → more saturated / brighter blue (not darker).
    """
    r = int(180 - t * 110)   # 180 → 70
    g = int(200 - t * 80)    # 200 → 120
    b = int(220 + t * 35)    # 220 → 255
    return f"rgba({r}, {g}, {b}, 0.85)"


def bar_colors(values: Sequence[float]) -> List[str]:
    """Assign bar colours by relative magnitude."""
    mx = max(values) if values else 1.0
    return [
        PALETTE["bar_highlight"] if v >= mx * 0.8
        else PALETTE["bar_main"] if v >= mx * 0.3
        else PALETTE["bar_low"]
        for v in values
    ]


def safe_write_html(fig: go.Figure, path: str) -> None:
    """Write a Plotly figure to an HTML file, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(p), include_plotlyjs="cdn")
    logger.info("Saved HTML: %s", p)


def fig_to_div(fig: go.Figure) -> str:
    """Convert a Plotly figure to an embeddable HTML ``<div>``."""
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


# ---------------------------------------------------------------------------
# Layout computation (does NOT mutate the original graph)
# ---------------------------------------------------------------------------

def compute_layout(
    g: nx.DiGraph,
    layout_k: float = 1.8,
    layout_seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute node positions; prefers hierarchical layout for DAGs.

    Unlike earlier implementations this function does **not** mutate *g*;
    it works on an internal copy when setting ``subset`` attributes.
    """
    if g.number_of_nodes() == 0:
        return {}

    if nx.is_directed_acyclic_graph(g) and g.number_of_edges() > 0:
        try:
            work = g.copy()
            for layer, nodes in enumerate(nx.topological_generations(work)):
                for node in nodes:
                    work.nodes[node]["subset"] = layer
            pos = nx.multipartite_layout(work, subset_key="subset", align="horizontal")
            return {n: (y, -x) for n, (x, y) in pos.items()}
        except Exception:
            pass

    return nx.spring_layout(g, k=layout_k, seed=layout_seed, iterations=80)


# ---------------------------------------------------------------------------
# Arrow annotations (shared by all network figures)
# ---------------------------------------------------------------------------

def _arrow_annotations(
    g: nx.DiGraph,
    pos: Dict[str, Tuple[float, float]],
    target_node: Optional[str] = None,
    shrink: float = 0.06,
) -> List[Dict[str, Any]]:
    """Build Plotly arrow annotations for directed edges."""
    annotations: List[Dict[str, Any]] = []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist > 0:
            x1 -= dx / dist * shrink
            y1 -= dy / dist * shrink

        is_hl = target_node is not None and (u == target_node or v == target_node)
        annotations.append(dict(
            ax=x0, ay=y0, x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=2.0 if is_hl else 1.2,
            arrowcolor=PALETTE["edge_highlight"] if is_hl else PALETTE["edge"],
            opacity=0.9 if is_hl else 0.6,
        ))
    return annotations


# =========================================================================
# 1. Network graph
# =========================================================================

def build_network_figure(
    columns: List[str],
    edges: List[Tuple[str, str]],
    layout_k: float = 1.8,
    layout_seed: int = 42,
    target_node: Optional[str] = None,
    evidence: Optional[Dict[str, str]] = None,
    posteriors: Optional[Dict[str, Dict[str, float]]] = None,
    mi_map: Optional[Dict[str, float]] = None,
    node_size_min: float = 18.0,
    node_size_max: float = 45.0,
    default_node_size: float = 34.0,
    width: int = 1000,
    height: int = 700,
    title: Optional[str] = None,
    show_title: bool = True,
) -> go.Figure:
    """Build a directed network graph with configurable node colouring.

    Supports two colouring modes (selected automatically):
    * **Posterior mode** (when *posteriors* is provided): nodes coloured by
      posterior probability; evidence nodes shown in green.
    * **MI mode** (when *mi_map* is provided): nodes sized and coloured by
      mutual-information score.

    All user-supplied text is HTML-escaped before embedding.
    """
    evidence = evidence or {}
    posteriors = posteriors or {}
    mi_map = mi_map or {}

    g = nx.DiGraph()
    g.add_nodes_from(columns)
    g.add_edges_from(edges)
    pos = compute_layout(g, layout_k, layout_seed)

    fig = go.Figure()

    # --- Edge lines ---
    if posteriors:
        # Simple edge lines (posterior mode)
        ex, ey = [], []
        for u, v in g.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(width=1.3, color=PALETTE["edge"]),
            hoverinfo="none", showlegend=False,
        ))
    else:
        # Highlight target-connected edges (MI mode)
        normal_x, normal_y = [], []
        hl_x, hl_y = [], []
        for u, v in g.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            seg = ([x0, x1, None], [y0, y1, None])
            if target_node and (u == target_node or v == target_node):
                hl_x += seg[0]; hl_y += seg[1]
            else:
                normal_x += seg[0]; normal_y += seg[1]
        if normal_x:
            fig.add_trace(go.Scatter(
                x=normal_x, y=normal_y, mode="lines",
                line=dict(width=1.0, color=PALETTE["edge"]),
                hoverinfo="none", showlegend=False,
            ))
        if hl_x:
            fig.add_trace(go.Scatter(
                x=hl_x, y=hl_y, mode="lines",
                line=dict(width=2.0, color=PALETTE["edge_highlight"]),
                hoverinfo="none", showlegend=False,
            ))

    # --- Arrow annotations ---
    annotations = _arrow_annotations(g, pos, target_node)

    # --- Nodes ---
    nodes = list(g.nodes())
    nx_arr = [pos[n][0] for n in nodes]
    ny_arr = [pos[n][1] for n in nodes]
    sizes, colours, borders, border_w, hovers = [], [], [], [], []

    for n in nodes:
        parts = [f"<b>{_esc(n)}</b>"]

        # -- Size --
        if mi_map:
            raw_vals = np.array([mi_map.get(nd, 0.0) for nd in nodes], dtype=float)
            raw_max = float(raw_vals.max()) if raw_vals.size else 0.0
            if raw_max > 0:
                normed = mi_map.get(n, 0.0) / raw_max
                sizes.append(node_size_min + normed * (node_size_max - node_size_min))
            else:
                sizes.append(default_node_size)
        else:
            sizes.append(default_node_size)

        # -- Colour --
        if n in evidence:
            colours.append("rgba(34,139,34,0.85)")
            parts.append(f"Evidence: {_esc(evidence[n])}")
        elif posteriors and n in posteriors:
            post = posteriors[n]
            colours.append(prob_color(max(post.values())))
        elif mi_map:
            raw_vals = np.array([mi_map.get(nd, 0.0) for nd in nodes], dtype=float)
            raw_max = float(raw_vals.max()) if raw_vals.size else 0.0
            if n == target_node:
                colours.append("rgba(220, 50, 50, 0.90)")
            elif raw_max > 0:
                colours.append(mi_node_color(mi_map.get(n, 0.0) / raw_max))
            else:
                colours.append(PALETTE["bar_main"])
        else:
            colours.append(PALETTE["muted"])

        # -- Border --
        if n == target_node:
            borders.append(PALETTE["target_border"]); border_w.append(3.0)
            parts.append("(TARGET)")
        elif n in evidence:
            borders.append(PALETTE["evidence_border"]); border_w.append(2.5)
        else:
            borders.append(PALETTE["node_border"]); border_w.append(1.5)

        # -- Hover extras --
        if mi_map and n in mi_map:
            parts.append(f"MI = {mi_map[n]:.6f}")
        if posteriors and n in posteriors:
            for st, pr in sorted(posteriors[n].items(), key=lambda x: -x[1]):
                parts.append(f"  {_esc(st)}: {pr:.1%}")

        hovers.append("<br>".join(parts))

    fig.add_trace(go.Scatter(
        x=nx_arr, y=ny_arr, mode="markers+text",
        text=[_esc(n) for n in nodes],
        textposition="top center",
        textfont=dict(size=11, color=PALETTE["text"]),
        hovertext=hovers, hoverinfo="text",
        marker=dict(size=sizes, color=colours,
                    line=dict(width=border_w, color=borders)),
        showlegend=False,
    ))

    # --- Layout ---
    auto_title = title
    if auto_title is None and show_title:
        auto_title = "Bayesian Network Structure"
        if not edges:
            auto_title += "  (empty graph — no edges learned)"
        elif target_node:
            auto_title += f"  — target: {target_node}"

    layout_kwargs: Dict[str, Any] = dict(
        annotations=annotations,
        template="plotly_white",
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        dragmode="pan",
        showlegend=False,
    )
    if show_title and auto_title:
        layout_kwargs["title"] = dict(
            text=auto_title, font=dict(size=16, color=PALETTE["text"]),
        )
    if width:
        layout_kwargs["width"] = width
    if height:
        layout_kwargs["height"] = height
    layout_kwargs["margin"] = dict(l=20, r=20, t=65 if show_title else 10, b=20)

    fig.update_layout(**layout_kwargs)
    return fig


# =========================================================================
# 2. Sensitivity bar chart
# =========================================================================

def build_sensitivity_figure(
    mi_scores: Dict[str, float],
    target_node: str,
    top_k: int = 20,
    width: int = 900,
    height: int = 500,
    show_title: bool = True,
) -> go.Figure:
    """Horizontal bar chart of mutual-information scores."""
    items = list(mi_scores.items())[:top_k]
    if not items:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(
                text="No sensitivity data available",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color=PALETTE["text_sub"]),
            )],
            height=200,
        )
        return fig

    names = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    colours = bar_colors(values)

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=colours, line=dict(width=0.5, color="rgba(0,0,0,0.15)")),
        hovertemplate="%{y}: MI = %{x:.6f}<extra></extra>",
    ))

    layout_kwargs: Dict[str, Any] = dict(
        template="plotly_white",
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor="white",
        xaxis=dict(title="Mutual Information", gridcolor=PALETTE["grid"],
                   zeroline=True, zerolinecolor=PALETTE["grid"]),
        yaxis=dict(title=""),
        margin=dict(l=140, r=30, t=65 if show_title else 30, b=50),
        width=width,
        height=max(height, len(names) * 28 + 120),
    )
    if show_title:
        layout_kwargs["title"] = dict(
            text=f"Sensitivity Analysis (MI) — target: {_esc(target_node)}",
            font=dict(size=15, color=PALETTE["text"]),
        )
    fig.update_layout(**layout_kwargs)
    return fig


# =========================================================================
# 3. CPD heatmap
# =========================================================================

def build_cpd_heatmap_figure(
    cpd: Any,
    node: str,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """Heatmap of the conditional probability distribution of *node*."""
    values = cpd.get_values()
    state_names = cpd.state_names
    var = cpd.variable

    row_labels = [str(s) for s in state_names[var]]

    parents = cpd.variables[1:]
    if parents:
        parent_states = [state_names[p] for p in parents]
        combos = list(itertools.product(*parent_states))
        col_labels = [
            " | ".join(f"{p}={s}" for p, s in zip(parents, combo))
            for combo in combos
        ]
    else:
        col_labels = ["(no parents)"]

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    fig = go.Figure(go.Heatmap(
        z=values, x=col_labels, y=row_labels,
        colorscale=PALETTE["heatmap"],
        hovertemplate="P(%{y} | %{x}) = %{z:.4f}<extra></extra>",
        colorbar=dict(title="P", thickness=15),
    ))
    fig.update_layout(
        title=dict(
            text=f"CPD: P({_esc(node)} | parents)",
            font=dict(size=15, color=PALETTE["text"]),
        ),
        template="plotly_white",
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor="white",
        xaxis=dict(title="Parent States", tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(title=f"{node} States", autorange="reversed"),
        margin=dict(l=80, r=30, t=65, b=120),
        width=width,
        height=max(height, len(row_labels) * 40 + 150),
    )
    return fig


# =========================================================================
# 4. Posterior grouped bars
# =========================================================================

def build_posterior_figure(
    posteriors: Dict[str, Dict[str, float]],
    target_node: str,
) -> go.Figure:
    """Grouped bar chart of posterior probabilities for all variables."""
    if not posteriors:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(
                text="No posterior data available",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color=PALETTE["text_sub"]),
            )],
            height=200,
        )
        return fig

    ordered: List[str] = []
    if target_node in posteriors:
        ordered.append(target_node)
    ordered += sorted(k for k in posteriors if k != target_node)

    fig = make_subplots(
        rows=len(ordered), cols=1,
        subplot_titles=ordered,
        vertical_spacing=max(0.02, 0.3 / max(len(ordered), 1)),
        shared_xaxes=False,
    )

    for i, var in enumerate(ordered, 1):
        probs = posteriors[var]
        states = sorted(probs.keys())
        vals = [probs[s] for s in states]
        color = PALETTE["bar_highlight"] if var == target_node else PALETTE["bar_main"]
        fig.add_trace(go.Bar(
            x=states, y=vals, name=var,
            marker=dict(color=color, line=dict(width=0.5, color="rgba(0,0,0,0.1)")),
            hovertemplate="%{x}: %{y:.1%}<extra>" + _esc(var) + "</extra>",
            showlegend=False,
        ), row=i, col=1)
        fig.update_yaxes(range=[0, 1], tickformat=".0%", row=i, col=1)

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor="white",
        margin=dict(l=50, r=20, t=40, b=20),
        height=max(200, len(ordered) * 160),
    )
    return fig


# =========================================================================
# 5. Data distribution
# =========================================================================

def build_data_distribution_figure(df: "pd.DataFrame") -> go.Figure:
    """Value-distribution bar charts for every column."""
    import pandas as pd

    cols = list(df.columns)
    n = len(cols)
    if n == 0:
        return go.Figure()

    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=cols,
        vertical_spacing=max(0.03, 0.4 / max(n_rows, 1)),
        horizontal_spacing=0.08,
    )

    for idx, col in enumerate(cols):
        r = idx // n_cols + 1
        c = idx % n_cols + 1
        vc = df[col].astype(str).value_counts()
        fig.add_trace(go.Bar(
            x=vc.index.tolist(), y=vc.values.tolist(),
            marker=dict(color=_DISTRIBUTION_PALETTE[idx % len(_DISTRIBUTION_PALETTE)]),
            hovertemplate="%{x}: %{y}<extra>" + _esc(col) + "</extra>",
            showlegend=False,
        ), row=r, col=c)

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=30),
        height=max(300, n_rows * 250),
    )
    return fig


# =========================================================================
# 6. MI matrix heatmap
# =========================================================================

def build_mi_matrix_figure(
    columns: List[str],
    mi_func: Any,
) -> go.Figure:
    """Mutual-information matrix heatmap.

    Parameters
    ----------
    columns : list[str]
        Column names.
    mi_func : callable(col_i, col_j) -> float
        Function returning MI between two column names.
    """
    n = len(columns)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            mi = mi_func(columns[i], columns[j])
            matrix[i][j] = mi
            matrix[j][i] = mi

    fig = go.Figure(go.Heatmap(
        z=matrix, x=columns, y=columns,
        colorscale="Viridis",
        hovertemplate="MI(%{x}, %{y}) = %{z:.4f}<extra></extra>",
        colorbar=dict(title="MI", thickness=12),
    ))
    fig.update_layout(
        title=dict(text="Mutual Information Matrix", font=dict(size=14)),
        template="plotly_white",
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor="white",
        margin=dict(l=80, r=20, t=50, b=80),
        height=max(400, n * 40 + 100),
        width=max(500, n * 40 + 150),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
    )
    return fig


# =========================================================================
# 7. Edge strength
# =========================================================================

def build_edge_strength_figure(
    edge_labels: List[str],
    mi_vals: List[float],
) -> go.Figure:
    """Bar chart of MI for each learned edge."""
    if not edge_labels:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(
                text="No edges learned",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color=PALETTE["text_sub"]),
            )],
            height=200,
        )
        return fig

    pairs = sorted(zip(edge_labels, mi_vals), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in pairs][::-1]
    vals = [p[1] for p in pairs][::-1]
    colours = bar_colors(vals)

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colours, line=dict(width=0.5, color="rgba(0,0,0,0.1)")),
        hovertemplate="%{y}: MI = %{x:.6f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Edge Strength (MI)", font=dict(size=14)),
        template="plotly_white",
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor="white",
        margin=dict(l=160, r=15, t=50, b=40),
        xaxis=dict(title="Mutual Information", gridcolor=PALETTE["grid"]),
        yaxis=dict(title=""),
        height=max(300, len(labels) * 26 + 100),
    )
    return fig


# =========================================================================
# 8. Markov blanket subgraph
# =========================================================================

def build_markov_blanket_figure(
    target_node: str,
    mb_nodes: set,
    edges: List[Tuple[str, str]],
    layout_k: float = 1.8,
    layout_seed: int = 42,
) -> go.Figure:
    """Subgraph showing only the Markov blanket of the target node."""
    if not mb_nodes:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=f"Markov Blanket of {_esc(target_node)}: empty",
                       font=dict(size=14)),
            height=200,
        )
        return fig

    all_nodes = mb_nodes | {target_node}
    sub_edges = [(u, v) for u, v in edges if u in all_nodes and v in all_nodes]

    g = nx.DiGraph()
    g.add_nodes_from(sorted(all_nodes))
    g.add_edges_from(sub_edges)
    pos = compute_layout(g, layout_k, layout_seed)

    fig = go.Figure()

    # Edges
    ex, ey = [], []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ex += [x0, x1, None]
        ey += [y0, y1, None]
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=1.5, color=PALETTE["edge"]),
        hoverinfo="none", showlegend=False,
    ))

    annotations = _arrow_annotations(g, pos, target_node, shrink=0.08)

    nodes = sorted(g.nodes())
    nx_arr = [pos[n][0] for n in nodes]
    ny_arr = [pos[n][1] for n in nodes]
    colors = [PALETTE["bar_highlight"] if n == target_node else PALETTE["bar_main"]
              for n in nodes]
    bw = [3.0 if n == target_node else 1.5 for n in nodes]

    fig.add_trace(go.Scatter(
        x=nx_arr, y=ny_arr, mode="markers+text",
        text=[_esc(n) for n in nodes],
        textposition="top center",
        textfont=dict(size=12, color=PALETTE["text"]),
        hovertext=[
            f"<b>{_esc(n)}</b>" + (" (TARGET)" if n == target_node else " (blanket)")
            for n in nodes
        ],
        hoverinfo="text",
        marker=dict(size=40, color=colors,
                    line=dict(width=bw, color=PALETTE["node_border"])),
        showlegend=False,
    ))

    fig.update_layout(
        annotations=annotations,
        title=dict(
            text=f"Markov Blanket of {_esc(target_node)} ({len(mb_nodes)} nodes)",
            font=dict(size=14),
        ),
        template="plotly_white",
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        dragmode="pan",
        height=400,
    )
    return fig


# =========================================================================
# 9. Dashboard (two-panel: network + sensitivity)
# =========================================================================

def build_dashboard_figure(
    columns: List[str],
    edges: List[Tuple[str, str]],
    mi_scores: Dict[str, float],
    target_node: str,
    layout_k: float = 1.8,
    layout_seed: int = 42,
    top_k: int = 15,
    node_size_min: float = 18.0,
    node_size_max: float = 45.0,
    default_node_size: float = 28.0,
    width: int = 1400,
    height: int = 750,
) -> go.Figure:
    """Two-panel dashboard: network graph (left) + sensitivity bars (right)."""
    items = list(mi_scores.items())[:top_k]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.58, 0.42],
        subplot_titles=[
            f"Network Structure — target: {target_node}",
            f"Sensitivity (MI) — top {min(top_k, len(items))}",
        ],
        horizontal_spacing=0.08,
    )

    # --- Left: network ---
    g = nx.DiGraph()
    g.add_nodes_from(columns)
    g.add_edges_from(edges)
    pos = compute_layout(g, layout_k, layout_seed)

    mi_map = dict(mi_scores)
    max_mi = max(mi_map.values()) if mi_map else 0.0
    mi_map[target_node] = max_mi * 1.2 if max_mi > 0 else 1.0

    # edges
    edge_x, edge_y = [], []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.0, color=PALETTE["edge"]),
        hoverinfo="none", showlegend=False,
    ), row=1, col=1)

    # nodes
    nodes = list(g.nodes())
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]

    raw = np.array([mi_map.get(n, 0.0) for n in nodes], dtype=float)
    raw_max = float(raw.max()) if raw.size else 0.0
    if raw_max > 0:
        normed = raw / raw_max
        sizes = (node_size_min + normed * (node_size_max - node_size_min)).tolist()
    else:
        sizes = [default_node_size] * len(nodes)

    colours: List[str] = []
    for i, n in enumerate(nodes):
        if n == target_node:
            colours.append("rgba(220, 50, 50, 0.90)")
        elif raw_max > 0:
            colours.append(mi_node_color(normed[i]))
        else:
            colours.append(PALETTE["bar_main"])

    hovers = []
    for n in nodes:
        parts = [f"<b>{_esc(n)}</b>"]
        if n in mi_map:
            parts.append(f"MI = {mi_map[n]:.6f}")
        if n == target_node:
            parts.append("(TARGET)")
        hovers.append("<br>".join(parts))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[_esc(n) for n in nodes],
        textposition="top center",
        textfont=dict(size=10, color=PALETTE["text"]),
        hovertext=hovers, hoverinfo="text",
        marker=dict(size=sizes, color=colours,
                    line=dict(width=1.5, color=PALETTE["node_border"])),
        showlegend=False,
    ), row=1, col=1)

    # --- Right: bars ---
    names = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    bar_c = bar_colors(values)

    fig.add_trace(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=bar_c, line=dict(width=0.5, color="rgba(0,0,0,0.12)")),
        hovertemplate="%{y}: MI = %{x:.6f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor="white",
        margin=dict(l=20, r=30, t=70, b=30),
        width=width,
        height=max(height, len(names) * 24 + 200),
        title=dict(
            text="Bayesian Network — Analysis Dashboard",
            font=dict(size=17, color=PALETTE["text"]),
        ),
    )

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="Mutual Information", gridcolor=PALETTE["grid"], row=1, col=2)
    fig.update_yaxes(title_text="", row=1, col=2)

    return fig
