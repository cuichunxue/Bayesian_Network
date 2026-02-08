"""Interactive Dash-based dashboard for Bayesian Network exploration.

Provides a real-time interface for setting evidence, running inference,
and visualising posterior probabilities on a network graph.

Usage::

    from bayesian_network.interactive import run_server
    run_server(analyzer)  # opens http://localhost:8050
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html

from bayesian_network.analyzer import BayesianAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour constants (consistent with visualizer.py)
# ---------------------------------------------------------------------------
_C = {
    "edge": "rgba(150,160,175,0.55)",
    "edge_hl": "rgba(55,90,180,0.80)",
    "node_border": "rgba(50,60,80,0.9)",
    "evidence_border": "rgba(34,139,34,0.95)",
    "target_border": "rgba(220,50,50,0.95)",
    "bg": "#FAFBFC",
    "text": "#2E3440",
    "text_sub": "#6B7280",
    "bar_fill": "#4C78A8",
    "bar_hi": "#E45756",
    "bar_lo": "#72B7B2",
}

# Probability → colour mapping (blue-white-red diverging)
def _prob_color(p: float) -> str:
    """Map probability [0,1] to a blue→white→red colour string."""
    if p < 0.5:
        t = p / 0.5
        r, g, b = int(76 + t * (220 - 76)), int(120 + t * (220 - 120)), int(200 - t * 32)
    else:
        t = (p - 0.5) / 0.5
        r, g, b = int(220 + t * 35), int(220 - t * 140), int(168 - t * 100)
    return f"rgba({r},{g},{b},0.88)"


# =========================================================================
# Graph building helpers
# =========================================================================

def _build_network_figure(
    analyzer: BayesianAnalyzer,
    evidence: Dict[str, str],
    posteriors: Dict[str, Dict[str, float]],
    target_node: str,
    selected_node: Optional[str] = None,
    search_term: str = "",
) -> go.Figure:
    """Build a Plotly figure of the BN with posterior-coloured nodes."""
    g = nx.DiGraph()
    g.add_nodes_from(analyzer.columns)
    g.add_edges_from(analyzer.edges)

    pos = _compute_layout(g, analyzer.config)

    fig = go.Figure()

    # --- Edge traces ---
    edge_x, edge_y = [], []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.2, color=_C["edge"]),
        hoverinfo="none", showlegend=False,
    ))

    # --- Arrow annotations ---
    import math
    annotations = []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist > 0:
            shrink = 0.06
            x1 -= dx / dist * shrink
            y1 -= dy / dist * shrink
        annotations.append(dict(
            ax=x0, ay=y0, x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=1.2,
            arrowcolor=_C["edge"], opacity=0.6,
        ))

    # --- Node trace ---
    nodes = list(g.nodes())
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]

    # Determine max posterior probability for target for colour scaling
    sizes, colours, borders, border_w, hovers, customdata = [], [], [], [], [], []
    for n in nodes:
        # Size
        sizes.append(32)

        # Colour: based on max posterior probability
        post = posteriors.get(n)
        if n in evidence:
            colours.append("rgba(34,139,34,0.85)")
        elif post:
            max_p = max(post.values()) if post else 0.5
            colours.append(_prob_color(max_p))
        else:
            colours.append("rgba(180,180,190,0.7)")

        # Border
        if n == target_node:
            borders.append(_C["target_border"])
            border_w.append(3.0)
        elif n in evidence:
            borders.append(_C["evidence_border"])
            border_w.append(2.5)
        elif n == selected_node:
            borders.append("rgba(255,165,0,0.9)")
            border_w.append(2.5)
        else:
            borders.append(_C["node_border"])
            border_w.append(1.5)

        # Hover text
        parts = [f"<b>{n}</b>"]
        if n in evidence:
            parts.append(f"Evidence: {evidence[n]}")
        if n == target_node:
            parts.append("(TARGET)")
        if post:
            for state, prob in sorted(post.items(), key=lambda x: -x[1]):
                parts.append(f"  {state}: {prob:.1%}")
        hovers.append("<br>".join(parts))

        customdata.append(n)

    # Search highlighting: enlarge matching nodes
    if search_term:
        term_lower = search_term.lower()
        for i, n in enumerate(nodes):
            if term_lower in n.lower():
                sizes[i] = 44
                border_w[i] = 3.5
                borders[i] = "rgba(255,200,0,0.95)"

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=nodes,
        textposition="top center",
        textfont=dict(size=10, color=_C["text"]),
        hovertext=hovers,
        hoverinfo="text",
        customdata=customdata,
        marker=dict(
            size=sizes, color=colours,
            line=dict(width=border_w, color=borders),
        ),
        showlegend=False,
    ))

    fig.update_layout(
        annotations=annotations,
        template="plotly_white",
        plot_bgcolor=_C["bg"],
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        clickmode="event",
        dragmode="pan",
    )

    return fig


def _compute_layout(g: nx.DiGraph, cfg: Any) -> Dict[str, Tuple[float, float]]:
    """Compute hierarchical or spring layout for the graph."""
    if g.number_of_nodes() == 0:
        return {}
    if nx.is_directed_acyclic_graph(g) and g.number_of_edges() > 0:
        try:
            for layer, layer_nodes in enumerate(nx.topological_generations(g)):
                for node in layer_nodes:
                    g.nodes[node]["subset"] = layer
            pos = nx.multipartite_layout(g, subset_key="subset", align="horizontal")
            return {n: (y, -x) for n, (x, y) in pos.items()}
        except Exception:
            pass
    return nx.spring_layout(g, k=cfg.layout_k, seed=cfg.layout_seed, iterations=80)


def _build_sensitivity_figure(
    analyzer: BayesianAnalyzer,
    target_node: str,
    top_k: int = 15,
) -> go.Figure:
    """Build a horizontal bar chart of MI sensitivity scores."""
    try:
        mi = analyzer.compute_sensitivity(target_node)
    except KeyError:
        return go.Figure()

    items = list(mi.items())[:top_k]
    if not items:
        return go.Figure()

    names = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    max_val = max(values) if values else 1.0

    bar_colours = [
        _C["bar_hi"] if v >= max_val * 0.8
        else _C["bar_fill"] if v >= max_val * 0.3
        else _C["bar_lo"]
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=bar_colours, line=dict(width=0.5, color="rgba(0,0,0,0.12)")),
        hovertemplate="%{y}: MI = %{x:.6f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=_C["bg"], paper_bgcolor="white",
        margin=dict(l=100, r=15, t=30, b=30),
        xaxis=dict(title="Mutual Information", gridcolor="rgba(0,0,0,0.04)"),
        yaxis=dict(title=""),
        height=max(280, len(names) * 26 + 80),
    )
    return fig


def _build_posterior_bars(posteriors: Dict[str, Dict[str, float]], target_node: str) -> List:
    """Build dbc cards showing posterior probability bars for each variable."""
    cards = []
    # Show target first, then rest
    ordered = []
    if target_node in posteriors:
        ordered.append(target_node)
    ordered += [k for k in posteriors if k != target_node]

    for var in ordered:
        probs = posteriors[var]
        if not probs:
            continue
        sorted_states = sorted(probs.items(), key=lambda x: -x[1])
        bars = []
        for state, prob in sorted_states:
            pct = prob * 100
            bar_color = _C["bar_hi"] if prob >= 0.5 else _C["bar_fill"] if prob >= 0.2 else _C["bar_lo"]
            bars.append(
                html.Div([
                    html.Div([
                        html.Span(str(state), style={"fontSize": "12px", "minWidth": "80px", "display": "inline-block"}),
                        html.Span(f"{pct:.1f}%", style={"fontSize": "11px", "color": _C["text_sub"], "marginLeft": "4px"}),
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "2px"}),
                    dbc.Progress(
                        value=pct, max=100,
                        style={"height": "14px", "marginBottom": "4px"},
                        color="danger" if prob >= 0.5 else "primary" if prob >= 0.2 else "info",
                    ),
                ], style={"marginBottom": "2px"})
            )

        is_target = var == target_node
        card = dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.B(var),
                    html.Span(" (TARGET)", style={"color": _C["bar_hi"], "fontSize": "11px", "marginLeft": "6px"}) if is_target else "",
                ]),
                style={"padding": "6px 12px", "backgroundColor": "#FFF3F3" if is_target else "#F8F9FA"},
            ),
            dbc.CardBody(bars, style={"padding": "8px 12px"}),
        ], className="mb-2", style={"border": f"2px solid {_C['target_border']}" if is_target else "1px solid #DEE2E6"})
        cards.append(card)

    return cards


# =========================================================================
# Layout
# =========================================================================

def layout(analyzer: BayesianAnalyzer) -> dbc.Container:
    """Build the Dash HTML layout."""
    columns = analyzer.columns
    default_target = columns[-1] if columns else ""

    # Get possible states for each variable
    node_states: Dict[str, List[str]] = {}
    for col in columns:
        vals = sorted(analyzer.data[col].astype(str).unique().tolist())
        node_states[col] = vals

    return dbc.Container([
        # Hidden stores
        dcc.Store(id="evidence-store", data={}),
        dcc.Store(id="node-states-store", data=node_states),
        dcc.Store(id="selected-node-store", data=""),

        # ---- Header ----
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("Bayesian Network Analyzer", className="fw-bold fs-5"),
                dbc.Row([
                    dbc.Col(html.Label("Target:", className="text-white me-2 mt-1"), width="auto"),
                    dbc.Col(dcc.Dropdown(
                        id="target-dropdown",
                        options=[{"label": c, "value": c} for c in columns],
                        value=default_target,
                        clearable=False,
                        searchable=True,
                        style={"width": "180px"},
                    ), width="auto"),
                    dbc.Col(dbc.Button(
                        "Reset", id="reset-btn", color="light", size="sm", className="ms-2",
                    ), width="auto"),
                ], align="center", className="g-2"),
            ], fluid=True),
            color="dark", dark=True, className="mb-3 py-2",
        ),

        # ---- Main content ----
        dbc.Row([
            # Left: Graph + Sensitivity
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id="network-graph",
                            config={"scrollZoom": True, "displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                            style={"height": "520px"},
                        ),
                    ], style={"padding": "4px"}),
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader(html.B("Sensitivity Analysis (MI)", style={"fontSize": "14px"})),
                    dbc.CardBody([
                        dcc.Graph(id="sensitivity-graph", style={"height": "280px"}),
                    ], style={"padding": "4px"}),
                ]),
            ], lg=7, md=12),

            # Right: Controls + Posteriors
            dbc.Col([
                # Search
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="bi bi-search") if False else "Search"),
                    dbc.Input(id="node-search", placeholder="Search nodes...", type="text", size="sm"),
                ], className="mb-3", size="sm"),

                # Evidence section
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Row([
                            dbc.Col(html.B(id="evidence-section-title", children="Evidence (0)"), width="auto"),
                            dbc.Col(
                                dbc.Button("Clear All", id="clear-evidence-btn", color="outline-secondary", size="sm"),
                                width="auto", className="ms-auto",
                            ),
                        ], align="center"),
                    ),
                    dbc.CardBody(
                        id="evidence-panel",
                        style={"maxHeight": "340px", "overflowY": "auto", "padding": "10px"},
                    ),
                ], className="mb-3"),

                # Posterior results
                dbc.Card([
                    dbc.CardHeader(html.B("Posterior Distributions", style={"fontSize": "14px"})),
                    dbc.CardBody(
                        id="posterior-panel",
                        style={"maxHeight": "450px", "overflowY": "auto", "padding": "10px"},
                    ),
                ]),
            ], lg=5, md=12),
        ]),

        # Footer
        html.Hr(),
        html.P(
            "Bayesian Network Interactive Dashboard — powered by pgmpy + Dash",
            className="text-center text-muted small",
        ),
    ], fluid=True, style={"maxWidth": "1600px"})


# =========================================================================
# Callbacks
# =========================================================================

def register_callbacks(app: dash.Dash, analyzer: BayesianAnalyzer) -> None:
    """Wire up all Dash callbacks."""

    columns = analyzer.columns
    node_states: Dict[str, List[str]] = {}
    for col in columns:
        vals = sorted(analyzer.data[col].astype(str).unique().tolist())
        node_states[col] = vals

    # ------------------------------------------------------------------
    # 1) Node click → update selected node
    # ------------------------------------------------------------------
    @app.callback(
        Output("selected-node-store", "data"),
        Input("network-graph", "clickData"),
        prevent_initial_call=True,
    )
    def on_node_click(click_data):
        if not click_data or "points" not in click_data:
            return dash.no_update
        point = click_data["points"][0]
        node_name = point.get("customdata")
        if node_name and node_name in columns:
            return node_name
        return dash.no_update

    # ------------------------------------------------------------------
    # 2) Build evidence panel dropdowns (grouped: set, blanket, other)
    # ------------------------------------------------------------------
    @app.callback(
        [Output("evidence-panel", "children"),
         Output("evidence-section-title", "children")],
        [Input("evidence-store", "data"),
         Input("selected-node-store", "data"),
         Input("target-dropdown", "value")],
    )
    def build_evidence_panel(evidence, selected_node, target):
        evidence = evidence or {}
        count = len(evidence)

        # Compute Markov blanket for selected or target
        focus = selected_node if selected_node else target
        try:
            blanket = analyzer.compute_markov_blanket(focus) if focus else set()
        except Exception:
            blanket = set()

        # Group nodes
        set_nodes = [c for c in columns if c in evidence]
        blanket_nodes = [c for c in columns if c in blanket and c not in evidence and c != target]
        other_nodes = [c for c in columns if c not in evidence and c not in blanket and c != target]

        def make_dropdown_row(node: str) -> dbc.Row:
            is_target = node == target
            options = [{"label": "(unset)", "value": ""}] + [
                {"label": s, "value": s} for s in node_states.get(node, [])
            ]
            current_val = evidence.get(node, "")
            label_parts = [html.Span(node, className="fw-semibold")]
            if is_target:
                label_parts.append(html.Span(" [target]", style={"color": _C["bar_hi"], "fontSize": "11px"}))
            return dbc.Row([
                dbc.Col(html.Div(label_parts), width=5, className="d-flex align-items-center"),
                dbc.Col(
                    dcc.Dropdown(
                        id={"type": "evidence-dd", "node": node},
                        options=options,
                        value=current_val,
                        clearable=False,
                        searchable=True,
                        disabled=is_target,
                        style={"fontSize": "13px"},
                    ), width=7,
                ),
            ], className="mb-2 g-1")

        sections = []

        # Set evidence section
        if set_nodes:
            sections.append(html.Div([
                html.Div(
                    html.B(f"Set ({len(set_nodes)})", style={"fontSize": "12px", "color": "green"}),
                    className="mb-1",
                ),
                *[make_dropdown_row(n) for n in set_nodes],
            ], className="mb-3"))

        # Markov blanket section (collapsible)
        if blanket_nodes:
            sections.append(html.Div([
                html.Details([
                    html.Summary(html.B(
                        f"Markov Blanket of {focus} ({len(blanket_nodes)})",
                        style={"fontSize": "12px", "color": "#4C78A8"},
                    )),
                    html.Div(
                        [make_dropdown_row(n) for n in blanket_nodes],
                        style={"marginTop": "6px"},
                    ),
                ], open=True),
            ], className="mb-3"))

        # Target
        if target and target not in evidence:
            sections.append(html.Div([
                html.Div(
                    html.B("Query Target", style={"fontSize": "12px", "color": _C["bar_hi"]}),
                    className="mb-1",
                ),
                make_dropdown_row(target),
            ], className="mb-3"))

        # Other section (collapsed by default)
        if other_nodes:
            sections.append(html.Div([
                html.Details([
                    html.Summary(html.B(
                        f"Other ({len(other_nodes)})",
                        style={"fontSize": "12px", "color": _C["text_sub"]},
                    )),
                    html.Div(
                        [make_dropdown_row(n) for n in other_nodes],
                        style={"marginTop": "6px"},
                    ),
                ], open=False),
            ]))

        return sections, f"Evidence ({count})"

    # ------------------------------------------------------------------
    # 3) Dropdown value changed → update evidence store
    # ------------------------------------------------------------------
    @app.callback(
        Output("evidence-store", "data"),
        [Input({"type": "evidence-dd", "node": node}, "value") for node in columns],
        [State("evidence-store", "data"),
         State("target-dropdown", "value")],
        prevent_initial_call=True,
    )
    def on_evidence_change(*args):
        values = args[:len(columns)]
        old_evidence = args[len(columns)] or {}
        target = args[len(columns) + 1]

        new_evidence = {}
        for node, val in zip(columns, values):
            if val and node != target:
                new_evidence[node] = val
        return new_evidence

    # ------------------------------------------------------------------
    # 4) Reset / Clear buttons
    # ------------------------------------------------------------------
    @app.callback(
        Output("evidence-store", "data", allow_duplicate=True),
        [Input("reset-btn", "n_clicks"),
         Input("clear-evidence-btn", "n_clicks")],
        prevent_initial_call=True,
    )
    def on_reset(n1, n2):
        return {}

    # ------------------------------------------------------------------
    # 5) Main update: evidence/target change → inference → graph + bars
    # ------------------------------------------------------------------
    @app.callback(
        [Output("network-graph", "figure"),
         Output("posterior-panel", "children"),
         Output("sensitivity-graph", "figure")],
        [Input("evidence-store", "data"),
         Input("target-dropdown", "value"),
         Input("node-search", "value")],
        [State("selected-node-store", "data")],
    )
    def update_all(evidence, target, search_term, selected_node):
        evidence = evidence or {}
        search_term = search_term or ""

        # Run inference
        posteriors: Dict[str, Dict[str, float]] = {}
        if analyzer.is_trained:
            try:
                # Use blanket query for focused node, full for small networks
                if len(columns) > 30:
                    focus = selected_node if selected_node else target
                    raw = analyzer.query_blanket(focus, evidence)
                else:
                    raw = analyzer.query_all(evidence)

                for var, factor in raw.items():
                    state_names = factor.state_names[var]
                    values = factor.values.flatten().tolist()
                    posteriors[var] = {
                        str(s): float(v) for s, v in zip(state_names, values)
                    }
            except Exception as exc:
                logger.warning("Inference error: %s", exc)

        # Build outputs
        net_fig = _build_network_figure(
            analyzer, evidence, posteriors, target,
            selected_node=selected_node,
            search_term=search_term,
        )
        post_cards = _build_posterior_bars(posteriors, target)
        sens_fig = _build_sensitivity_figure(analyzer, target)

        return net_fig, post_cards, sens_fig


# =========================================================================
# Entry point
# =========================================================================

def run_server(
    analyzer: BayesianAnalyzer,
    host: str = "0.0.0.0",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Create and run the interactive dashboard server.

    Parameters
    ----------
    analyzer : BayesianAnalyzer
        A trained analyzer instance.
    host : str
        Bind address.
    port : int
        Bind port.
    debug : bool
        Enable Dash debug mode.
    """
    if not analyzer.is_trained:
        raise RuntimeError("Analyzer must be trained before launching the dashboard.")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )
    app.title = "BN Interactive Dashboard"
    app.layout = layout(analyzer)
    register_callbacks(app, analyzer)

    logger.info("Starting interactive dashboard at http://%s:%d", host, port)
    app.run(host=host, port=port, debug=debug)
