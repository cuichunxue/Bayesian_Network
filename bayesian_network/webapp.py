"""Flask web application for interactive Bayesian Network analysis.

Upload or paste CSV data, configure analysis parameters, train models,
and explore results through multiple interactive Plotly charts.

Usage::

    # From CLI
    python -m bayesian_network.webapp
    python App.py --web

    # Programmatic
    from bayesian_network.webapp import create_app
    app = create_app()
    app.run(debug=True)
"""

from __future__ import annotations

import csv
import io
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flask import (
    Flask,
    flash,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)

from bayesian_network.analyzer import BayesianAnalyzer
from bayesian_network.config import BNConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory model store with TTL
# ---------------------------------------------------------------------------
SESSION_TTL = 3600  # 1 hour

@dataclass
class _Session:
    analyzer: BayesianAnalyzer
    df_raw: pd.DataFrame
    created: float
    target_node: str
    evidence: Dict[str, str]

_store: Dict[str, _Session] = {}
_store_lock = threading.Lock()


def _cleanup_expired() -> None:
    """Remove sessions older than TTL."""
    now = time.time()
    with _store_lock:
        expired = [k for k, v in _store.items() if now - v.created > SESSION_TTL]
        for k in expired:
            del _store[k]


def _get_session(sid: str) -> Optional[_Session]:
    _cleanup_expired()
    with _store_lock:
        return _store.get(sid)


def _put_session(sid: str, sess: _Session) -> None:
    _cleanup_expired()
    with _store_lock:
        _store[sid] = sess


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------
_COLORS = {
    "primary": "#4C78A8",
    "danger": "#E45756",
    "success": "#54A24B",
    "info": "#72B7B2",
    "warning": "#EECA3B",
    "muted": "#B4B4B4",
    "edge": "rgba(150,160,175,0.55)",
    "bg": "#FAFBFC",
    "text": "#2E3440",
}

import math
import networkx as nx


def _compute_layout(g: nx.DiGraph, cfg: BNConfig) -> Dict[str, Tuple[float, float]]:
    if g.number_of_nodes() == 0:
        return {}
    if nx.is_directed_acyclic_graph(g) and g.number_of_edges() > 0:
        try:
            for layer, nodes in enumerate(nx.topological_generations(g)):
                for node in nodes:
                    g.nodes[node]["subset"] = layer
            pos = nx.multipartite_layout(g, subset_key="subset", align="horizontal")
            return {n: (y, -x) for n, (x, y) in pos.items()}
        except Exception:
            pass
    return nx.spring_layout(g, k=cfg.layout_k, seed=cfg.layout_seed, iterations=80)


def build_network_fig(
    analyzer: BayesianAnalyzer,
    target_node: str,
    evidence: Dict[str, str],
    posteriors: Dict[str, Dict[str, float]],
) -> go.Figure:
    """Directed network graph with posterior-coloured nodes."""
    g = nx.DiGraph()
    g.add_nodes_from(analyzer.columns)
    g.add_edges_from(analyzer.edges)
    pos = _compute_layout(g, analyzer.config)

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
        line=dict(width=1.3, color=_COLORS["edge"]),
        hoverinfo="none", showlegend=False,
    ))

    # Arrow annotations
    annotations = []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist > 0:
            x1a = x1 - dx / dist * 0.06
            y1a = y1 - dy / dist * 0.06
        else:
            x1a, y1a = x1, y1
        is_target_edge = (u == target_node or v == target_node)
        annotations.append(dict(
            ax=x0, ay=y0, x=x1a, y=y1a,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.1,
            arrowwidth=2.0 if is_target_edge else 1.2,
            arrowcolor="rgba(55,90,180,0.8)" if is_target_edge else _COLORS["edge"],
            opacity=0.85 if is_target_edge else 0.6,
        ))

    # Nodes
    nodes = list(g.nodes())
    nx_arr = [pos[n][0] for n in nodes]
    ny_arr = [pos[n][1] for n in nodes]
    sizes, colours, borders, bw, hovers = [], [], [], [], []

    for n in nodes:
        sizes.append(36)
        post = posteriors.get(n)
        if n in evidence:
            colours.append("rgba(34,139,34,0.85)")
        elif post:
            max_p = max(post.values())
            if max_p >= 0.7:
                colours.append(_COLORS["danger"])
            elif max_p >= 0.4:
                colours.append(_COLORS["primary"])
            else:
                colours.append(_COLORS["info"])
        else:
            colours.append(_COLORS["muted"])

        if n == target_node:
            borders.append(_COLORS["danger"])
            bw.append(3.0)
        elif n in evidence:
            borders.append(_COLORS["success"])
            bw.append(2.5)
        else:
            borders.append("rgba(50,60,80,0.9)")
            bw.append(1.5)

        parts = [f"<b>{n}</b>"]
        if n in evidence:
            parts.append(f"Evidence: {evidence[n]}")
        if n == target_node:
            parts.append("(TARGET)")
        if post:
            for st, pr in sorted(post.items(), key=lambda x: -x[1]):
                parts.append(f"  {st}: {pr:.1%}")
        hovers.append("<br>".join(parts))

    fig.add_trace(go.Scatter(
        x=nx_arr, y=ny_arr, mode="markers+text",
        text=nodes, textposition="top center",
        textfont=dict(size=11, color=_COLORS["text"]),
        hovertext=hovers, hoverinfo="text",
        marker=dict(size=sizes, color=colours,
                    line=dict(width=bw, color=borders)),
        showlegend=False,
    ))

    fig.update_layout(
        annotations=annotations, template="plotly_white",
        plot_bgcolor=_COLORS["bg"], paper_bgcolor="white",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        dragmode="pan", height=500,
    )
    return fig


def build_sensitivity_fig(analyzer: BayesianAnalyzer, target_node: str, top_k: int = 15) -> go.Figure:
    """Horizontal MI bar chart."""
    try:
        mi = analyzer.compute_sensitivity(target_node)
    except KeyError:
        return go.Figure()
    items = list(mi.items())[:top_k]
    if not items:
        return go.Figure()

    names = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    mx = max(values) if values else 1.0
    bar_c = [
        _COLORS["danger"] if v >= mx * 0.8
        else _COLORS["primary"] if v >= mx * 0.3
        else _COLORS["info"]
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=bar_c, line=dict(width=0.5, color="rgba(0,0,0,0.12)")),
        hovertemplate="%{y}: MI = %{x:.6f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=_COLORS["bg"], paper_bgcolor="white",
        margin=dict(l=120, r=15, t=30, b=40),
        xaxis=dict(title="Mutual Information", gridcolor="rgba(0,0,0,0.04)"),
        yaxis=dict(title=""),
        height=max(300, len(names) * 28 + 80),
    )
    return fig


def build_posterior_fig(posteriors: Dict[str, Dict[str, float]], target_node: str) -> go.Figure:
    """Grouped bar chart of posterior probabilities for all variables."""
    if not posteriors:
        return go.Figure()

    # Order: target first, then by name
    ordered = []
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
        color = _COLORS["danger"] if var == target_node else _COLORS["primary"]
        fig.add_trace(go.Bar(
            x=states, y=vals, name=var,
            marker=dict(color=color, line=dict(width=0.5, color="rgba(0,0,0,0.1)")),
            hovertemplate="%{x}: %{y:.1%}<extra>" + var + "</extra>",
            showlegend=False,
        ), row=i, col=1)
        fig.update_yaxes(range=[0, 1], tickformat=".0%", row=i, col=1)

    total_h = max(200, len(ordered) * 160)
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=_COLORS["bg"], paper_bgcolor="white",
        margin=dict(l=50, r=20, t=40, b=20),
        height=total_h,
    )
    return fig


def build_cpd_fig(analyzer: BayesianAnalyzer, node: str) -> go.Figure:
    """CPD heatmap for a single node."""
    import itertools

    model = analyzer.model
    cpd = model.get_cpds(node)
    values = cpd.get_values()
    state_names = cpd.state_names
    var = cpd.variable
    row_labels = [str(s) for s in state_names[var]]

    parents = cpd.variables[1:]
    if parents:
        parent_states = [state_names[p] for p in parents]
        combos = list(itertools.product(*parent_states))
        col_labels = [" | ".join(f"{p}={s}" for p, s in zip(parents, c)) for c in combos]
    else:
        col_labels = ["(no parents)"]

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    fig = go.Figure(go.Heatmap(
        z=values, x=col_labels, y=row_labels,
        colorscale="Blues",
        hovertemplate="P(%{y} | %{x}) = %{z:.4f}<extra></extra>",
        colorbar=dict(title="P", thickness=12),
    ))
    fig.update_layout(
        title=dict(text=f"P({node} | parents)", font=dict(size=14)),
        template="plotly_white",
        plot_bgcolor=_COLORS["bg"], paper_bgcolor="white",
        xaxis=dict(title="Parent States", tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(title=f"{node}", autorange="reversed"),
        margin=dict(l=70, r=20, t=50, b=100),
        height=max(300, len(row_labels) * 40 + 150),
    )
    return fig


def build_data_distribution_fig(df: pd.DataFrame) -> go.Figure:
    """Value distribution bar charts for every column in the dataset."""
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

    palette = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6"]

    for idx, col in enumerate(cols):
        r = idx // n_cols + 1
        c = idx % n_cols + 1
        vc = df[col].astype(str).value_counts()
        fig.add_trace(go.Bar(
            x=vc.index.tolist(), y=vc.values.tolist(),
            marker=dict(color=palette[idx % len(palette)]),
            hovertemplate="%{x}: %{y}<extra>" + col + "</extra>",
            showlegend=False,
        ), row=r, col=c)

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=_COLORS["bg"], paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=30),
        height=max(300, n_rows * 250),
    )
    return fig


def build_mi_matrix_fig(analyzer: BayesianAnalyzer) -> go.Figure:
    """Mutual information matrix heatmap between all pairs of columns."""
    cols = analyzer.columns
    n = len(cols)
    matrix = np.zeros((n, n))
    data = analyzer.data

    for i in range(n):
        xi = data[cols[i]].astype(str).to_numpy()
        for j in range(i + 1, n):
            from sklearn.metrics import mutual_info_score
            xj = data[cols[j]].astype(str).to_numpy()
            mi = float(mutual_info_score(xi, xj))
            matrix[i][j] = mi
            matrix[j][i] = mi

    fig = go.Figure(go.Heatmap(
        z=matrix, x=cols, y=cols,
        colorscale="Viridis",
        hovertemplate="MI(%{x}, %{y}) = %{z:.4f}<extra></extra>",
        colorbar=dict(title="MI", thickness=12),
    ))
    fig.update_layout(
        title=dict(text="Mutual Information Matrix", font=dict(size=14)),
        template="plotly_white",
        plot_bgcolor=_COLORS["bg"], paper_bgcolor="white",
        margin=dict(l=80, r=20, t=50, b=80),
        height=max(400, n * 40 + 100),
        width=max(500, n * 40 + 150),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
    )
    return fig


def build_edge_strength_fig(analyzer: BayesianAnalyzer) -> go.Figure:
    """Bar chart showing MI for each learned edge."""
    edges = analyzer.edges
    if not edges:
        return go.Figure()

    data = analyzer.data
    from sklearn.metrics import mutual_info_score

    edge_labels = []
    mi_vals = []
    for u, v in edges:
        xu = data[u].astype(str).to_numpy()
        xv = data[v].astype(str).to_numpy()
        mi = float(mutual_info_score(xu, xv))
        edge_labels.append(f"{u} -> {v}")
        mi_vals.append(mi)

    # Sort by MI descending
    pairs = sorted(zip(edge_labels, mi_vals), key=lambda x: x[1], reverse=True)
    edge_labels = [p[0] for p in pairs][::-1]
    mi_vals = [p[1] for p in pairs][::-1]

    mx = max(mi_vals) if mi_vals else 1.0
    bar_c = [
        _COLORS["danger"] if v >= mx * 0.7
        else _COLORS["primary"] if v >= mx * 0.3
        else _COLORS["info"]
        for v in mi_vals
    ]

    fig = go.Figure(go.Bar(
        x=mi_vals, y=edge_labels, orientation="h",
        marker=dict(color=bar_c, line=dict(width=0.5, color="rgba(0,0,0,0.1)")),
        hovertemplate="%{y}: MI = %{x:.6f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Edge Strength (MI)", font=dict(size=14)),
        template="plotly_white",
        plot_bgcolor=_COLORS["bg"], paper_bgcolor="white",
        margin=dict(l=160, r=15, t=50, b=40),
        xaxis=dict(title="Mutual Information", gridcolor="rgba(0,0,0,0.04)"),
        yaxis=dict(title=""),
        height=max(300, len(edge_labels) * 26 + 100),
    )
    return fig


def build_markov_blanket_fig(analyzer: BayesianAnalyzer, target_node: str) -> go.Figure:
    """Subgraph showing only the Markov blanket of the target node."""
    try:
        mb = analyzer.compute_markov_blanket(target_node)
    except Exception:
        return go.Figure()

    if not mb:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=f"Markov Blanket of {target_node}: empty", font=dict(size=14)),
            height=200,
        )
        return fig

    mb_nodes = mb | {target_node}
    sub_edges = [(u, v) for u, v in analyzer.edges if u in mb_nodes and v in mb_nodes]

    g = nx.DiGraph()
    g.add_nodes_from(sorted(mb_nodes))
    g.add_edges_from(sub_edges)
    pos = _compute_layout(g, analyzer.config)

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
        line=dict(width=1.5, color=_COLORS["edge"]),
        hoverinfo="none", showlegend=False,
    ))

    # Arrows
    annotations = []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist > 0:
            x1a = x1 - dx / dist * 0.08
            y1a = y1 - dy / dist * 0.08
        else:
            x1a, y1a = x1, y1
        annotations.append(dict(
            ax=x0, ay=y0, x=x1a, y=y1a,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.3,
            arrowwidth=1.5, arrowcolor=_COLORS["primary"], opacity=0.75,
        ))

    # Nodes
    nodes = sorted(g.nodes())
    nx_arr = [pos[n][0] for n in nodes]
    ny_arr = [pos[n][1] for n in nodes]
    colors = [_COLORS["danger"] if n == target_node else _COLORS["primary"] for n in nodes]
    bw = [3.0 if n == target_node else 1.5 for n in nodes]

    fig.add_trace(go.Scatter(
        x=nx_arr, y=ny_arr, mode="markers+text",
        text=nodes, textposition="top center",
        textfont=dict(size=12, color=_COLORS["text"]),
        hovertext=[f"<b>{n}</b>" + (" (TARGET)" if n == target_node else " (blanket)") for n in nodes],
        hoverinfo="text",
        marker=dict(size=40, color=colors, line=dict(width=bw, color="rgba(50,60,80,0.9)")),
        showlegend=False,
    ))

    fig.update_layout(
        annotations=annotations,
        title=dict(text=f"Markov Blanket of {target_node} ({len(mb)} nodes)", font=dict(size=14)),
        template="plotly_white",
        plot_bgcolor=_COLORS["bg"], paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        dragmode="pan", height=400,
    )
    return fig


def _fig_to_div(fig: go.Figure) -> str:
    """Convert Plotly figure to embeddable HTML div."""
    return fig.to_html(
        include_plotlyjs=False, full_html=False,
        config={"scrollZoom": True, "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(analyzer: BayesianAnalyzer, evidence: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Run inference for all non-evidence variables."""
    posteriors: Dict[str, Dict[str, float]] = {}
    try:
        raw = analyzer.query_all(evidence)
        for var, factor in raw.items():
            state_names = factor.state_names[var]
            values = factor.values.flatten().tolist()
            posteriors[var] = {str(s): float(v) for s, v in zip(state_names, values)}
    except Exception as exc:
        logger.warning("Inference failed: %s", exc)
    return posteriors


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_app(secret_key: Optional[str] = None) -> Flask:
    """Create and return the Flask application."""
    app = Flask(__name__)
    app.secret_key = secret_key or uuid.uuid4().hex
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(INDEX_HTML)

    @app.route("/analyze", methods=["POST"])
    def analyze():
        # --- Parse input data ---
        df = None
        source = "unknown"

        # File upload
        f = request.files.get("csvfile")
        if f and f.filename:
            try:
                raw = f.read().decode("utf-8", errors="replace")
                df = pd.read_csv(io.StringIO(raw))
                source = f.filename
            except Exception as exc:
                flash(f"CSV file parse error: {exc}", "danger")
                return redirect(url_for("index"))

        # Text paste fallback
        if df is None:
            text = request.form.get("csvtext", "").strip()
            if not text:
                flash("Please upload a CSV file or paste CSV data.", "warning")
                return redirect(url_for("index"))
            try:
                # Auto-detect delimiter
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(text[:4096])
                    sep = dialect.delimiter
                except csv.Error:
                    sep = ","
                df = pd.read_csv(io.StringIO(text), sep=sep)
                source = "pasted data"
            except Exception as exc:
                flash(f"CSV parse error: {exc}", "danger")
                return redirect(url_for("index"))

        if df.empty or df.shape[1] < 2:
            flash("Data must have at least 2 columns and 1 row.", "warning")
            return redirect(url_for("index"))

        # --- Config from form ---
        scoring = request.form.get("scoring_method", "bic-d")
        max_indegree = int(request.form.get("max_indegree", "3"))
        prior_type = request.form.get("prior_type", "BDeu")
        ess = float(request.form.get("equivalent_sample_size", "5.0"))
        discretize = request.form.get("discretize_numeric") == "on"
        bins = int(request.form.get("discretize_bins", "5"))
        target_node = request.form.get("target_node", "").strip()

        try:
            config = BNConfig(
                scoring_method=scoring,
                max_indegree=max_indegree,
                prior_type=prior_type,
                equivalent_sample_size=ess,
                discretize_numeric=discretize,
                discretize_bins=bins,
                show_progress=False,
            )
        except Exception as exc:
            flash(f"Configuration error: {exc}", "danger")
            return redirect(url_for("index"))

        # --- Train ---
        try:
            analyzer = BayesianAnalyzer(df, config)
            analyzer.train_model()
        except Exception as exc:
            flash(f"Training error: {exc}", "danger")
            return redirect(url_for("index"))

        # Default target
        if not target_node or target_node not in analyzer.columns:
            target_node = analyzer.columns[0]

        # Store session
        sid = uuid.uuid4().hex[:12]
        _put_session(sid, _Session(
            analyzer=analyzer,
            df_raw=df,
            created=time.time(),
            target_node=target_node,
            evidence={},
        ))

        logger.info("Session %s created: %s (%d x %d), target=%s",
                     sid, source, df.shape[0], df.shape[1], target_node)

        return redirect(url_for("dashboard", sid=sid))

    @app.route("/dashboard/<sid>", methods=["GET", "POST"])
    def dashboard(sid: str):
        sess = _get_session(sid)
        if sess is None:
            flash("Session expired or not found. Please re-upload.", "warning")
            return redirect(url_for("index"))

        analyzer = sess.analyzer
        evidence = dict(sess.evidence)

        # Handle POST (evidence update)
        if request.method == "POST":
            action = request.form.get("action", "")

            if action == "update_evidence":
                new_evidence = {}
                for col in analyzer.columns:
                    val = request.form.get(f"ev_{col}", "")
                    if val:
                        new_evidence[col] = val
                sess.evidence = new_evidence
                evidence = new_evidence

            elif action == "clear_evidence":
                sess.evidence = {}
                evidence = {}

            elif action == "change_target":
                new_target = request.form.get("new_target", "")
                if new_target in analyzer.columns:
                    sess.target_node = new_target

        target_node = sess.target_node

        # --- Compute all charts ---
        posteriors = run_inference(analyzer, evidence)

        # Build figures
        net_fig = build_network_fig(analyzer, target_node, evidence, posteriors)
        sens_fig = build_sensitivity_fig(analyzer, target_node)
        post_fig = build_posterior_fig(posteriors, target_node)
        dist_fig = build_data_distribution_fig(sess.df_raw)
        mi_matrix_fig = build_mi_matrix_fig(analyzer)
        edge_fig = build_edge_strength_fig(analyzer)
        mb_fig = build_markov_blanket_fig(analyzer, target_node)

        # CPD heatmaps for nodes with parents
        cpd_divs = []
        model_nodes = set(analyzer.model.nodes()) if analyzer.model else set()
        for node in analyzer.columns:
            if node not in model_nodes:
                continue
            cpd = analyzer.model.get_cpds(node)
            if cpd is not None and len(cpd.variables) > 1:
                try:
                    cpd_fig = build_cpd_fig(analyzer, node)
                    cpd_divs.append((node, _fig_to_div(cpd_fig)))
                except Exception:
                    pass

        # Convert figures to divs
        net_div = _fig_to_div(net_fig)
        sens_div = _fig_to_div(sens_fig)
        post_div = _fig_to_div(post_fig)
        dist_div = _fig_to_div(dist_fig)
        mi_div = _fig_to_div(mi_matrix_fig)
        edge_div = _fig_to_div(edge_fig)
        mb_div = _fig_to_div(mb_fig)

        # State values per column (for evidence dropdowns)
        col_states: Dict[str, List[str]] = {}
        for col in analyzer.columns:
            col_states[col] = sorted(analyzer.data[col].astype(str).unique().tolist())

        # Summary
        summary = analyzer.summary()

        return render_template_string(
            DASHBOARD_HTML,
            sid=sid,
            summary=summary,
            target_node=target_node,
            evidence=evidence,
            columns=analyzer.columns,
            col_states=col_states,
            net_div=net_div,
            sens_div=sens_div,
            post_div=post_div,
            dist_div=dist_div,
            mi_div=mi_div,
            edge_div=edge_div,
            mb_div=mb_div,
            cpd_divs=cpd_divs,
            n_edges=len(analyzer.edges),
            edges=analyzer.edges[:30],
        )

    return app


# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bayesian Network Analyzer</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body { background: #f0f2f5; }
  .hero { background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
          color: white; padding: 48px 0; }
  .hero h1 { font-size: 2rem; font-weight: 700; }
  .hero p { color: #9ca3af; font-size: 1.05rem; }
  .card { border: none; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .form-label { font-weight: 600; font-size: 0.9rem; color: #374151; }
  textarea { font-family: 'Consolas', 'Monaco', monospace; font-size: 0.85rem; }
  .btn-primary { background: #4C78A8; border-color: #4C78A8; }
  .btn-primary:hover { background: #3d6291; border-color: #3d6291; }
  .param-group { background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .param-group h6 { color: #6b7280; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
</style>
</head>
<body>

<div class="hero text-center">
  <div class="container">
    <h1>Bayesian Network Analyzer</h1>
    <p>Upload CSV data or paste directly — analyze structure, sensitivity, and posteriors interactively</p>
  </div>
</div>

<div class="container py-4" style="max-width: 900px;">
  {% with messages = get_flashed_messages(with_categories=true) %}
  {% if messages %}
  {% for cat, msg in messages %}
  <div class="alert alert-{{ cat }} alert-dismissible fade show" role="alert">
    {{ msg }}
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
  </div>
  {% endfor %}
  {% endif %}
  {% endwith %}

  <form method="POST" action="/analyze" enctype="multipart/form-data">
    <div class="row g-4">
      <!-- Data Input -->
      <div class="col-12">
        <div class="card">
          <div class="card-body">
            <h5 class="card-title mb-3">Data Input</h5>
            <ul class="nav nav-tabs mb-3" role="tablist">
              <li class="nav-item">
                <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-file" type="button">Upload CSV File</button>
              </li>
              <li class="nav-item">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-paste" type="button">Paste CSV Data</button>
              </li>
            </ul>
            <div class="tab-content">
              <div class="tab-pane fade show active" id="tab-file">
                <input type="file" name="csvfile" accept=".csv,.tsv,.txt" class="form-control">
                <div class="form-text">Accepts .csv, .tsv files (max 50 MB)</div>
              </div>
              <div class="tab-pane fade" id="tab-paste">
                <textarea name="csvtext" class="form-control" rows="8"
                  placeholder="col1,col2,col3&#10;A,High,Yes&#10;B,Low,No&#10;..."></textarea>
                <div class="form-text">Paste CSV/TSV data. Delimiter is auto-detected.</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Parameters -->
      <div class="col-md-6">
        <div class="card h-100">
          <div class="card-body">
            <h5 class="card-title mb-3">Structure Learning</h5>
            <div class="mb-3">
              <label class="form-label">Scoring Method</label>
              <select name="scoring_method" class="form-select">
                <option value="bic-d" selected>BIC-D (default)</option>
                <option value="bic">BIC</option>
                <option value="k2">K2</option>
                <option value="bdeu">BDeu</option>
                <option value="aic-d">AIC-D</option>
                <option value="aic">AIC</option>
              </select>
            </div>
            <div class="mb-3">
              <label class="form-label">Max In-degree</label>
              <input type="number" name="max_indegree" value="3" min="1" max="10" class="form-control">
              <div class="form-text">Maximum number of parents per node</div>
            </div>
            <div class="mb-3">
              <label class="form-label">Target Node (optional)</label>
              <input type="text" name="target_node" class="form-control" placeholder="Auto-detect first column">
              <div class="form-text">Primary variable of interest</div>
            </div>
          </div>
        </div>
      </div>

      <div class="col-md-6">
        <div class="card h-100">
          <div class="card-body">
            <h5 class="card-title mb-3">Parameter Learning & Data</h5>
            <div class="mb-3">
              <label class="form-label">Prior Type</label>
              <select name="prior_type" class="form-select">
                <option value="BDeu" selected>BDeu</option>
                <option value="K2">K2</option>
                <option value="dirichlet">Dirichlet</option>
              </select>
            </div>
            <div class="mb-3">
              <label class="form-label">Equivalent Sample Size</label>
              <input type="number" name="equivalent_sample_size" value="5.0" min="0.1" step="0.1" class="form-control">
            </div>
            <div class="form-check mb-3">
              <input type="checkbox" name="discretize_numeric" class="form-check-input" id="disc_check" checked>
              <label class="form-check-label" for="disc_check">Auto-discretize numeric columns</label>
            </div>
            <div class="mb-3">
              <label class="form-label">Discretization Bins</label>
              <input type="number" name="discretize_bins" value="5" min="2" max="20" class="form-control">
            </div>
          </div>
        </div>
      </div>

      <div class="col-12 text-center">
        <button type="submit" class="btn btn-primary btn-lg px-5">
          Analyze
        </button>
      </div>
    </div>
  </form>
</div>

<div class="text-center py-3" style="color: #9ca3af; font-size: 0.85rem;">
  Bayesian Network Analyzer v1.2.0
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BN Dashboard — {{ target_node }}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { background: #f0f2f5; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  .topbar { background: #1f2937; color: white; padding: 10px 24px; display: flex; align-items: center; justify-content: space-between; }
  .topbar h1 { font-size: 1.1rem; font-weight: 600; margin: 0; }
  .topbar .meta { font-size: 0.8rem; color: #9ca3af; }
  .topbar a { color: #60a5fa; text-decoration: none; font-size: 0.85rem; }
  .stats-bar { background: white; padding: 8px 24px; border-bottom: 1px solid #e5e7eb; font-size: 0.85rem; color: #6b7280; display: flex; gap: 24px; flex-wrap: wrap; }
  .stats-bar b { color: #374151; }
  .card { border: none; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 16px; }
  .card-header { background: #f8f9fa; border-bottom: 1px solid #e5e7eb; font-weight: 600; font-size: 0.9rem; padding: 10px 16px; border-radius: 10px 10px 0 0 !important; }
  .card-body { padding: 12px 16px; }
  .nav-pills .nav-link { font-size: 0.85rem; color: #374151; border-radius: 6px; padding: 6px 14px; }
  .nav-pills .nav-link.active { background: #4C78A8; color: white; }
  .evidence-form select { font-size: 0.85rem; }
  .evidence-form label { font-size: 0.8rem; font-weight: 600; color: #374151; }
  .badge-ev { background: #dcfce7; color: #166534; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; }
  .badge-target { background: #fee2e2; color: #991b1b; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; }
  .edge-list { font-size: 0.8rem; color: #374151; max-height: 200px; overflow-y: auto; }
  .edge-list span { display: inline-block; background: #f3f4f6; padding: 2px 8px; border-radius: 4px; margin: 2px; }
  .tab-content > .tab-pane { padding-top: 12px; }
</style>
</head>
<body>

<div class="topbar">
  <div>
    <h1>Bayesian Network Dashboard</h1>
    <div class="meta">
      Target: <b>{{ target_node }}</b> |
      Evidence: {{ evidence|length }} variable(s) |
      {{ summary.n_rows }} rows x {{ summary.n_columns }} cols |
      {{ n_edges }} edges |
      Training: {{ summary.training_time_sec }}s
    </div>
  </div>
  <a href="/">New Analysis</a>
</div>

<div class="container-fluid py-3" style="max-width: 1600px;">
  <div class="row g-3">

    <!-- Left Sidebar: Evidence & Controls -->
    <div class="col-lg-3 col-md-4">
      <!-- Target selector -->
      <div class="card">
        <div class="card-header">Target Node</div>
        <div class="card-body">
          <form method="POST" action="/dashboard/{{ sid }}">
            <input type="hidden" name="action" value="change_target">
            <div class="d-flex gap-2">
              <select name="new_target" class="form-select form-select-sm">
                {% for col in columns %}
                <option value="{{ col }}" {{ 'selected' if col == target_node }}>{{ col }}</option>
                {% endfor %}
              </select>
              <button type="submit" class="btn btn-sm btn-outline-primary text-nowrap">Set</button>
            </div>
          </form>
        </div>
      </div>

      <!-- Evidence panel -->
      <div class="card">
        <div class="card-header d-flex justify-content-between align-items-center">
          Evidence Settings
          {% if evidence %}
          <form method="POST" action="/dashboard/{{ sid }}" class="d-inline">
            <input type="hidden" name="action" value="clear_evidence">
            <button type="submit" class="btn btn-sm btn-outline-secondary py-0 px-2" style="font-size:0.75rem;">Clear</button>
          </form>
          {% endif %}
        </div>
        <div class="card-body evidence-form">
          <form method="POST" action="/dashboard/{{ sid }}">
            <input type="hidden" name="action" value="update_evidence">
            {% for col in columns %}
            <div class="mb-2">
              <label class="d-flex align-items-center gap-1">
                {{ col }}
                {% if col == target_node %}<span class="badge-target">TARGET</span>{% endif %}
                {% if col in evidence %}<span class="badge-ev">SET</span>{% endif %}
              </label>
              <select name="ev_{{ col }}" class="form-select form-select-sm">
                <option value="">— none —</option>
                {% for state in col_states[col] %}
                <option value="{{ state }}" {{ 'selected' if evidence.get(col) == state }}>{{ state }}</option>
                {% endfor %}
              </select>
            </div>
            {% endfor %}
            <button type="submit" class="btn btn-primary btn-sm w-100 mt-2">Update Inference</button>
          </form>
        </div>
      </div>

      <!-- Edges -->
      <div class="card">
        <div class="card-header">Learned Edges ({{ n_edges }})</div>
        <div class="card-body edge-list">
          {% for u, v in edges %}
          <span>{{ u }} &rarr; {{ v }}</span>
          {% endfor %}
          {% if n_edges > 30 %}<div class="text-muted mt-1">... and {{ n_edges - 30 }} more</div>{% endif %}
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <div class="col-lg-9 col-md-8">
      <!-- Tab navigation -->
      <ul class="nav nav-pills mb-3" role="tablist">
        <li class="nav-item">
          <button class="nav-link active" data-bs-toggle="pill" data-bs-target="#tab-network" type="button">Network</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-sensitivity" type="button">Sensitivity</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-posterior" type="button">Posteriors</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-cpd" type="button">CPD Heatmaps</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-data" type="button">Data</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-mi" type="button">MI Matrix</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-edges" type="button">Edge Strength</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-blanket" type="button">Markov Blanket</button>
        </li>
      </ul>

      <div class="tab-content">
        <!-- Network -->
        <div class="tab-pane fade show active" id="tab-network">
          <div class="card">
            <div class="card-header">Network Structure</div>
            <div class="card-body p-1">{{ net_div | safe }}</div>
          </div>
        </div>

        <!-- Sensitivity -->
        <div class="tab-pane fade" id="tab-sensitivity">
          <div class="card">
            <div class="card-header">Sensitivity Analysis — Mutual Information</div>
            <div class="card-body p-1">{{ sens_div | safe }}</div>
          </div>
        </div>

        <!-- Posteriors -->
        <div class="tab-pane fade" id="tab-posterior">
          <div class="card">
            <div class="card-header">Posterior Distributions{% if evidence %} (given evidence){% endif %}</div>
            <div class="card-body p-1">{{ post_div | safe }}</div>
          </div>
        </div>

        <!-- CPD Heatmaps -->
        <div class="tab-pane fade" id="tab-cpd">
          {% if cpd_divs %}
          {% for node_name, cpd_div in cpd_divs %}
          <div class="card">
            <div class="card-header">CPD: {{ node_name }}</div>
            <div class="card-body p-1">{{ cpd_div | safe }}</div>
          </div>
          {% endfor %}
          {% else %}
          <div class="card">
            <div class="card-body text-muted">No nodes with parents found — no conditional probability tables to display.</div>
          </div>
          {% endif %}
        </div>

        <!-- Data Distribution -->
        <div class="tab-pane fade" id="tab-data">
          <div class="card">
            <div class="card-header">Data Distribution (raw values)</div>
            <div class="card-body p-1">{{ dist_div | safe }}</div>
          </div>
        </div>

        <!-- MI Matrix -->
        <div class="tab-pane fade" id="tab-mi">
          <div class="card">
            <div class="card-header">Mutual Information Matrix (all pairs)</div>
            <div class="card-body p-1">{{ mi_div | safe }}</div>
          </div>
        </div>

        <!-- Edge Strength -->
        <div class="tab-pane fade" id="tab-edges">
          <div class="card">
            <div class="card-header">Edge Strength (MI between connected nodes)</div>
            <div class="card-body p-1">{{ edge_div | safe }}</div>
          </div>
        </div>

        <!-- Markov Blanket -->
        <div class="tab-pane fade" id="tab-blanket">
          <div class="card">
            <div class="card-header">Markov Blanket of {{ target_node }}</div>
            <div class="card-body p-1">{{ mb_div | safe }}</div>
          </div>
        </div>
      </div>
    </div>

  </div>
</div>

<div class="text-center py-3" style="color: #9ca3af; font-size: 0.8rem;">
  Bayesian Network Analyzer v1.2.0
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Direct entry point
# ---------------------------------------------------------------------------
def main(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """Run the Flask development server."""
    app = create_app()
    logger.info("Starting BN Web App on http://%s:%d", host, port)
    print(f"\n  Bayesian Network Analyzer")
    print(f"  Open http://{host}:{port} in your browser\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    from bayesian_network.logging_config import setup_logging
    setup_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description="BN Web App")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(host=args.host, port=args.port, debug=args.debug)
