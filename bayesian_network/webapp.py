"""Flask web application for interactive Bayesian Network analysis.

Upload or paste CSV data, configure analysis parameters, train models,
and explore results through multiple interactive Plotly charts.

Delegates to :mod:`bayesian_network.charts` for all chart construction.

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
from sklearn.metrics import mutual_info_score

from flask import (
    Flask,
    flash,
    redirect,
    render_template_string,
    request,
    url_for,
)

from bayesian_network.analyzer import BayesianAnalyzer
from bayesian_network.charts import (
    PALETTE,
    build_cpd_heatmap_figure,
    build_data_distribution_figure,
    build_edge_strength_figure,
    build_markov_blanket_figure,
    build_mi_matrix_figure,
    build_network_figure,
    build_posterior_figure,
    build_sensitivity_figure,
    fig_to_div,
)
from bayesian_network.config import BNConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory model store with TTL and max-session cap
# ---------------------------------------------------------------------------
SESSION_TTL = 3600  # 1 hour
MAX_SESSIONS = 50


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
        # Evict oldest sessions when exceeding cap
        while len(_store) >= MAX_SESSIONS:
            oldest_key = min(_store, key=lambda k: _store[k].created)
            del _store[oldest_key]
        _store[sid] = sess


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_inference(
    analyzer: BayesianAnalyzer,
    evidence: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
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


def _compute_edge_mi(analyzer: BayesianAnalyzer) -> Tuple[List[str], List[float]]:
    """Compute MI for each learned edge."""
    edges = analyzer.edges
    if not edges:
        return [], []
    data = analyzer.data
    edge_labels: List[str] = []
    mi_vals: List[float] = []
    for u, v in edges:
        xu = data[u].astype(str).to_numpy()
        xv = data[v].astype(str).to_numpy()
        mi = float(mutual_info_score(xu, xv))
        edge_labels.append(f"{u} -> {v}")
        mi_vals.append(mi)
    return edge_labels, mi_vals


def _mi_func_for(analyzer: BayesianAnalyzer):
    """Return a closure for pairwise MI computation (cached per call)."""
    data = analyzer.data

    def _mi(col_i: str, col_j: str) -> float:
        xi = data[col_i].astype(str).to_numpy()
        xj = data[col_j].astype(str).to_numpy()
        return float(mutual_info_score(xi, xj))

    return _mi


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

        if not target_node or target_node not in analyzer.columns:
            target_node = analyzer.columns[0]

        sid = uuid.uuid4().hex[:12]
        _put_session(sid, _Session(
            analyzer=analyzer,
            df_raw=df,
            created=time.time(),
            target_node=target_node,
            evidence={},
        ))

        logger.info(
            "Session %s created: %s (%d x %d), target=%s",
            sid, source, df.shape[0], df.shape[1], target_node,
        )

        return redirect(url_for("dashboard", sid=sid))

    @app.route("/dashboard/<sid>", methods=["GET", "POST"])
    def dashboard(sid: str):
        sess = _get_session(sid)
        if sess is None:
            flash("Session expired or not found. Please re-upload.", "warning")
            return redirect(url_for("index"))

        analyzer = sess.analyzer
        evidence = dict(sess.evidence)
        cfg = analyzer.config

        # Handle POST (evidence update)
        if request.method == "POST":
            action = request.form.get("action", "")

            if action == "update_evidence":
                new_evidence: Dict[str, str] = {}
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

        # --- Compute all charts (using shared charts module) ---
        posteriors = _run_inference(analyzer, evidence)

        net_fig = build_network_figure(
            columns=analyzer.columns,
            edges=analyzer.edges,
            layout_k=cfg.layout_k,
            layout_seed=cfg.layout_seed,
            target_node=target_node,
            evidence=evidence,
            posteriors=posteriors,
            show_title=False,
            width=0,
            height=500,
        )

        try:
            mi_scores = analyzer.compute_sensitivity(target_node)
        except KeyError:
            mi_scores = {}
        sens_fig = build_sensitivity_figure(
            mi_scores=mi_scores,
            target_node=target_node,
            show_title=False,
        )

        post_fig = build_posterior_figure(posteriors, target_node)
        dist_fig = build_data_distribution_figure(sess.df_raw)

        mi_matrix_fig = build_mi_matrix_figure(
            columns=analyzer.columns,
            mi_func=_mi_func_for(analyzer),
        )

        edge_labels, edge_mi_vals = _compute_edge_mi(analyzer)
        edge_fig = build_edge_strength_figure(edge_labels, edge_mi_vals)

        try:
            mb = analyzer.compute_markov_blanket(target_node)
        except Exception:
            mb = set()
        mb_fig = build_markov_blanket_figure(
            target_node=target_node,
            mb_nodes=mb,
            edges=analyzer.edges,
            layout_k=cfg.layout_k,
            layout_seed=cfg.layout_seed,
        )

        # CPD heatmaps for nodes with parents
        cpd_divs: List[Tuple[str, str]] = []
        model_nodes = set(analyzer.model.nodes()) if analyzer.model else set()
        for node in analyzer.columns:
            if node not in model_nodes:
                continue
            cpd = analyzer.model.get_cpds(node)
            if cpd is not None and len(cpd.variables) > 1:
                try:
                    cpd_fig = build_cpd_heatmap_figure(cpd=cpd, node=node)
                    cpd_divs.append((node, fig_to_div(cpd_fig)))
                except Exception:
                    pass

        # Convert figures to divs
        net_div = fig_to_div(net_fig)
        sens_div = fig_to_div(sens_fig)
        post_div = fig_to_div(post_fig)
        dist_div = fig_to_div(dist_fig)
        mi_div = fig_to_div(mi_matrix_fig)
        edge_div = fig_to_div(edge_fig)
        mb_div = fig_to_div(mb_fig)

        # State values per column (for evidence dropdowns)
        col_states: Dict[str, List[str]] = {}
        for col in analyzer.columns:
            col_states[col] = sorted(
                analyzer.data[col].astype(str).unique().tolist()
            )

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
