"""Flask web application for interactive Bayesian Network analysis.

Upload or paste CSV data, configure analysis parameters, train models,
and explore results through multiple interactive Plotly charts.

Designed to be beginner-friendly with:
- One-click demo data to get started instantly
- Step-by-step guidance with Japanese/English tooltips
- Contextual help panels explaining every concept
- Guided wizard-style parameter configuration

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
from sklearn.metrics import mutual_info_score

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template_string,
    request,
    url_for,
)

from bayesian_network.analyzer import BayesianAnalyzer
from bayesian_network.charts import (
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
from bayesian_network.demo_data import DEMO_TARGET, generate_demo_data

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
# Demo data defaults
# ---------------------------------------------------------------------------
DEMO_N = 500
DEMO_SEED = 42


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

    @app.route("/api/demo-csv", methods=["GET"])
    def demo_csv():
        """Return demo CSV data for the frontend to populate the textarea."""
        df = generate_demo_data(n=DEMO_N, seed=DEMO_SEED)
        csv_text = df.to_csv(index=False)
        return jsonify({
            "csv": csv_text,
            "description": "Commute time prediction demo: Weather, DayOfWeek, Traffic, Accident, Commute",
            "target_node": DEMO_TARGET,
            "rows": len(df),
            "columns": list(df.columns),
        })

    @app.route("/demo", methods=["POST"])
    def demo():
        """One-click demo: generate data, train, and redirect to dashboard."""
        df = generate_demo_data(n=DEMO_N, seed=DEMO_SEED)

        config = BNConfig(
            scoring_method="bic-d",
            max_indegree=3,
            prior_type="BDeu",
            equivalent_sample_size=5.0,
            discretize_numeric=False,
            show_progress=False,
        )

        try:
            analyzer = BayesianAnalyzer(df, config)
            analyzer.train_model()
        except Exception as exc:
            flash(f"Demo training error: {exc}", "danger")
            return redirect(url_for("index"))

        sid = uuid.uuid4().hex[:12]
        _put_session(sid, _Session(
            analyzer=analyzer,
            df_raw=df,
            created=time.time(),
            target_node=DEMO_TARGET,
            evidence={},
        ))

        logger.info("Demo session %s created: %d rows, target=%s", sid, DEMO_N, DEMO_TARGET)
        flash("Demo data loaded! Explore the dashboard below.", "success")
        return redirect(url_for("dashboard", sid=sid))

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
                flash("Please upload a CSV file or paste CSV data. / CSVファイルをアップロードするか、CSVデータを貼り付けてください。", "warning")
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
            flash("Data must have at least 2 columns and 1 row. / データには少なくとも2列と1行が必要です。", "warning")
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
            flash("Session expired or not found. Please re-upload. / セッションが期限切れです。再度アップロードしてください。", "warning")
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
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bayesian Network Analyzer</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  :root {
    --primary: #4C78A8;
    --primary-dark: #3d6291;
    --accent: #E45756;
    --bg: #f0f2f5;
    --card-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  body { background: var(--bg); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Hiragino Sans', 'Yu Gothic', sans-serif; }

  /* Hero */
  .hero {
    background: linear-gradient(135deg, #1f2937 0%, #374151 50%, #4C78A8 100%);
    color: white; padding: 48px 0 40px;
    position: relative; overflow: hidden;
  }
  .hero::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 5 L55 20 L55 40 L30 55 L5 40 L5 20 Z' fill='none' stroke='rgba(255,255,255,0.04)' stroke-width='1'/%3E%3C/svg%3E");
    pointer-events: none;
  }
  .hero h1 { font-size: 2.2rem; font-weight: 700; position: relative; }
  .hero p { color: #c0c8d4; font-size: 1.05rem; position: relative; }

  /* Quick start */
  .quick-start {
    background: linear-gradient(135deg, #dbeafe 0%, #ede9fe 100%);
    border: 2px solid #93c5fd;
    border-radius: 16px; padding: 28px; margin-bottom: 28px;
    text-align: center; position: relative;
  }
  .quick-start h3 { color: #1e40af; font-weight: 700; margin-bottom: 6px; }
  .quick-start p { color: #4b5563; margin-bottom: 16px; }
  .quick-start .badge-new {
    position: absolute; top: -10px; right: 20px;
    background: var(--accent); color: white; padding: 4px 14px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 700;
  }

  .btn-demo {
    background: linear-gradient(135deg, #4C78A8, #6366f1);
    border: none; color: white; font-size: 1.1rem; font-weight: 600;
    padding: 14px 40px; border-radius: 12px;
    box-shadow: 0 4px 14px rgba(76,120,168,0.35);
    transition: all 0.2s;
  }
  .btn-demo:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(76,120,168,0.45);
    color: white;
  }
  .btn-demo:active { transform: translateY(0); }

  /* Cards */
  .card { border: none; border-radius: 12px; box-shadow: var(--card-shadow); }
  .form-label { font-weight: 600; font-size: 0.9rem; color: #374151; }
  textarea { font-family: 'Consolas', 'Monaco', 'SFMono-Regular', monospace; font-size: 0.85rem; }
  .btn-primary { background: var(--primary); border-color: var(--primary); }
  .btn-primary:hover { background: var(--primary-dark); border-color: var(--primary-dark); }

  /* Param groups */
  .param-group { background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .param-group h6 { color: #6b7280; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }

  /* Help tooltips */
  .help-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 18px; height: 18px; border-radius: 50%;
    background: #e5e7eb; color: #6b7280; font-size: 11px; font-weight: 700;
    cursor: help; margin-left: 4px; vertical-align: middle;
  }
  .help-text { font-size: 0.8rem; color: #6b7280; margin-top: 2px; }

  /* Steps indicator */
  .steps { display: flex; gap: 0; margin-bottom: 28px; }
  .step {
    flex: 1; text-align: center; padding: 12px 8px;
    background: white; border: 1px solid #e5e7eb;
    font-size: 0.85rem; color: #6b7280;
    position: relative;
  }
  .step:first-child { border-radius: 10px 0 0 10px; }
  .step:last-child { border-radius: 0 10px 10px 0; }
  .step.active { background: var(--primary); color: white; border-color: var(--primary); font-weight: 600; }
  .step .step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 24px; height: 24px; border-radius: 50%;
    background: rgba(0,0,0,0.1); font-weight: 700; font-size: 0.8rem;
    margin-right: 6px;
  }
  .step.active .step-num { background: rgba(255,255,255,0.3); }

  /* Loading overlay */
  .loading-overlay {
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.5); z-index: 9999;
    align-items: center; justify-content: center;
  }
  .loading-overlay.show { display: flex; }
  .loading-box {
    background: white; border-radius: 16px; padding: 40px 50px;
    text-align: center; box-shadow: 0 10px 40px rgba(0,0,0,0.2);
  }
  .loading-box .spinner-border { width: 3rem; height: 3rem; color: var(--primary); }
  .loading-box p { margin-top: 16px; font-weight: 600; color: #374151; }
  .loading-box .sub { font-size: 0.85rem; color: #6b7280; font-weight: 400; }

  /* Preview table */
  .preview-table { font-size: 0.8rem; max-height: 200px; overflow-y: auto; }
  .preview-table table { margin-bottom: 0; }
  .preview-table th { background: #f8f9fa; position: sticky; top: 0; }

  /* Section divider */
  .section-or {
    display: flex; align-items: center; gap: 12px;
    margin: 20px 0; color: #9ca3af; font-size: 0.85rem;
  }
  .section-or::before, .section-or::after {
    content: ''; flex: 1; height: 1px; background: #e5e7eb;
  }
</style>
</head>
<body>

<div class="hero text-center">
  <div class="container" style="position:relative;">
    <h1>Bayesian Network Analyzer</h1>
    <p>CSVデータをアップロード or 貼り付けるだけ &mdash; ベイジアンネットワークを自動学習&可視化</p>
    <p style="font-size:0.85rem; color:#9ca3af;">Upload CSV data or paste directly &mdash; analyze structure, sensitivity, and posteriors interactively</p>
  </div>
</div>

<div class="container py-4" style="max-width: 960px;">
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

  <!-- Quick Start: Demo Button -->
  <div class="quick-start">
    <span class="badge-new">RECOMMENDED</span>
    <h3>はじめての方はこちら / Quick Start</h3>
    <p>サンプルデータで体験してみましょう。ワンクリックでデモデータの分析結果を確認できます。</p>
    <form method="POST" action="/demo" id="demo-form">
      <button type="submit" class="btn btn-demo" id="demo-btn">
        Demo Data で試す
      </button>
    </form>
    <div style="margin-top:12px; font-size:0.8rem; color:#6b7280;">
      Weather &rarr; Traffic &rarr; Commute の因果関係を持つ500行のサンプルデータで分析します
    </div>
  </div>

  <div class="section-or">または自分のデータを使う / Or use your own data</div>

  <!-- Step indicator -->
  <div class="steps">
    <div class="step active">
      <span class="step-num">1</span>データ入力
    </div>
    <div class="step">
      <span class="step-num">2</span>設定
    </div>
    <div class="step">
      <span class="step-num">3</span>分析開始
    </div>
  </div>

  <form method="POST" action="/analyze" enctype="multipart/form-data" id="analyze-form">
    <div class="row g-4">
      <!-- Data Input -->
      <div class="col-12">
        <div class="card">
          <div class="card-body">
            <h5 class="card-title mb-1">
              Step 1: データ入力 / Data Input
              <span class="help-icon" data-bs-toggle="tooltip" title="CSVファイルをアップロードするか、テキストエリアに貼り付けてください。カンマ区切りのデータが必要です。">?</span>
            </h5>
            <p class="help-text mb-3">分析したいCSVデータを用意してください。ファイルアップロードまたは直接貼り付けが可能です。</p>
            <ul class="nav nav-tabs mb-3" role="tablist">
              <li class="nav-item">
                <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-file" type="button">
                  ファイルアップロード / Upload CSV
                </button>
              </li>
              <li class="nav-item">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-paste" type="button">
                  貼り付け / Paste Data
                </button>
              </li>
            </ul>
            <div class="tab-content">
              <div class="tab-pane fade show active" id="tab-file">
                <input type="file" name="csvfile" accept=".csv,.tsv,.txt" class="form-control" id="csvfile-input">
                <div class="form-text">.csv, .tsv ファイルに対応 (最大 50 MB) / Accepts .csv, .tsv files (max 50 MB)</div>
                <div id="file-preview" class="preview-table mt-2" style="display:none;"></div>
              </div>
              <div class="tab-pane fade" id="tab-paste">
                <textarea name="csvtext" class="form-control" rows="8" id="csvtext-input"
                  placeholder="col1,col2,col3&#10;A,High,Yes&#10;B,Low,No&#10;...&#10;&#10;ヒント: Excelからコピー&ペーストもできます"></textarea>
                <div class="form-text">CSVまたはTSVデータを貼り付けてください。区切り文字は自動検出されます。</div>
                <button type="button" class="btn btn-outline-secondary btn-sm mt-2" id="load-sample-btn">
                  サンプルデータを読み込む / Load Sample
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Parameters (collapsible for beginners) -->
      <div class="col-12">
        <div class="card">
          <div class="card-body">
            <h5 class="card-title mb-1">
              Step 2: 分析設定 / Analysis Settings
              <span class="help-icon" data-bs-toggle="tooltip" title="初心者の方はデフォルト設定のままで大丈夫です。上級者向けの設定も展開して調整できます。">?</span>
            </h5>
            <p class="help-text mb-3">初心者の方はデフォルトのままでOKです。必要に応じて調整してください。</p>

            <!-- Target node (always visible) -->
            <div class="mb-3">
              <label class="form-label">
                ターゲット変数 / Target Node
                <span class="help-icon" data-bs-toggle="tooltip" title="最も注目したい変数を指定します。空欄の場合、最初の列が自動選択されます。例: 「Commute」（通勤時間）を予測したい場合に指定します。">?</span>
              </label>
              <input type="text" name="target_node" class="form-control" placeholder="自動検出（最初の列） / Auto-detect first column">
              <div class="help-text">この変数に対する他の変数の影響度を分析します</div>
            </div>

            <!-- Advanced settings (collapsed by default) -->
            <div class="accordion" id="advancedAccordion">
              <div class="accordion-item border-0">
                <h2 class="accordion-header">
                  <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#advancedSettings" style="font-size:0.9rem; background:#f8f9fa; border-radius:8px;">
                    上級者向け設定を表示 / Show Advanced Settings
                  </button>
                </h2>
                <div id="advancedSettings" class="accordion-collapse collapse" data-bs-parent="#advancedAccordion">
                  <div class="accordion-body px-0 pb-0">
                    <div class="row g-3">
                      <div class="col-md-6">
                        <div class="param-group">
                          <h6>構造学習 / Structure Learning</h6>
                          <div class="mb-3">
                            <label class="form-label">
                              スコアリング手法 / Scoring Method
                              <span class="help-icon" data-bs-toggle="tooltip" title="ネットワーク構造の評価基準です。BIC-D（デフォルト）はバランスの取れた選択です。">?</span>
                            </label>
                            <select name="scoring_method" class="form-select">
                              <option value="bic-d" selected>BIC-D (推奨 / recommended)</option>
                              <option value="bic">BIC</option>
                              <option value="k2">K2</option>
                              <option value="bdeu">BDeu</option>
                              <option value="aic-d">AIC-D</option>
                              <option value="aic">AIC</option>
                            </select>
                          </div>
                          <div class="mb-3">
                            <label class="form-label">
                              最大入次数 / Max In-degree
                              <span class="help-icon" data-bs-toggle="tooltip" title="各ノード（変数）が持てる親ノードの最大数です。大きくすると複雑なモデルになりますが、計算時間が増えます。">?</span>
                            </label>
                            <input type="number" name="max_indegree" value="3" min="1" max="10" class="form-control">
                          </div>
                        </div>
                      </div>
                      <div class="col-md-6">
                        <div class="param-group">
                          <h6>パラメータ学習 & データ / Parameter Learning</h6>
                          <div class="mb-3">
                            <label class="form-label">
                              事前分布 / Prior Type
                              <span class="help-icon" data-bs-toggle="tooltip" title="データが少ない場合の補正方法です。BDeu（デフォルト）が一般的です。">?</span>
                            </label>
                            <select name="prior_type" class="form-select">
                              <option value="BDeu" selected>BDeu (推奨 / recommended)</option>
                              <option value="K2">K2</option>
                              <option value="dirichlet">Dirichlet</option>
                            </select>
                          </div>
                          <div class="mb-3">
                            <label class="form-label">
                              等価サンプルサイズ / Equivalent Sample Size
                              <span class="help-icon" data-bs-toggle="tooltip" title="事前分布の強さです。大きいほど事前情報を重視します。5.0がデフォルトです。">?</span>
                            </label>
                            <input type="number" name="equivalent_sample_size" value="5.0" min="0.1" step="0.1" class="form-control">
                          </div>
                          <div class="form-check mb-3">
                            <input type="checkbox" name="discretize_numeric" class="form-check-input" id="disc_check" checked>
                            <label class="form-check-label" for="disc_check">数値列を自動離散化 / Auto-discretize numeric columns</label>
                          </div>
                          <div class="mb-3">
                            <label class="form-label">離散化ビン数 / Discretization Bins</label>
                            <input type="number" name="discretize_bins" value="5" min="2" max="20" class="form-control">
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Submit -->
      <div class="col-12 text-center">
        <div class="mb-2">
          <small class="text-muted">Step 3: 準備ができたら「分析開始」をクリック</small>
        </div>
        <button type="submit" class="btn btn-primary btn-lg px-5" id="analyze-btn">
          分析開始 / Analyze
        </button>
      </div>
    </div>
  </form>

  <!-- Getting started guide -->
  <div class="card mt-4">
    <div class="card-body">
      <h5 class="card-title">はじめてのベイジアンネットワーク / Getting Started Guide</h5>
      <div class="row g-3 mt-1">
        <div class="col-md-4">
          <div class="param-group h-100">
            <h6 style="color: var(--primary);">ベイジアンネットワークとは？</h6>
            <p style="font-size:0.85rem; color:#4b5563;">
              変数間の因果関係や依存関係をグラフ（矢印付きの図）で表現し、
              確率的な推論を行う手法です。「天気が雨なら通勤時間は長くなる」
              といった関係を自動で発見します。
            </p>
          </div>
        </div>
        <div class="col-md-4">
          <div class="param-group h-100">
            <h6 style="color: var(--primary);">どんなデータが必要？</h6>
            <p style="font-size:0.85rem; color:#4b5563;">
              CSVファイル形式で、各列が変数、各行が観測データです。
              カテゴリ変数（Sunny, Rainy等）が最適ですが、
              数値データも自動で離散化できます。2列以上必要です。
            </p>
          </div>
        </div>
        <div class="col-md-4">
          <div class="param-group h-100">
            <h6 style="color: var(--primary);">分析結果の見方</h6>
            <p style="font-size:0.85rem; color:#4b5563;">
              ネットワーク図で変数間の因果関係を、感度分析で
              どの変数が影響力が強いかを確認できます。
              エビデンス（観測値）を設定して推論結果も見られます。
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="text-center py-3" style="color: #9ca3af; font-size: 0.85rem;">
  Bayesian Network Analyzer v2.0.0
</div>

<!-- Loading overlay -->
<div class="loading-overlay" id="loading-overlay">
  <div class="loading-box">
    <div class="spinner-border" role="status">
      <span class="visually-hidden">Loading...</span>
    </div>
    <p id="loading-text">分析中... / Analyzing...</p>
    <div class="sub">ネットワーク構造を学習しています。データ量によって数秒〜数分かかります。</div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
  var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.map(function(el) { return new bootstrap.Tooltip(el); });
});

// Show loading overlay on form submit
document.getElementById('analyze-form').addEventListener('submit', function() {
  document.getElementById('loading-overlay').classList.add('show');
  document.getElementById('loading-text').textContent = '分析中... / Analyzing...';
});
document.getElementById('demo-form').addEventListener('submit', function() {
  document.getElementById('loading-overlay').classList.add('show');
  document.getElementById('loading-text').textContent = 'デモデータを準備中... / Loading demo...';
});

// Load sample data button
document.getElementById('load-sample-btn').addEventListener('click', function() {
  fetch('/api/demo-csv')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      document.getElementById('csvtext-input').value = data.csv;
      // Switch to paste tab
      var pasteTab = document.querySelector('[data-bs-target="#tab-paste"]');
      bootstrap.Tab.getOrCreateInstance(pasteTab).show();
    });
});

// CSV file preview
document.getElementById('csvfile-input').addEventListener('change', function(e) {
  var file = e.target.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(ev) {
    var text = ev.target.result;
    var lines = text.split('\n').slice(0, 6);
    if (lines.length < 2) return;
    var headers = lines[0].split(',');
    var html = '<div class="alert alert-info py-2 px-3 mb-2" style="font-size:0.8rem;">Preview: ' +
      headers.length + ' columns detected</div>' +
      '<table class="table table-sm table-bordered"><thead><tr>';
    headers.forEach(function(h) { html += '<th>' + h.trim() + '</th>'; });
    html += '</tr></thead><tbody>';
    for (var i = 1; i < lines.length && i < 6; i++) {
      if (!lines[i].trim()) continue;
      html += '<tr>';
      lines[i].split(',').forEach(function(c) { html += '<td>' + c.trim() + '</td>'; });
      html += '</tr>';
    }
    html += '</tbody></table>';
    var preview = document.getElementById('file-preview');
    preview.innerHTML = html;
    preview.style.display = 'block';
  };
  reader.readAsText(file);
});
</script>
</body>
</html>"""


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BN Dashboard — {{ target_node }}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { background: #f0f2f5; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Hiragino Sans', 'Yu Gothic', sans-serif; }
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

  /* Help panel */
  .help-panel {
    background: linear-gradient(135deg, #eff6ff, #f0fdf4);
    border: 1px solid #bfdbfe;
    border-radius: 10px; padding: 16px; margin-bottom: 16px;
  }
  .help-panel h6 { color: #1e40af; font-size: 0.85rem; margin-bottom: 8px; }
  .help-panel p { font-size: 0.8rem; color: #4b5563; margin-bottom: 4px; line-height: 1.5; }
  .help-panel .help-toggle { font-size: 0.75rem; color: #6b7280; cursor: pointer; text-decoration: underline; }

  /* Tab help descriptions */
  .tab-help { font-size: 0.8rem; color: #6b7280; padding: 8px 0 4px; }
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
  <div>
    <a href="/" style="margin-right:12px;">New Analysis / 新規分析</a>
    <a href="#" onclick="document.getElementById('help-guide').style.display=document.getElementById('help-guide').style.display==='none'?'block':'none'; return false;" style="color:#fbbf24;">Help / ヘルプ</a>
  </div>
</div>

<!-- Beginner help guide (collapsible) -->
<div id="help-guide" style="display:none; background:#fffbeb; border-bottom:1px solid #fde68a; padding:16px 24px;">
  <div class="container-fluid" style="max-width:1600px;">
    <div class="row g-3">
      <div class="col-md-3">
        <h6 style="color:#92400e; font-size:0.85rem;">ダッシュボードの使い方</h6>
        <p style="font-size:0.8rem; color:#78350f;">
          左サイドバーでターゲット変数とエビデンス（観測値）を設定し、
          右側のタブで様々な分析結果を確認できます。
        </p>
      </div>
      <div class="col-md-3">
        <h6 style="color:#92400e; font-size:0.85rem;">ネットワーク図</h6>
        <p style="font-size:0.8rem; color:#78350f;">
          矢印は因果関係の方向を示します。赤いノードがターゲット、
          緑のノードがエビデンスとして設定した変数です。
        </p>
      </div>
      <div class="col-md-3">
        <h6 style="color:#92400e; font-size:0.85rem;">エビデンスとは？</h6>
        <p style="font-size:0.8rem; color:#78350f;">
          「Weather=Rainy」のように変数の値を固定すると、
          その条件下での他の変数の確率分布が更新されます。
          「もし雨なら？」というシミュレーションができます。
        </p>
      </div>
      <div class="col-md-3">
        <h6 style="color:#92400e; font-size:0.85rem;">感度分析</h6>
        <p style="font-size:0.8rem; color:#78350f;">
          相互情報量(MI)で各変数がターゲットにどれだけ影響するかを測定します。
          MIが高いほど、その変数はターゲットの予測に重要です。
        </p>
      </div>
    </div>
    <div class="text-end mt-2">
      <a href="#" onclick="document.getElementById('help-guide').style.display='none'; return false;" style="font-size:0.8rem; color:#92400e;">閉じる / Close</a>
    </div>
  </div>
</div>

<div class="container-fluid py-3" style="max-width: 1600px;">
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

  <div class="row g-3">

    <!-- Left Sidebar: Evidence & Controls -->
    <div class="col-lg-3 col-md-4">
      <!-- Help panel for beginners -->
      <div class="help-panel" id="sidebar-help">
        <h6>操作ガイド / Quick Guide</h6>
        <p>1. <b>ターゲット</b>を選択（分析したい変数）</p>
        <p>2. <b>エビデンス</b>を設定（「もし〇〇なら？」）</p>
        <p>3. 右側のタブで結果を確認</p>
        <span class="help-toggle" onclick="this.parentElement.style.display='none'">非表示にする</span>
      </div>

      <!-- Target selector -->
      <div class="card">
        <div class="card-header">
          ターゲット変数 / Target Node
        </div>
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
          <div style="font-size:0.75rem; color:#6b7280; margin-top:6px;">この変数を中心に分析します</div>
        </div>
      </div>

      <!-- Evidence panel -->
      <div class="card">
        <div class="card-header d-flex justify-content-between align-items-center">
          エビデンス / Evidence
          {% if evidence %}
          <form method="POST" action="/dashboard/{{ sid }}" class="d-inline">
            <input type="hidden" name="action" value="clear_evidence">
            <button type="submit" class="btn btn-sm btn-outline-secondary py-0 px-2" style="font-size:0.75rem;">クリア / Clear</button>
          </form>
          {% endif %}
        </div>
        <div class="card-body evidence-form">
          <div style="font-size:0.75rem; color:#6b7280; margin-bottom:8px;">
            変数の値を固定して「もし〇〇なら？」を試せます
          </div>
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
            <button type="submit" class="btn btn-primary btn-sm w-100 mt-2">推論を更新 / Update</button>
          </form>
        </div>
      </div>

      <!-- Edges -->
      <div class="card">
        <div class="card-header">学習されたエッジ / Edges ({{ n_edges }})</div>
        <div class="card-body edge-list">
          {% for u, v in edges %}
          <span>{{ u }} &rarr; {{ v }}</span>
          {% endfor %}
          {% if n_edges > 30 %}<div class="text-muted mt-1">... and {{ n_edges - 30 }} more</div>{% endif %}
          {% if n_edges == 0 %}
          <div class="text-muted">エッジが検出されませんでした。データの変数間に有意な依存関係がない可能性があります。</div>
          {% endif %}
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <div class="col-lg-9 col-md-8">
      <!-- Tab navigation -->
      <ul class="nav nav-pills mb-3" role="tablist">
        <li class="nav-item">
          <button class="nav-link active" data-bs-toggle="pill" data-bs-target="#tab-network" type="button">Network / ネットワーク</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-sensitivity" type="button">Sensitivity / 感度</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-posterior" type="button">Posteriors / 事後確率</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-cpd" type="button">CPD / 条件付確率</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-data" type="button">Data / データ分布</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-mi" type="button">MI Matrix / 相互情報量</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-edges" type="button">Edge Strength / エッジ強度</button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-blanket" type="button">Markov Blanket</button>
        </li>
      </ul>

      <div class="tab-content">
        <!-- Network -->
        <div class="tab-pane fade show active" id="tab-network">
          <div class="card">
            <div class="card-header">ネットワーク構造 / Network Structure</div>
            <div class="card-body">
              <div class="tab-help">
                矢印は因果関係の方向を示します。ノードにマウスを合わせると詳細情報が表示されます。
                赤 = ターゲット変数、緑 = エビデンス設定済み
              </div>
              <div class="p-1">{{ net_div | safe }}</div>
            </div>
          </div>
        </div>

        <!-- Sensitivity -->
        <div class="tab-pane fade" id="tab-sensitivity">
          <div class="card">
            <div class="card-header">感度分析 / Sensitivity Analysis — 相互情報量 (Mutual Information)</div>
            <div class="card-body">
              <div class="tab-help">
                各変数がターゲット変数にどれだけ影響するかを示します。バーが長いほど影響が大きい変数です。
              </div>
              <div class="p-1">{{ sens_div | safe }}</div>
            </div>
          </div>
        </div>

        <!-- Posteriors -->
        <div class="tab-pane fade" id="tab-posterior">
          <div class="card">
            <div class="card-header">事後確率分布 / Posterior Distributions{% if evidence %} （エビデンス適用済み）{% endif %}</div>
            <div class="card-body">
              <div class="tab-help">
                {% if evidence %}
                エビデンスを設定した条件での各変数の確率分布です。エビデンスなしの場合と比較してみましょう。
                {% else %}
                各変数の事前確率分布です。左サイドバーでエビデンスを設定すると、条件付き確率に更新されます。
                {% endif %}
              </div>
              <div class="p-1">{{ post_div | safe }}</div>
            </div>
          </div>
        </div>

        <!-- CPD Heatmaps -->
        <div class="tab-pane fade" id="tab-cpd">
          <div class="tab-help mb-2">
            条件付き確率表 (CPD) は、親ノードの各状態における子ノードの確率分布を示します。色が濃いほど確率が高くなります。
          </div>
          {% if cpd_divs %}
          {% for node_name, cpd_div in cpd_divs %}
          <div class="card">
            <div class="card-header">CPD: {{ node_name }}</div>
            <div class="card-body p-1">{{ cpd_div | safe }}</div>
          </div>
          {% endfor %}
          {% else %}
          <div class="card">
            <div class="card-body text-muted">親ノードを持つノードが見つかりません — 条件付き確率表を表示できません。</div>
          </div>
          {% endif %}
        </div>

        <!-- Data Distribution -->
        <div class="tab-pane fade" id="tab-data">
          <div class="card">
            <div class="card-header">データ分布 / Data Distribution (raw values)</div>
            <div class="card-body">
              <div class="tab-help">
                各変数のデータ分布（ヒストグラム）です。データの偏りや特徴を確認できます。
              </div>
              <div class="p-1">{{ dist_div | safe }}</div>
            </div>
          </div>
        </div>

        <!-- MI Matrix -->
        <div class="tab-pane fade" id="tab-mi">
          <div class="card">
            <div class="card-header">相互情報量行列 / Mutual Information Matrix (all pairs)</div>
            <div class="card-body">
              <div class="tab-help">
                全変数ペア間の相互情報量を行列で表示します。色が明るいほど、2つの変数間の依存関係が強いことを示します。
              </div>
              <div class="p-1">{{ mi_div | safe }}</div>
            </div>
          </div>
        </div>

        <!-- Edge Strength -->
        <div class="tab-pane fade" id="tab-edges">
          <div class="card">
            <div class="card-header">エッジ強度 / Edge Strength (MI between connected nodes)</div>
            <div class="card-body">
              <div class="tab-help">
                学習されたエッジ（因果関係）ごとの相互情報量です。値が大きいほど、その因果関係が強いことを示します。
              </div>
              <div class="p-1">{{ edge_div | safe }}</div>
            </div>
          </div>
        </div>

        <!-- Markov Blanket -->
        <div class="tab-pane fade" id="tab-blanket">
          <div class="card">
            <div class="card-header">マルコフブランケット / Markov Blanket of {{ target_node }}</div>
            <div class="card-body">
              <div class="tab-help">
                ターゲット変数の「マルコフブランケット」は、その変数の確率分布を完全に決定するのに必要十分な変数の集合です。
                親ノード、子ノード、子ノードの他の親ノードで構成されます。
              </div>
              <div class="p-1">{{ mb_div | safe }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

  </div>
</div>

<div class="text-center py-3" style="color: #9ca3af; font-size: 0.8rem;">
  Bayesian Network Analyzer v2.0.0
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
