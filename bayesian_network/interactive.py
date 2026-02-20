"""Static HTML dashboard generator for Bayesian Network analysis.

Generates a self-contained HTML file combining network graph,
sensitivity analysis, evidence settings, and posterior probability bars.
No server required — just open the HTML in a browser.

Delegates to :mod:`bayesian_network.charts` for all chart construction.

Usage::

    from bayesian_network.interactive import save_dashboard
    save_dashboard(analyzer, "Commute", evidence={"Weather": "Rainy"},
                   html_path="output/dashboard_interactive.html")
"""

from __future__ import annotations

import html as html_mod
import logging
from pathlib import Path
from typing import Dict, Optional

from bayesian_network.analyzer import BayesianAnalyzer
from bayesian_network.charts import (
    PALETTE,
    build_network_figure,
    build_sensitivity_figure,
)

logger = logging.getLogger(__name__)


# =========================================================================
# HTML fragment builders
# =========================================================================

def _evidence_html(evidence: Dict[str, str]) -> str:
    """Render evidence settings as HTML."""
    if not evidence:
        return '<p style="color:#6B7280;font-size:13px;">No evidence set (prior distribution)</p>'
    rows = []
    for k, v in evidence.items():
        ek = html_mod.escape(str(k))
        ev = html_mod.escape(str(v))
        rows.append(
            f'<tr><td style="padding:4px 12px 4px 0;font-weight:600;">{ek}</td>'
            f'<td style="padding:4px 0;"><span style="background:#E8F5E9;'
            f'padding:2px 10px;border-radius:4px;font-size:13px;">{ev}</span></td></tr>'
        )
    return f'<table style="border-collapse:collapse;">{"".join(rows)}</table>'


def _posterior_html(
    posteriors: Dict[str, Dict[str, float]],
    target_node: str,
) -> str:
    """Render posterior probability bars as HTML."""
    if not posteriors:
        return '<p style="color:#6B7280;font-size:13px;">No posteriors computed</p>'

    ordered = []
    if target_node in posteriors:
        ordered.append(target_node)
    ordered += [k for k in posteriors if k != target_node]

    cards = []
    for var in ordered:
        probs = posteriors[var]
        if not probs:
            continue
        is_target = var == target_node
        border = f"2px solid {PALETTE['target_border']}" if is_target else "1px solid #DEE2E6"
        header_bg = "#FFF3F3" if is_target else "#F8F9FA"
        tag = ' <span style="color:#E45756;font-size:11px;">(TARGET)</span>' if is_target else ""
        ev = html_mod.escape(var)

        bars_html = []
        for state, prob in sorted(probs.items(), key=lambda x: -x[1]):
            pct = prob * 100
            if prob >= 0.5:
                color = "#DC3545"
            elif prob >= 0.2:
                color = "#4C78A8"
            else:
                color = "#72B7B2"
            es = html_mod.escape(str(state))
            bars_html.append(
                f'<div style="margin-bottom:5px;">'
                f'<div style="display:flex;align-items:center;margin-bottom:2px;">'
                f'<span style="font-size:12px;min-width:90px;display:inline-block;">{es}</span>'
                f'<span style="font-size:11px;color:#6B7280;">{pct:.1f}%</span></div>'
                f'<div style="background:#E9ECEF;border-radius:4px;height:14px;width:100%;">'
                f'<div style="background:{color};height:14px;border-radius:4px;'
                f'width:{pct:.1f}%;min-width:2px;"></div></div></div>'
            )

        cards.append(
            f'<div style="border:{border};border-radius:6px;margin-bottom:10px;">'
            f'<div style="padding:6px 12px;background:{header_bg};border-bottom:1px solid #DEE2E6;'
            f'border-radius:6px 6px 0 0;"><b>{ev}</b>{tag}</div>'
            f'<div style="padding:8px 12px;">{"".join(bars_html)}</div></div>'
        )

    return "".join(cards)


def _markov_blanket_html(analyzer: BayesianAnalyzer, target_node: str) -> str:
    """Render Markov blanket info as HTML."""
    try:
        mb = analyzer.compute_markov_blanket(target_node)
    except Exception:
        return ""
    if not mb:
        return ""
    items = ", ".join(f"<b>{html_mod.escape(n)}</b>" for n in sorted(mb))
    return (
        f'<div style="margin-top:12px;padding:8px 12px;background:#EDF2FF;'
        f'border-radius:6px;font-size:13px;">'
        f'<b>Markov Blanket</b> of {html_mod.escape(target_node)}: {items}</div>'
    )


# =========================================================================
# Main public API
# =========================================================================

def save_dashboard(
    analyzer: BayesianAnalyzer,
    target_node: str,
    evidence: Optional[Dict[str, str]] = None,
    html_path: str = "output/dashboard_interactive.html",
    top_k: int = 15,
) -> str:
    """Generate and save a self-contained interactive HTML dashboard.

    Parameters
    ----------
    analyzer : BayesianAnalyzer
        Trained analyzer instance.
    target_node : str
        Target variable for sensitivity analysis and display.
    evidence : dict, optional
        Evidence mapping (variable -> observed value).
    html_path : str
        Output file path.
    top_k : int
        Number of variables to show in sensitivity chart.

    Returns
    -------
    str
        Absolute path of the saved HTML file.
    """
    if not analyzer.is_trained:
        raise RuntimeError("Analyzer must be trained before generating dashboard.")

    evidence = evidence or {}

    # --- Run inference ---
    posteriors: Dict[str, Dict[str, float]] = {}
    try:
        raw = analyzer.query_all(evidence)
        for var, factor in raw.items():
            state_names = factor.state_names[var]
            values = factor.values.flatten().tolist()
            posteriors[var] = {str(s): float(v) for s, v in zip(state_names, values)}
    except Exception as exc:
        logger.warning("Inference failed: %s", exc)

    # --- Build Plotly figures (delegating to charts module) ---
    cfg = analyzer.config
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
        height=0,
    )

    try:
        mi_scores = analyzer.compute_sensitivity(target_node)
    except KeyError:
        mi_scores = {}
    sens_fig = build_sensitivity_figure(
        mi_scores=mi_scores,
        target_node=target_node,
        top_k=top_k,
        show_title=False,
    )

    net_div = net_fig.to_html(include_plotlyjs=False, full_html=False, config={
        "scrollZoom": True, "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    })
    sens_div = sens_fig.to_html(include_plotlyjs=False, full_html=False)

    # --- Build HTML sections ---
    evidence_section = _evidence_html(evidence)
    posterior_section = _posterior_html(posteriors, target_node)
    blanket_section = _markov_blanket_html(analyzer, target_node)

    ev_title = html_mod.escape(target_node)
    ev_count = len(evidence)
    ev_desc = ", ".join(
        f"{html_mod.escape(k)}={html_mod.escape(v)}" for k, v in evidence.items()
    ) if evidence else "none"

    # --- Summary stats ---
    summary = analyzer.summary()
    stats_html = (
        f'<span style="margin-right:20px;">Nodes: <b>{summary["n_columns"]}</b></span>'
        f'<span style="margin-right:20px;">Edges: <b>{summary["n_edges"]}</b></span>'
        f'<span style="margin-right:20px;">Rows: <b>{summary["n_rows"]}</b></span>'
        f'<span>Training: <b>{summary["training_time_sec"]}s</b></span>'
    )

    # --- Full HTML ---
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BN Dashboard — {ev_title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
         sans-serif; background: #F0F2F5; color: {PALETTE["text"]}; }}
  .header {{ background: #1F2937; color: white; padding: 14px 24px;
             display: flex; align-items: center; justify-content: space-between; }}
  .header h1 {{ font-size: 18px; font-weight: 600; }}
  .header .meta {{ font-size: 12px; color: #9CA3AF; }}
  .stats {{ background: white; padding: 8px 24px; border-bottom: 1px solid #E5E7EB;
            font-size: 13px; color: #6B7280; }}
  .main {{ display: grid; grid-template-columns: 1fr 380px; gap: 16px;
           padding: 16px 24px; max-width: 1600px; margin: 0 auto; }}
  .card {{ background: white; border-radius: 8px; border: 1px solid #E5E7EB;
           overflow: hidden; }}
  .card-head {{ padding: 10px 16px; background: #F8F9FA;
               border-bottom: 1px solid #E5E7EB; font-weight: 600; font-size: 14px; }}
  .card-body {{ padding: 12px 16px; }}
  .right {{ display: flex; flex-direction: column; gap: 16px; }}
  .footer {{ text-align: center; padding: 16px; color: #9CA3AF; font-size: 12px; }}
  .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  @media (max-width: 900px) {{
    .main {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Bayesian Network Dashboard</h1>
    <div class="meta">Target: <b>{ev_title}</b> &nbsp;|&nbsp; Evidence ({ev_count}): {ev_desc}</div>
  </div>
</div>
<div class="stats">{stats_html}</div>

<div class="main">
  <div>
    <!-- Network Graph -->
    <div class="card" style="margin-bottom:16px;">
      <div class="card-head">
        Network Structure — target: {ev_title}
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:rgba(220,50,50,0.9);border:2px solid {PALETTE["target_border"]};"></div> Target</div>
          <div class="legend-item"><div class="legend-dot" style="background:rgba(34,139,34,0.85);border:2px solid {PALETTE["evidence_border"]};"></div> Evidence</div>
          <div class="legend-item"><div class="legend-dot" style="background:rgba(180,180,190,0.7);border:2px solid {PALETTE["node_border"]};"></div> Other</div>
        </div>
      </div>
      <div class="card-body" style="padding:4px;">{net_div}</div>
    </div>

    <!-- Sensitivity -->
    <div class="card">
      <div class="card-head">Sensitivity Analysis (Mutual Information) — top {min(top_k, len(analyzer.columns) - 1)}</div>
      <div class="card-body" style="padding:4px;">{sens_div}</div>
    </div>
  </div>

  <div class="right">
    <!-- Evidence -->
    <div class="card">
      <div class="card-head">Evidence Settings ({ev_count})</div>
      <div class="card-body">{evidence_section}</div>
    </div>

    <!-- Markov Blanket -->
    {f'<div class="card"><div class="card-body" style="padding:8px 12px;">{blanket_section}</div></div>' if blanket_section else ''}

    <!-- Posteriors -->
    <div class="card">
      <div class="card-head">Posterior Distributions</div>
      <div class="card-body" style="max-height:500px;overflow-y:auto;">{posterior_section}</div>
    </div>
  </div>
</div>

<div class="footer">Bayesian Network Dashboard — generated by bayesian_network toolkit</div>
</body>
</html>"""

    p = Path(html_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(page, encoding="utf-8")
    logger.info("Dashboard saved: %s", p.resolve())
    return str(p.resolve())
