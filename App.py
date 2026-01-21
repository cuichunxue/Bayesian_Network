# =============================================================================
# Robust Bayesian Network Analyzer (pgmpy 0.1.x 〜 1.x 対応)  [FULL WORKING]
# - Structure learning: HillClimbSearch (scoring_method="bic-d")
# - Constraints: black_list/white_list -> ExpertKnowledge (new API) / old API fallback
# - Parameter learning: BayesianEstimator (BDeu)
# - Inference: VariableElimination
# - Sensitivity: Mutual Information (MI)
# - Visualization: Plotly + NetworkX
# - Debug output: learned edges / node count / MI top
# - HTML export option (bn.html) for environments where fig.show() doesn't render
# =============================================================================

import inspect
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics import mutual_info_score

from pgmpy.estimators import HillClimbSearch, BayesianEstimator, ExpertKnowledge
from pgmpy.inference import VariableElimination

# pgmpy 1.x: DiscreteBayesianNetwork / 0.1系: BayesianNetwork
try:
    from pgmpy.models import DiscreteBayesianNetwork as BNClass  # pgmpy 1.x
except Exception:
    from pgmpy.models import BayesianNetwork as BNClass          # pgmpy 0.1.x


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class BNConfig:
    # Structure learning
    scoring_method: str = "bic-d"     # もし環境でエラーなら "bic" を試す
    max_indegree: int = 3
    tabu_length: int = 0             # 0: 使わない（estimateのデフォルト）
    show_progress: bool = True

    # Parameter learning
    prior_type: str = "BDeu"
    equivalent_sample_size: float = 5.0

    # Data handling
    na_token: str = "__MISSING__"
    max_states_per_col: int = 30     # 状態数が多い列は str 扱い
    discretize_numeric: bool = False
    discretize_bins: int = 5

    # Visualization
    node_size_min: float = 18.0
    node_size_max: float = 45.0
    default_node_size: float = 28.0


# -----------------------------------------------------------------------------
# Analyzer
# -----------------------------------------------------------------------------
class BayesianAnalyzer:
    def __init__(self, data: pd.DataFrame, config: Optional[BNConfig] = None):
        self.config = config or BNConfig()
        self.raw_data = data.copy()
        self.data = self._prepare_data(self.raw_data)

        self.model = None
        self.inference = None
        self._mi_cache: Dict[str, float] = {}

    # ---------------------------
    # Data preparation
    # ---------------------------
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # inf -> NaN, NaN -> token
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(self.config.na_token)

        # Optional: discretize numeric columns with many unique values
        if self.config.discretize_numeric:
            num_cols = df.select_dtypes(include=["number"]).columns
            for c in num_cols:
                nunique = df[c].nunique(dropna=False)
                if nunique <= self.config.max_states_per_col:
                    continue
                try:
                    df[c] = pd.qcut(df[c], q=self.config.discretize_bins, duplicates="drop").astype(str)
                except Exception:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round(3).astype(str)

        # Make columns discrete-friendly
        for c in df.columns:
            if df[c].nunique(dropna=False) > self.config.max_states_per_col:
                df[c] = df[c].astype(str)
            else:
                df[c] = df[c].astype("category")

        return df

    # ---------------------------
    # Training
    # ---------------------------
    def train_model(self,
                    black_list: Optional[List[Tuple[str, str]]] = None,
                    white_list: Optional[List[Tuple[str, str]]] = None):
        print(f"Starting Structure Learning ({self.config.scoring_method})...")

        hc = HillClimbSearch(self.data)

        # New API path: ExpertKnowledge
        expert_knowledge = None
        if black_list or white_list:
            expert_knowledge = ExpertKnowledge(
                forbidden_edges=black_list or [],
                required_edges=white_list or []
            )

        # Candidate kwargs (will be filtered by signature)
        candidate_kwargs = dict(
            scoring_method=self.config.scoring_method,
            max_indegree=self.config.max_indegree,
            expert_knowledge=expert_knowledge,
            show_progress=self.config.show_progress,
        )
        if self.config.tabu_length and self.config.tabu_length > 0:
            candidate_kwargs["tabu_length"] = int(self.config.tabu_length)

        sig = inspect.signature(hc.estimate)
        kwargs = {k: v for k, v in candidate_kwargs.items()
                  if k in sig.parameters and v is not None}

        # Old API fallback if supported
        if "black_list" in sig.parameters:
            kwargs["black_list"] = black_list
        if "white_list" in sig.parameters:
            kwargs["white_list"] = white_list

        best_dag = hc.estimate(**kwargs)

        self.model = BNClass(best_dag.edges())
        self.model.fit(
            self.data,
            estimator=BayesianEstimator,
            prior_type=self.config.prior_type,
            equivalent_sample_size=self.config.equivalent_sample_size
        )

        self.inference = VariableElimination(self.model)

        # Debug prints (always visible)
        edges = list(self.model.edges())
        print("Model training completed.")
        print(f"Learned edges: {len(edges)}")
        if edges:
            print("Edge samples:", edges[:20])
        else:
            print("No edges learned (BIC may prefer empty graph).")
        return self

    # ---------------------------
    # Inference
    # ---------------------------
    def get_probability(self, target: str, evidence: Dict[str, Any]):
        if self.inference is None:
            raise RuntimeError("Model not trained. Call train_model() first.")
        unknown = [k for k in evidence.keys() if k not in self.data.columns]
        if unknown:
            raise KeyError(f"Evidence has unknown columns: {unknown}")
        return self.inference.query(variables=[target], evidence=evidence)

    # ---------------------------
    # Sensitivity (Mutual Information)
    # ---------------------------
    def compute_sensitivity_mi(self, target_node: str) -> Dict[str, float]:
        if target_node not in self.data.columns:
            raise KeyError(f"target_node '{target_node}' not in data columns.")

        y = self.data[target_node].astype(str).to_numpy()
        mi = {}
        for n in self.data.columns:
            if n == target_node:
                continue
            x = self.data[n].astype(str).to_numpy()
            mi[n] = float(mutual_info_score(y, x))

        self._mi_cache = dict(sorted(mi.items(), key=lambda kv: kv[1], reverse=True))
        return self._mi_cache

    def plot_sensitivity(self, target_node: str, topk: int = 30, html_path: Optional[str] = None):
        mi = self._mi_cache or self.compute_sensitivity_mi(target_node)
        items = list(mi.items())[:topk]
        x = [k for k, _ in items]
        y = [v for _, v in items]

        print(f"Top-{min(topk, len(items))} MI variables for '{target_node}':")
        for k, v in items[:10]:
            print(f"  {k}: {v:.6f}")

        fig = go.Figure([go.Bar(x=x, y=y)])
        fig.update_layout(
            title=f"Sensitivity (Mutual Information): {target_node}",
            template="plotly_white",
            xaxis_title="Variable",
            yaxis_title="MI"
        )
        fig.show()

        if html_path:
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"Saved sensitivity HTML: {html_path}")

        return fig

    # ---------------------------
    # Visualization (SAFE)
    # ---------------------------
    def visualize_network(self,
                          target_node: Optional[str] = None,
                          show_mi_on_nodes: bool = True,
                          arrows: bool = True,
                          html_path: Optional[str] = None):
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_model() first.")

        learned_edges = list(self.model.edges())

        # Always include all nodes from columns (even if edges=0)
        g = nx.DiGraph()
        g.add_nodes_from(list(self.data.columns))
        g.add_edges_from(learned_edges)

        if g.number_of_nodes() == 0:
            raise RuntimeError("No nodes to visualize. Check input data columns.")

        pos = nx.spring_layout(g, k=1.0, seed=42)

        # MI -> node_strength
        node_strength: Dict[str, float] = {}
        if show_mi_on_nodes and target_node is not None and target_node in self.data.columns:
            mi = self._mi_cache or self.compute_sensitivity_mi(target_node)
            for n in g.nodes():
                if n == target_node:
                    continue
                node_strength[n] = mi.get(n, 0.0)
            max_mi = max(node_strength.values()) if node_strength else 0.0
            node_strength[target_node] = (max_mi if max_mi > 0 else 1.0) * 1.2

        # edges
        edge_x, edge_y = [], []
        for u, v in g.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        # nodes
        nodes = list(g.nodes())
        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]

        # size + hover (robust)
        if node_strength:
            raw = np.array([node_strength.get(n, 0.0) for n in nodes], dtype=float)
            raw_max = float(raw.max()) if raw.size > 0 else 0.0
            if raw_max > 0:
                sizes = (
                    self.config.node_size_min
                    + (raw / raw_max) * (self.config.node_size_max - self.config.node_size_min)
                ).tolist()
            else:
                sizes = [self.config.default_node_size] * len(nodes)

            hover = [
                f"{n}<br>MI={node_strength.get(n, 0.0):.6f}"
                + (f"<br>(target={target_node})" if target_node else "")
                for n in nodes
            ]
        else:
            sizes = [self.config.default_node_size] * len(nodes)
            hover = [str(n) for n in nodes]

        fig = go.Figure()

        # Edge trace (empty ok)
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines",
            name="edges"
        ))

        # Node trace
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            hovertext=hover,
            hoverinfo="text",
            marker=dict(size=sizes, line_width=2),
            name="nodes"
        ))

        # Arrows (annotations)
        annotations = []
        if arrows and g.number_of_edges() > 0:
            for u, v in g.edges():
                annotations.append(dict(
                    ax=pos[u][0], ay=pos[u][1],
                    x=pos[v][0], y=pos[v][1],
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowwidth=1.2, opacity=0.85
                ))

        title = "Bayesian Network Structure"
        if g.number_of_edges() == 0:
            title += " (No edges learned: empty graph selected)"

        fig.update_layout(
            title=title,
            showlegend=False,
            annotations=annotations,
            template="plotly_white",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=60, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        # Always show some debug text output too
        print(f"Visualize: nodes={g.number_of_nodes()}, edges={g.number_of_edges()}")

        fig.show()

        if html_path:
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"Saved network HTML: {html_path}")

        return fig


# =============================================================================
# Run example
# =============================================================================
if __name__ == "__main__":
    np.random.seed(0)

    # 「エッジが出やすい」ダミーデータ（XORではなく、素直な依存にする）
    n = 1000
    A = np.random.randint(0, 2, size=n)
    B = np.random.randint(0, 2, size=n)
    C = np.random.randint(0, 2, size=n)

    # Target depends on A and B (OR)
    Target = ((A + B) >= 1).astype(int)

    df = pd.DataFrame({"A": A, "B": B, "C": C, "Target": Target})

    analyzer = BayesianAnalyzer(df, config=BNConfig(
        scoring_method="bic-d",
        max_indegree=3,
        prior_type="BDeu",
        equivalent_sample_size=5.0,
        discretize_numeric=False,
        show_progress=True
    ))

    analyzer.train_model()

    # If fig.show() doesn't render in your environment, set html_path="bn.html"
    analyzer.visualize_network(
        target_node="Target",
        show_mi_on_nodes=True,
        arrows=True,
        html_path=None  # e.g. "bn.html"
    )

    analyzer.plot_sensitivity(
        "Target",
        topk=10,
        html_path=None  # e.g. "sensitivity.html"
    )
