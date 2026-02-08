"""Core Bayesian Network analysis engine.

Handles data preparation, structure learning, parameter learning,
probabilistic inference, and sensitivity analysis.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
from sklearn.metrics import mutual_info_score

from bayesian_network.config import BNConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# pgmpy compatibility: resolve model class once at import time
# ---------------------------------------------------------------------------
try:
    from pgmpy.models import DiscreteBayesianNetwork as _BNModel
except ImportError:
    from pgmpy.models import BayesianNetwork as _BNModel  # pgmpy < 1.0

try:
    from pgmpy.estimators import ExpertKnowledge as _ExpertKnowledge
except ImportError:
    _ExpertKnowledge = None  # pgmpy < 0.1.24


class AnalyzerError(RuntimeError):
    """Base exception for BayesianAnalyzer errors."""


class ModelNotTrainedError(AnalyzerError):
    """Raised when inference is attempted before training."""


class BayesianAnalyzer:
    """Production-grade Bayesian Network analyser.

    Lifecycle::

        analyzer = BayesianAnalyzer(df, config)
        analyzer.train_model()
        result  = analyzer.query("Target", {"A": "1"})
        mi      = analyzer.compute_sensitivity("Target")

    Parameters
    ----------
    data : pd.DataFrame
        Input data.  All columns are treated as discrete variables.
    config : BNConfig, optional
        Analysis configuration.  Uses defaults when omitted.

    Raises
    ------
    ValueError
        If *data* is empty or has fewer than 2 columns.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, data: pd.DataFrame, config: Optional[BNConfig] = None) -> None:
        if data.empty:
            raise ValueError("Input DataFrame is empty.")
        if data.shape[1] < 2:
            raise ValueError(
                f"At least 2 columns required for a Bayesian Network, "
                f"got {data.shape[1]}."
            )

        self._config = config or BNConfig()
        self._raw_data = data.copy()
        self._data = self._prepare_data(self._raw_data)

        self._model: Optional[_BNModel] = None
        self._inference: Optional[VariableElimination] = None
        self._mi_cache: Dict[str, Dict[str, float]] = {}
        self._edges: List[Tuple[str, str]] = []
        self._training_time: float = 0.0

        logger.info(
            "BayesianAnalyzer initialized: %d rows x %d cols",
            self._data.shape[0],
            self._data.shape[1],
        )

    # -------------------------------------------------------------- properties
    @property
    def config(self) -> BNConfig:
        """Return the immutable configuration."""
        return self._config

    @property
    def columns(self) -> list[str]:
        """Return column names of the prepared data."""
        return list(self._data.columns)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """Return learned edges (empty before training)."""
        return list(self._edges)

    @property
    def is_trained(self) -> bool:
        """Return whether the model has been trained."""
        return self._model is not None

    @property
    def model(self) -> Optional[_BNModel]:
        """Return the underlying pgmpy model (or ``None``)."""
        return self._model

    @property
    def data(self) -> pd.DataFrame:
        """Return the prepared (discretised) data as a copy."""
        return self._data.copy()

    @property
    def training_time(self) -> float:
        """Return wall-clock seconds spent on the last training run."""
        return self._training_time

    # -------------------------------------------------------- data preparation
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and discretise the raw input data.

        Steps:
        1. Replace ``inf`` / ``-inf`` with ``NaN``, then fill with sentinel.
        2. Optionally quantile-bin numeric columns.
        3. Coerce every column to ``category`` or ``str``.
        """
        out = df.copy()

        # 1. inf -> NaN -> sentinel
        out = out.replace([np.inf, -np.inf], np.nan)
        n_missing = int(out.isna().sum().sum())
        if n_missing > 0:
            logger.info("Replacing %d missing/inf values with '%s'", n_missing, self._config.na_token)
        out = out.fillna(self._config.na_token)

        # 2. Optional discretisation
        if self._config.discretize_numeric:
            out = self._discretize_numerics(out)

        # 3. Ensure all columns are discrete-friendly
        for col in out.columns:
            nunique = out[col].nunique(dropna=False)
            if nunique > self._config.max_states_per_col:
                logger.warning(
                    "Column '%s' has %d unique values (> %d); casting to str",
                    col, nunique, self._config.max_states_per_col,
                )
                out[col] = out[col].astype(str)
            else:
                out[col] = out[col].astype("category")

        return out

    def _discretize_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Quantile-bin numeric columns that exceed *max_states_per_col*."""
        num_cols = df.select_dtypes(include=["number"]).columns
        for col in num_cols:
            if df[col].nunique(dropna=False) <= self._config.max_states_per_col:
                continue
            try:
                df[col] = pd.qcut(
                    df[col],
                    q=self._config.discretize_bins,
                    duplicates="drop",
                ).astype(str)
                logger.info("Discretised column '%s' into %d bins", col, self._config.discretize_bins)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "qcut failed for '%s' (%s); falling back to rounded str",
                    col, exc,
                )
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .fillna(0)
                    .round(3)
                    .astype(str)
                )
        return df

    # --------------------------------------------------------------- training
    def train_model(
        self,
        black_list: Optional[Sequence[Tuple[str, str]]] = None,
        white_list: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> "BayesianAnalyzer":
        """Learn structure and parameters from data.

        Parameters
        ----------
        black_list : sequence of (str, str), optional
            Forbidden directed edges.
        white_list : sequence of (str, str), optional
            Required directed edges.

        Returns
        -------
        BayesianAnalyzer
            ``self`` for method chaining.

        Raises
        ------
        AnalyzerError
            If structure learning fails.
        """
        t0 = time.perf_counter()
        self._mi_cache.clear()

        logger.info(
            "Structure learning started (scoring=%s, max_indegree=%d)",
            self._config.scoring_method,
            self._config.max_indegree,
        )

        try:
            dag = self._learn_structure(black_list, white_list)
        except Exception as exc:
            raise AnalyzerError(f"Structure learning failed: {exc}") from exc

        self._edges = list(dag.edges())
        logger.info("Learned %d edges", len(self._edges))

        self._model = _BNModel(self._edges)
        self._model.fit(
            self._data,
            estimator=BayesianEstimator,
            prior_type=self._config.prior_type,
            equivalent_sample_size=self._config.equivalent_sample_size,
        )
        self._inference = VariableElimination(self._model)

        self._training_time = time.perf_counter() - t0
        logger.info("Training completed in %.2f s", self._training_time)

        return self

    def _learn_structure(
        self,
        black_list: Optional[Sequence[Tuple[str, str]]],
        white_list: Optional[Sequence[Tuple[str, str]]],
    ):
        """Run HillClimbSearch with pgmpy version-adaptive kwargs."""
        hc = HillClimbSearch(self._data)

        # Build expert knowledge for new pgmpy API
        expert_knowledge = None
        if _ExpertKnowledge is not None and (black_list or white_list):
            expert_knowledge = _ExpertKnowledge(
                forbidden_edges=list(black_list or []),
                required_edges=list(white_list or []),
            )

        candidate_kwargs: dict[str, Any] = {
            "scoring_method": self._config.scoring_method,
            "max_indegree": self._config.max_indegree,
            "show_progress": self._config.show_progress,
        }
        if expert_knowledge is not None:
            candidate_kwargs["expert_knowledge"] = expert_knowledge
        if self._config.tabu_length > 0:
            candidate_kwargs["tabu_length"] = self._config.tabu_length

        # Filter kwargs to only those accepted by the installed pgmpy version
        sig = inspect.signature(hc.estimate)
        kwargs = {
            k: v for k, v in candidate_kwargs.items()
            if k in sig.parameters and v is not None
        }

        # Legacy API fallback
        if "black_list" in sig.parameters and black_list:
            kwargs["black_list"] = list(black_list)
        if "white_list" in sig.parameters and white_list:
            kwargs["white_list"] = list(white_list)

        return hc.estimate(**kwargs)

    # --------------------------------------------------------------- inference
    def query(
        self,
        target: str,
        evidence: Optional[Dict[str, Any]] = None,
    ):
        """Run probabilistic inference via variable elimination.

        Parameters
        ----------
        target : str
            Query variable name.
        evidence : dict, optional
            Mapping of observed variable names to their values.

        Returns
        -------
        pgmpy.factors.discrete.DiscreteFactor
            Posterior distribution over *target*.

        Raises
        ------
        ModelNotTrainedError
            If called before :meth:`train_model`.
        KeyError
            If *target* or an evidence key is not in the data.
        """
        if self._inference is None:
            raise ModelNotTrainedError("Call train_model() before querying.")

        if target not in self._data.columns:
            raise KeyError(f"Target '{target}' is not in data columns: {self.columns}")

        if evidence:
            unknown = [k for k in evidence if k not in self._data.columns]
            if unknown:
                raise KeyError(f"Unknown evidence columns: {unknown}")

        return self._inference.query(
            variables=[target],
            evidence=evidence or {},
        )

    # --------------------------------------------------- sensitivity analysis
    def compute_sensitivity(self, target: str) -> Dict[str, float]:
        """Compute pairwise mutual information between *target* and all other columns.

        Results are cached per target; call :meth:`invalidate_cache` to clear.

        Parameters
        ----------
        target : str
            The target variable name.

        Returns
        -------
        dict[str, float]
            Column name -> MI score, sorted descending.

        Raises
        ------
        KeyError
            If *target* is not in data columns.
        """
        if target not in self._data.columns:
            raise KeyError(
                f"target '{target}' not in data columns: {self.columns}"
            )

        if target in self._mi_cache:
            logger.debug("MI cache hit for target='%s'", target)
            return dict(self._mi_cache[target])

        logger.info("Computing mutual information for target='%s'", target)
        y = self._data[target].astype(str).to_numpy()

        mi: dict[str, float] = {}
        for col in self._data.columns:
            if col == target:
                continue
            x = self._data[col].astype(str).to_numpy()
            mi[col] = float(mutual_info_score(y, x))

        sorted_mi = dict(sorted(mi.items(), key=lambda kv: kv[1], reverse=True))
        self._mi_cache[target] = sorted_mi
        return dict(sorted_mi)

    # ------------------------------------------------- Markov blanket & bulk query
    def compute_markov_blanket(self, node: str) -> set[str]:
        """Return the Markov blanket of *node*: parents + children + children's other parents.

        Parameters
        ----------
        node : str
            Variable name.

        Returns
        -------
        set[str]
            Set of variable names in the Markov blanket (excludes *node* itself).

        Raises
        ------
        ModelNotTrainedError
            If called before :meth:`train_model`.
        KeyError
            If *node* is not in the model.
        """
        if self._model is None:
            raise ModelNotTrainedError("Call train_model() before computing Markov blanket.")
        if node not in self._data.columns:
            raise KeyError(f"Node '{node}' is not in data columns: {self.columns}")

        import networkx as nx
        g = nx.DiGraph(self._edges)

        parents = set(g.predecessors(node)) if node in g else set()
        children = set(g.successors(node)) if node in g else set()
        co_parents: set[str] = set()
        for child in children:
            co_parents |= set(g.predecessors(child))

        blanket = (parents | children | co_parents) - {node}
        return blanket

    def query_all(
        self,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run inference for every non-evidence variable and return posteriors.

        Parameters
        ----------
        evidence : dict, optional
            Observed variable → value mapping.

        Returns
        -------
        dict[str, DiscreteFactor]
            Variable name → posterior distribution.
        """
        if self._inference is None:
            raise ModelNotTrainedError("Call train_model() before querying.")

        evidence = evidence or {}
        model_nodes = set(self._model.nodes())
        results: Dict[str, Any] = {}
        for col in self._data.columns:
            if col in evidence or col not in model_nodes:
                continue
            try:
                results[col] = self._inference.query(
                    variables=[col], evidence=evidence,
                )
            except Exception as exc:
                logger.warning("query_all: failed for '%s': %s", col, exc)
        return results

    def query_blanket(
        self,
        node: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run inference only for nodes within the Markov blanket of *node*.

        This is significantly faster than :meth:`query_all` for large networks.

        Parameters
        ----------
        node : str
            Centre node whose blanket will be queried.
        evidence : dict, optional
            Observed variable → value mapping.

        Returns
        -------
        dict[str, DiscreteFactor]
            Variable name → posterior distribution (blanket members + *node*).
        """
        if self._inference is None:
            raise ModelNotTrainedError("Call train_model() before querying.")

        evidence = evidence or {}
        blanket = self.compute_markov_blanket(node)
        model_nodes = set(self._model.nodes())
        targets = (blanket | {node}) - set(evidence.keys())

        results: Dict[str, Any] = {}
        for col in targets:
            if col not in model_nodes:
                continue
            try:
                results[col] = self._inference.query(
                    variables=[col], evidence=evidence,
                )
            except Exception as exc:
                logger.warning("query_blanket: failed for '%s': %s", col, exc)
        return results

    def invalidate_cache(self) -> None:
        """Clear the mutual-information cache."""
        self._mi_cache.clear()
        logger.debug("MI cache cleared")

    # ----------------------------------------------------------- serialisation
    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the current model state."""
        return {
            "is_trained": self.is_trained,
            "n_rows": self._data.shape[0],
            "n_columns": self._data.shape[1],
            "columns": self.columns,
            "n_edges": len(self._edges),
            "edges": self._edges[:20],
            "training_time_sec": round(self._training_time, 3),
            "config": {
                "scoring_method": self._config.scoring_method,
                "max_indegree": self._config.max_indegree,
                "prior_type": self._config.prior_type,
                "equivalent_sample_size": self._config.equivalent_sample_size,
            },
        }
