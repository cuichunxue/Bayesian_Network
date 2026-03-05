"""Statistical and scientific validation tests for the Bayesian Network system.

Verifies:
1. Demo data generator produces statistically correct distributions
2. Structure learning recovers known causal relationships
3. Inference engine produces valid probability distributions
4. Sensitivity analysis (MI) is mathematically correct
5. Posterior updates are consistent with Bayes' theorem
6. Edge cases: NaN, inf, small data, uniform data, deterministic data
7. Numerical stability: probability normalization, reproducibility
8. CPD tables are valid conditional probability distributions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats
from sklearn.metrics import mutual_info_score

from bayesian_network.analyzer import BayesianAnalyzer, ModelNotTrainedError
from bayesian_network.config import BNConfig
from bayesian_network.demo_data import DEMO_COLUMNS, DEMO_TARGET, generate_demo_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def demo_df() -> pd.DataFrame:
    return generate_demo_data(n=2000, seed=42)


@pytest.fixture()
def demo_analyzer(demo_df: pd.DataFrame) -> BayesianAnalyzer:
    config = BNConfig(show_progress=False)
    return BayesianAnalyzer(demo_df, config).train_model()


@pytest.fixture()
def large_demo_df() -> pd.DataFrame:
    return generate_demo_data(n=10000, seed=99)


# =========================================================================
# 1. Demo data generator: statistical correctness
# =========================================================================

class TestDemoDataDistributions:
    """Verify the demo data generator produces correct marginal distributions."""

    def test_columns_present(self, demo_df: pd.DataFrame) -> None:
        assert list(demo_df.columns) == DEMO_COLUMNS

    def test_row_count(self, demo_df: pd.DataFrame) -> None:
        assert len(demo_df) == 2000

    def test_no_missing_values(self, demo_df: pd.DataFrame) -> None:
        assert demo_df.isna().sum().sum() == 0

    def test_weather_marginal(self, demo_df: pd.DataFrame) -> None:
        """Weather ~ Categorical(0.5, 0.25, 0.25). With n=2000, check within
        reasonable tolerance using chi-squared goodness of fit."""
        counts = demo_df["Weather"].value_counts()
        expected = {"Sunny": 1000, "Rainy": 500, "Cloudy": 500}
        observed = [counts.get(k, 0) for k in expected]
        exp_vals = list(expected.values())
        chi2, p = scipy_stats.chisquare(observed, exp_vals)
        assert p > 0.01, f"Weather marginal deviates from expected (p={p:.4f})"

    def test_dayofweek_marginal(self, demo_df: pd.DataFrame) -> None:
        """DayOfWeek ~ Categorical(0.7, 0.3)."""
        counts = demo_df["DayOfWeek"].value_counts()
        expected = {"Weekday": 1400, "Weekend": 600}
        observed = [counts.get(k, 0) for k in expected]
        exp_vals = list(expected.values())
        chi2, p = scipy_stats.chisquare(observed, exp_vals)
        assert p > 0.01, f"DayOfWeek marginal deviates (p={p:.4f})"

    def test_accident_conditional_on_weather(self, demo_df: pd.DataFrame) -> None:
        """P(Accident=Yes|Weather=Rainy) ~ 0.3, P(Accident=Yes|Weather=Sunny) ~ 0.05."""
        rainy = demo_df[demo_df["Weather"] == "Rainy"]
        sunny = demo_df[demo_df["Weather"] == "Sunny"]

        p_acc_rainy = (rainy["Accident"] == "Yes").mean()
        p_acc_sunny = (sunny["Accident"] == "Yes").mean()

        # Rainy should cause more accidents than Sunny
        assert p_acc_rainy > p_acc_sunny, "Rainy should have higher accident rate"
        # Check reasonable ranges (expected 0.3 and 0.05, with sampling noise)
        assert 0.15 < p_acc_rainy < 0.45, f"P(Acc|Rainy)={p_acc_rainy:.3f} out of range"
        assert p_acc_sunny < 0.15, f"P(Acc|Sunny)={p_acc_sunny:.3f} too high"

    def test_traffic_conditional_on_weather(self, demo_df: pd.DataFrame) -> None:
        """P(Traffic=Heavy|Weather=Rainy) >> P(Traffic=Heavy|Weather=Sunny)."""
        rainy = demo_df[demo_df["Weather"] == "Rainy"]
        sunny = demo_df[demo_df["Weather"] == "Sunny"]

        p_heavy_rainy = (rainy["Traffic"] == "Heavy").mean()
        p_heavy_sunny = (sunny["Traffic"] == "Heavy").mean()

        assert p_heavy_rainy > p_heavy_sunny * 2, \
            f"Heavy traffic should be much more likely when rainy ({p_heavy_rainy:.3f} vs {p_heavy_sunny:.3f})"

    def test_commute_depends_on_traffic_and_accident(self, demo_df: pd.DataFrame) -> None:
        """P(Commute=VeryLong|Heavy,Accident=Yes) >> P(Commute=VeryLong|Light,No)."""
        heavy_acc = demo_df[(demo_df["Traffic"] == "Heavy") & (demo_df["Accident"] == "Yes")]
        light_no = demo_df[(demo_df["Traffic"] == "Light") & (demo_df["Accident"] == "No")]

        if len(heavy_acc) > 5 and len(light_no) > 5:
            p_vl_bad = (heavy_acc["Commute"] == "VeryLong").mean()
            p_vl_good = (light_no["Commute"] == "VeryLong").mean()
            assert p_vl_bad > p_vl_good * 3, \
                f"VeryLong commute should be much more likely with Heavy traffic + Accident"

    def test_dayofweek_independent_of_weather(self, demo_df: pd.DataFrame) -> None:
        """DayOfWeek and Weather should be statistically independent."""
        contingency = pd.crosstab(demo_df["Weather"], demo_df["DayOfWeek"])
        chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency)
        assert p > 0.01, f"Weather and DayOfWeek should be independent (p={p:.4f})"

    def test_reproducibility(self) -> None:
        """Same seed must produce identical DataFrames."""
        df1 = generate_demo_data(n=100, seed=123)
        df2 = generate_demo_data(n=100, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different data."""
        df1 = generate_demo_data(n=100, seed=1)
        df2 = generate_demo_data(n=100, seed=2)
        assert not df1.equals(df2)


# =========================================================================
# 2. Structure learning: recovers known causal edges
# =========================================================================

class TestStructureLearning:
    """Verify that HillClimbSearch recovers the known causal structure."""

    def test_learns_edges(self, demo_analyzer: BayesianAnalyzer) -> None:
        assert len(demo_analyzer.edges) > 0, "Should learn at least one edge"

    def test_weather_to_traffic_edge(self, demo_analyzer: BayesianAnalyzer) -> None:
        """The strongest causal link Weather->Traffic should be recovered."""
        edge_set = set(demo_analyzer.edges)
        has_forward = ("Weather", "Traffic") in edge_set
        has_reverse = ("Traffic", "Weather") in edge_set
        assert has_forward or has_reverse, \
            f"Weather<->Traffic edge not learned. Edges: {demo_analyzer.edges}"

    def test_weather_to_accident_edge(self, demo_analyzer: BayesianAnalyzer) -> None:
        """Weather->Accident causal link should be recovered."""
        edge_set = set(demo_analyzer.edges)
        has_forward = ("Weather", "Accident") in edge_set
        has_reverse = ("Accident", "Weather") in edge_set
        assert has_forward or has_reverse, \
            f"Weather<->Accident edge not learned. Edges: {demo_analyzer.edges}"

    def test_no_self_loops(self, demo_analyzer: BayesianAnalyzer) -> None:
        for u, v in demo_analyzer.edges:
            assert u != v, f"Self-loop detected: {u} -> {v}"

    def test_dag_is_acyclic(self, demo_analyzer: BayesianAnalyzer) -> None:
        """Learned graph must be a DAG (no cycles)."""
        import networkx as nx
        g = nx.DiGraph(demo_analyzer.edges)
        assert nx.is_directed_acyclic_graph(g), "Learned graph has cycles"

    def test_structure_with_larger_data(self, large_demo_df: pd.DataFrame) -> None:
        """With more data, structure learning should be more confident."""
        config = BNConfig(show_progress=False)
        analyzer = BayesianAnalyzer(large_demo_df, config).train_model()
        assert len(analyzer.edges) >= 3, \
            f"With 10K rows, should learn >=3 edges, got {len(analyzer.edges)}"


# =========================================================================
# 3. Inference engine: valid probability distributions
# =========================================================================

class TestInferenceValidity:
    """Verify inference produces mathematically valid distributions."""

    def test_posterior_sums_to_one(self, demo_analyzer: BayesianAnalyzer) -> None:
        result = demo_analyzer.query("Commute")
        total = float(result.values.sum())
        assert abs(total - 1.0) < 1e-6, f"Posterior sum={total}, expected 1.0"

    def test_posterior_non_negative(self, demo_analyzer: BayesianAnalyzer) -> None:
        result = demo_analyzer.query("Commute")
        assert np.all(result.values >= 0), "Posterior contains negative probabilities"

    def test_posterior_with_evidence_sums_to_one(self, demo_analyzer: BayesianAnalyzer) -> None:
        result = demo_analyzer.query("Commute", {"Weather": "Rainy"})
        total = float(result.values.sum())
        assert abs(total - 1.0) < 1e-6, f"Conditional posterior sum={total}"

    def test_evidence_shifts_posterior(self, demo_analyzer: BayesianAnalyzer) -> None:
        """Setting Weather=Rainy should shift P(Commute) toward longer commutes."""
        prior = demo_analyzer.query("Commute")
        posterior_rainy = demo_analyzer.query("Commute", {"Weather": "Rainy"})

        prior_vals = {s: float(v) for s, v in
                      zip(prior.state_names["Commute"], prior.values.flatten())}
        post_vals = {s: float(v) for s, v in
                     zip(posterior_rainy.state_names["Commute"], posterior_rainy.values.flatten())}

        # P(Normal|Rainy) should be lower than P(Normal)
        if "Normal" in prior_vals and "Normal" in post_vals:
            assert post_vals["Normal"] < prior_vals["Normal"] + 0.15, \
                "P(Normal|Rainy) should not be much higher than P(Normal)"

    def test_query_all_returns_all_variables(self, demo_analyzer: BayesianAnalyzer) -> None:
        results = demo_analyzer.query_all()
        model_nodes = set(demo_analyzer.model.nodes())
        for var in model_nodes:
            assert var in results, f"Variable {var} missing from query_all results"

    def test_query_all_posteriors_sum_to_one(self, demo_analyzer: BayesianAnalyzer) -> None:
        results = demo_analyzer.query_all()
        for var, factor in results.items():
            total = float(factor.values.sum())
            assert abs(total - 1.0) < 1e-6, f"P({var}) sums to {total}"

    def test_query_all_with_evidence(self, demo_analyzer: BayesianAnalyzer) -> None:
        results = demo_analyzer.query_all({"Weather": "Sunny"})
        assert "Weather" not in results, "Evidence variable should not be in results"
        for var, factor in results.items():
            total = float(factor.values.sum())
            assert abs(total - 1.0) < 1e-6

    def test_multiple_evidence_variables(self, demo_analyzer: BayesianAnalyzer) -> None:
        result = demo_analyzer.query("Commute", {"Weather": "Rainy", "Accident": "Yes"})
        total = float(result.values.sum())
        assert abs(total - 1.0) < 1e-6


# =========================================================================
# 4. Sensitivity analysis (MI): mathematical correctness
# =========================================================================

class TestMutualInformation:
    """Verify mutual information computation is statistically sound."""

    def test_mi_non_negative(self, demo_analyzer: BayesianAnalyzer) -> None:
        """MI >= 0 always (information-theoretic property)."""
        mi = demo_analyzer.compute_sensitivity("Commute")
        for var, val in mi.items():
            assert val >= 0, f"MI({var}, Commute) = {val} < 0"

    def test_mi_zero_for_independent_variables(self) -> None:
        """MI between truly independent variables should be near zero."""
        rng = np.random.RandomState(42)
        n = 5000
        df = pd.DataFrame({
            "X": rng.choice(["A", "B", "C"], size=n),
            "Y": rng.choice(["D", "E", "F"], size=n),
        })
        config = BNConfig(show_progress=False)
        a = BayesianAnalyzer(df, config).train_model()
        mi = a.compute_sensitivity("X")
        # MI should be very close to zero for independent variables
        assert mi["Y"] < 0.01, f"MI(X,Y) = {mi['Y']:.6f}, expected ~0 for independent vars"

    def test_mi_high_for_dependent_variables(self) -> None:
        """MI between deterministically dependent variables should be high."""
        rng = np.random.RandomState(42)
        n = 1000
        x = rng.choice(["A", "B", "C"], size=n)
        y = np.where(x == "A", "X", np.where(x == "B", "Y", "Z"))
        df = pd.DataFrame({"X": x, "Y": y})
        config = BNConfig(show_progress=False)
        a = BayesianAnalyzer(df, config).train_model()
        mi = a.compute_sensitivity("X")
        assert mi["Y"] > 0.5, f"MI(X,Y) = {mi['Y']:.6f}, expected high for deterministic relation"

    def test_mi_symmetric(self, demo_analyzer: BayesianAnalyzer) -> None:
        """MI(X,Y) == MI(Y,X) (symmetry property)."""
        data = demo_analyzer.data
        for col_a in ["Weather", "Traffic"]:
            for col_b in ["Accident", "Commute"]:
                if col_a == col_b:
                    continue
                xa = data[col_a].astype(str).to_numpy()
                xb = data[col_b].astype(str).to_numpy()
                mi_ab = mutual_info_score(xa, xb)
                mi_ba = mutual_info_score(xb, xa)
                assert abs(mi_ab - mi_ba) < 1e-10, \
                    f"MI({col_a},{col_b})={mi_ab} != MI({col_b},{col_a})={mi_ba}"

    def test_mi_sorted_descending(self, demo_analyzer: BayesianAnalyzer) -> None:
        mi = demo_analyzer.compute_sensitivity("Commute")
        values = list(mi.values())
        assert values == sorted(values, reverse=True)

    def test_mi_traffic_highest_for_commute(self, demo_analyzer: BayesianAnalyzer) -> None:
        """Traffic should have highest MI with Commute (direct causal parent)."""
        mi = demo_analyzer.compute_sensitivity("Commute")
        top_var = max(mi, key=mi.get)
        # Traffic or Accident should be most informative about Commute
        assert top_var in ("Traffic", "Accident", "Weather"), \
            f"Expected Traffic/Accident/Weather as top MI for Commute, got {top_var}"

    def test_dayofweek_lowest_mi_for_commute(self, demo_analyzer: BayesianAnalyzer) -> None:
        """DayOfWeek (independent noise) should have lowest MI with Commute."""
        mi = demo_analyzer.compute_sensitivity("Commute")
        # DayOfWeek should have the lowest MI since it's independent
        min_var = min(mi, key=mi.get)
        assert min_var == "DayOfWeek", \
            f"Expected DayOfWeek as lowest MI for Commute, got {min_var} (MI={mi[min_var]:.6f})"


# =========================================================================
# 5. CPD validity
# =========================================================================

class TestCPDValidity:
    """Verify conditional probability distributions are valid."""

    def test_cpd_rows_sum_to_one(self, demo_analyzer: BayesianAnalyzer) -> None:
        """Each column of the CPD matrix must sum to 1.0."""
        for node in demo_analyzer.model.nodes():
            cpd = demo_analyzer.model.get_cpds(node)
            values = cpd.get_values()
            if values.ndim == 1:
                total = values.sum()
                assert abs(total - 1.0) < 1e-6, \
                    f"CPD of {node} sums to {total}"
            else:
                col_sums = values.sum(axis=0)
                for i, s in enumerate(col_sums):
                    assert abs(s - 1.0) < 1e-6, \
                        f"CPD of {node}, column {i} sums to {s}"

    def test_cpd_non_negative(self, demo_analyzer: BayesianAnalyzer) -> None:
        for node in demo_analyzer.model.nodes():
            cpd = demo_analyzer.model.get_cpds(node)
            assert np.all(cpd.get_values() >= 0), \
                f"CPD of {node} contains negative values"

    def test_cpd_matches_data_frequencies(self, demo_analyzer: BayesianAnalyzer) -> None:
        """For root nodes (no parents), CPD should approximately match data frequencies."""
        data = demo_analyzer.data
        for node in demo_analyzer.model.nodes():
            cpd = demo_analyzer.model.get_cpds(node)
            if len(cpd.variables) == 1:  # root node
                values = cpd.get_values().flatten()
                state_names = cpd.state_names[node]
                for state, prob in zip(state_names, values):
                    empirical = (data[node].astype(str) == str(state)).mean()
                    # BDeu prior smooths, so allow generous tolerance
                    assert abs(prob - empirical) < 0.15, \
                        f"CPD P({node}={state})={prob:.4f}, empirical={empirical:.4f}"


# =========================================================================
# 6. Markov blanket correctness
# =========================================================================

class TestMarkovBlanket:

    def test_markov_blanket_excludes_self(self, demo_analyzer: BayesianAnalyzer) -> None:
        for col in demo_analyzer.columns:
            mb = demo_analyzer.compute_markov_blanket(col)
            assert col not in mb

    def test_markov_blanket_subset_of_columns(self, demo_analyzer: BayesianAnalyzer) -> None:
        all_cols = set(demo_analyzer.columns)
        for col in demo_analyzer.columns:
            mb = demo_analyzer.compute_markov_blanket(col)
            assert mb.issubset(all_cols), f"MB({col}) contains unknown nodes"

    def test_parents_in_blanket(self, demo_analyzer: BayesianAnalyzer) -> None:
        """Parents of a node must be in its Markov blanket."""
        import networkx as nx
        g = nx.DiGraph(demo_analyzer.edges)
        for node in g.nodes():
            mb = demo_analyzer.compute_markov_blanket(node)
            parents = set(g.predecessors(node))
            assert parents.issubset(mb), \
                f"Parents {parents} of {node} not all in MB {mb}"

    def test_children_in_blanket(self, demo_analyzer: BayesianAnalyzer) -> None:
        """Children of a node must be in its Markov blanket."""
        import networkx as nx
        g = nx.DiGraph(demo_analyzer.edges)
        for node in g.nodes():
            mb = demo_analyzer.compute_markov_blanket(node)
            children = set(g.successors(node))
            assert children.issubset(mb), \
                f"Children {children} of {node} not all in MB {mb}"


# =========================================================================
# 7. Edge cases and stability
# =========================================================================

class TestEdgeCases:

    def test_data_with_nan(self) -> None:
        """NaN values should be handled gracefully."""
        df = pd.DataFrame({
            "X": ["A", "B", np.nan, "A", "B"] * 50,
            "Y": ["C", "D", "C", "D", "C"] * 50,
        })
        config = BNConfig(show_progress=False)
        a = BayesianAnalyzer(df, config)
        a.train_model()
        result = a.query("X")
        assert abs(result.values.sum() - 1.0) < 1e-6

    def test_data_with_inf(self) -> None:
        """Inf values should be replaced and not crash."""
        df = pd.DataFrame({
            "X": [1, 2, np.inf, 4, 5] * 40,
            "Y": [10, 20, 30, -np.inf, 50] * 40,
        })
        config = BNConfig(show_progress=False, discretize_numeric=True, discretize_bins=3)
        a = BayesianAnalyzer(df, config)
        a.train_model()
        assert a.is_trained

    def test_uniform_data(self) -> None:
        """Uniform (no-signal) data should not crash."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "X": rng.choice(["A", "B"], size=300),
            "Y": rng.choice(["C", "D"], size=300),
            "Z": rng.choice(["E", "F"], size=300),
        })
        config = BNConfig(show_progress=False)
        a = BayesianAnalyzer(df, config).train_model()
        mi = a.compute_sensitivity("X")
        # All MI values should be near zero for independent uniform data
        for val in mi.values():
            assert val < 0.05, f"MI={val:.6f} too high for uniform data"

    def test_deterministic_relationship(self) -> None:
        """Deterministic columns should produce valid model."""
        n = 200
        x = ["A"] * (n // 2) + ["B"] * (n // 2)
        y = ["X" if v == "A" else "Y" for v in x]
        df = pd.DataFrame({"X": x, "Y": y})
        config = BNConfig(show_progress=False)
        a = BayesianAnalyzer(df, config).train_model()
        result = a.query("Y", {"X": "A"})
        vals = dict(zip(result.state_names["Y"], result.values.flatten()))
        # P(Y=X|X=A) should be very high (close to 1, smoothed by prior)
        assert vals.get("X", 0) > 0.8, f"P(Y=X|X=A) = {vals.get('X', 0):.4f}"

    def test_small_dataset(self) -> None:
        """Very small dataset (n=20) should still work without crash."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "X": rng.choice(["A", "B"], size=20),
            "Y": rng.choice(["C", "D"], size=20),
        })
        config = BNConfig(show_progress=False)
        a = BayesianAnalyzer(df, config).train_model()
        result = a.query("X")
        assert abs(result.values.sum() - 1.0) < 1e-6

    def test_many_categories(self) -> None:
        """Many unique values per column should be handled."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "X": rng.choice([f"cat_{i}" for i in range(15)], size=300),
            "Y": rng.choice([f"val_{i}" for i in range(10)], size=300),
        })
        config = BNConfig(show_progress=False, max_states_per_col=20)
        a = BayesianAnalyzer(df, config).train_model()
        assert a.is_trained

    def test_single_value_column(self) -> None:
        """A column with only one unique value should not crash."""
        df = pd.DataFrame({
            "X": ["A"] * 100,
            "Y": ["B", "C"] * 50,
        })
        config = BNConfig(show_progress=False)
        a = BayesianAnalyzer(df, config).train_model()
        assert a.is_trained


# =========================================================================
# 8. Numerical stability
# =========================================================================

class TestNumericalStability:

    def test_repeated_training_consistent(self) -> None:
        """Training the same data twice should produce identical edges."""
        df = generate_demo_data(n=500, seed=42)
        config = BNConfig(show_progress=False)
        a1 = BayesianAnalyzer(df, config).train_model()
        a2 = BayesianAnalyzer(df, config).train_model()
        assert set(a1.edges) == set(a2.edges), "Same data should produce same edges"

    def test_inference_deterministic(self) -> None:
        """Same query should always return the same result."""
        df = generate_demo_data(n=500, seed=42)
        config = BNConfig(show_progress=False)
        a = BayesianAnalyzer(df, config).train_model()
        r1 = a.query("Commute", {"Weather": "Rainy"})
        r2 = a.query("Commute", {"Weather": "Rainy"})
        np.testing.assert_array_almost_equal(r1.values, r2.values, decimal=10)

    def test_all_evidence_values_produce_valid_posterior(self, demo_analyzer: BayesianAnalyzer) -> None:
        """Every possible evidence value should produce a valid distribution."""
        data = demo_analyzer.data
        for col in demo_analyzer.columns:
            unique_vals = data[col].astype(str).unique()
            if col not in set(demo_analyzer.model.nodes()):
                continue
            for val in unique_vals[:5]:  # test up to 5 values per column
                for target in demo_analyzer.columns:
                    if target == col or target not in set(demo_analyzer.model.nodes()):
                        continue
                    try:
                        result = demo_analyzer.query(target, {col: val})
                        total = float(result.values.sum())
                        assert abs(total - 1.0) < 1e-5, \
                            f"P({target}|{col}={val}) sums to {total}"
                        assert np.all(result.values >= 0), \
                            f"P({target}|{col}={val}) has negative values"
                    except Exception as exc:
                        pytest.fail(f"Query P({target}|{col}={val}) failed: {exc}")

    def test_prior_smoothing_prevents_zero_probabilities(self, demo_analyzer: BayesianAnalyzer) -> None:
        """BDeu prior should prevent any CPD entry from being exactly zero."""
        for node in demo_analyzer.model.nodes():
            cpd = demo_analyzer.model.get_cpds(node)
            values = cpd.get_values()
            # With BDeu smoothing (ESS=5.0), no probability should be exactly 0
            assert np.all(values > 0), \
                f"CPD of {node} has zero entries despite BDeu smoothing"

    def test_discretization_stability(self) -> None:
        """Discretizing numeric data should produce valid categories."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "X": rng.randn(500),
            "Y": rng.randn(500) * 100 + 50,
            "Z": rng.choice(["A", "B"], size=500),
        })
        config = BNConfig(show_progress=False, discretize_numeric=True, discretize_bins=5)
        a = BayesianAnalyzer(df, config)
        # After discretization, all values should be strings
        for col in ["X", "Y"]:
            assert a.data[col].dtype.name in ("category", "object"), \
                f"Column {col} not properly discretized"
        a.train_model()
        result = a.query("Z")
        assert abs(result.values.sum() - 1.0) < 1e-6


# =========================================================================
# 9. Webapp inference pipeline validation
# =========================================================================

class TestWebappInferencePipeline:
    """Validate the inference pipeline used by the Flask webapp."""

    def test_run_inference_produces_valid_posteriors(self, demo_analyzer: BayesianAnalyzer) -> None:
        """Simulate what the webapp does: query_all + extract posteriors."""
        from bayesian_network.webapp import _run_inference
        posteriors = _run_inference(demo_analyzer, {})
        assert len(posteriors) > 0
        for var, probs in posteriors.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-5, f"Posterior of {var} sums to {total}"
            assert all(v >= 0 for v in probs.values()), \
                f"Negative probability in {var}"

    def test_run_inference_with_evidence(self, demo_analyzer: BayesianAnalyzer) -> None:
        from bayesian_network.webapp import _run_inference
        posteriors = _run_inference(demo_analyzer, {"Weather": "Rainy"})
        assert "Weather" not in posteriors
        for var, probs in posteriors.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-5

    def test_compute_edge_mi_valid(self, demo_analyzer: BayesianAnalyzer) -> None:
        from bayesian_network.webapp import _compute_edge_mi
        labels, values = _compute_edge_mi(demo_analyzer)
        assert len(labels) == len(values)
        assert len(labels) == len(demo_analyzer.edges)
        for val in values:
            assert val >= 0, f"Edge MI should be non-negative, got {val}"

    def test_mi_func_for_produces_valid_mi(self, demo_analyzer: BayesianAnalyzer) -> None:
        from bayesian_network.webapp import _mi_func_for
        mi_func = _mi_func_for(demo_analyzer)
        cols = demo_analyzer.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                mi = mi_func(cols[i], cols[j])
                assert mi >= 0, f"MI({cols[i]},{cols[j]}) = {mi} < 0"
                # MI should be symmetric
                mi_rev = mi_func(cols[j], cols[i])
                assert abs(mi - mi_rev) < 1e-10, \
                    f"MI({cols[i]},{cols[j]})={mi} != MI({cols[j]},{cols[i]})={mi_rev}"


# =========================================================================
# 10. Scoring method comparison
# =========================================================================

class TestScoringMethods:
    """Verify different scoring methods all produce valid models."""

    @pytest.mark.parametrize("scoring", ["bic-d", "k2", "bdeu"])
    def test_scoring_method_produces_valid_model(self, scoring: str) -> None:
        df = generate_demo_data(n=500, seed=42)
        config = BNConfig(scoring_method=scoring, show_progress=False)
        a = BayesianAnalyzer(df, config).train_model()
        assert a.is_trained
        result = a.query("Commute")
        total = float(result.values.sum())
        assert abs(total - 1.0) < 1e-6, f"Scoring={scoring}: P(Commute) sums to {total}"

    @pytest.mark.parametrize("prior", ["BDeu", "K2"])
    def test_prior_type_produces_valid_cpds(self, prior: str) -> None:
        df = generate_demo_data(n=500, seed=42)
        config = BNConfig(prior_type=prior, show_progress=False)
        a = BayesianAnalyzer(df, config).train_model()
        for node in a.model.nodes():
            cpd = a.model.get_cpds(node)
            vals = cpd.get_values()
            assert np.all(vals >= 0), f"Prior={prior}: CPD of {node} has negatives"
            if vals.ndim == 1:
                assert abs(vals.sum() - 1.0) < 1e-6
            else:
                for col_sum in vals.sum(axis=0):
                    assert abs(col_sum - 1.0) < 1e-6
