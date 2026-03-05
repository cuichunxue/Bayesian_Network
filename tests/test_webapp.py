"""Tests for the Flask web application."""

from __future__ import annotations

import io

import pytest

from bayesian_network.webapp import create_app


@pytest.fixture()
def client():
    """Create a Flask test client."""
    app = create_app(secret_key="test-secret")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture()
def _sample_csv() -> str:
    """Return CSV text with clear dependencies."""
    return (
        "Weather,Traffic,Accident,Commute\n"
        + "\n".join(
            f"{'Sunny' if i % 3 == 0 else 'Rainy' if i % 3 == 1 else 'Cloudy'},"
            f"{'Heavy' if i % 2 == 0 else 'Light'},"
            f"{'Yes' if i % 5 == 0 else 'No'},"
            f"{'Long' if i % 4 == 0 else 'Normal'}"
            for i in range(200)
        )
    )


class TestIndexPage:
    def test_get_index(self, client) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Bayesian Network Analyzer" in resp.data

    def test_index_has_form(self, client) -> None:
        resp = client.get("/")
        assert b"<form" in resp.data
        assert b"csvfile" in resp.data

    def test_index_has_demo_button(self, client) -> None:
        resp = client.get("/")
        assert b"demo" in resp.data.lower()
        assert b"Demo Data" in resp.data

    def test_index_has_japanese_text(self, client) -> None:
        resp = client.get("/")
        html = resp.data.decode("utf-8")
        assert "データ入力" in html
        assert "分析開始" in html

    def test_index_has_getting_started_guide(self, client) -> None:
        resp = client.get("/")
        html = resp.data.decode("utf-8")
        assert "Getting Started" in html or "はじめて" in html


class TestDemoEndpoint:
    def test_demo_creates_session(self, client) -> None:
        resp = client.post("/demo")
        assert resp.status_code == 302
        assert "/dashboard/" in resp.headers["Location"]

    def test_demo_dashboard_loads(self, client) -> None:
        resp = client.post("/demo", follow_redirects=True)
        assert resp.status_code == 200
        assert b"Dashboard" in resp.data or b"dashboard" in resp.data

    def test_demo_shows_network(self, client) -> None:
        resp = client.post("/demo", follow_redirects=True)
        assert resp.status_code == 200
        assert b"Network" in resp.data


class TestDemoCsvApi:
    def test_api_returns_json(self, client) -> None:
        resp = client.get("/api/demo-csv")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "csv" in data
        assert "columns" in data
        assert "target_node" in data

    def test_api_csv_has_headers(self, client) -> None:
        resp = client.get("/api/demo-csv")
        data = resp.get_json()
        assert "Weather" in data["csv"]
        assert "Commute" in data["csv"]

    def test_api_columns_list(self, client) -> None:
        resp = client.get("/api/demo-csv")
        data = resp.get_json()
        assert isinstance(data["columns"], list)
        assert len(data["columns"]) >= 2

    def test_api_target_node(self, client) -> None:
        resp = client.get("/api/demo-csv")
        data = resp.get_json()
        assert data["target_node"] == "Commute"


class TestAnalyzeEndpoint:
    def test_no_data_redirects(self, client) -> None:
        resp = client.post("/analyze", data={}, follow_redirects=True)
        assert resp.status_code == 200
        assert b"upload" in resp.data.lower() or b"paste" in resp.data.lower()

    def test_csv_paste(self, client, _sample_csv: str) -> None:
        resp = client.post("/analyze", data={
            "csvtext": _sample_csv,
            "scoring_method": "bic-d",
            "max_indegree": "3",
            "prior_type": "BDeu",
            "equivalent_sample_size": "5.0",
            "discretize_bins": "5",
            "target_node": "Commute",
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"Dashboard" in resp.data or b"dashboard" in resp.data

    def test_csv_file_upload(self, client, _sample_csv: str) -> None:
        data = {
            "csvfile": (io.BytesIO(_sample_csv.encode()), "test.csv"),
            "scoring_method": "bic-d",
            "max_indegree": "3",
            "prior_type": "BDeu",
            "equivalent_sample_size": "5.0",
            "discretize_bins": "5",
        }
        resp = client.post(
            "/analyze",
            data=data,
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200

    def test_invalid_csv_shows_error(self, client) -> None:
        resp = client.post("/analyze", data={
            "csvtext": "not valid csv \x00\x01\x02",
            "scoring_method": "bic-d",
            "max_indegree": "3",
            "prior_type": "BDeu",
            "equivalent_sample_size": "5.0",
            "discretize_bins": "5",
        }, follow_redirects=True)
        assert resp.status_code == 200


class TestDashboardEndpoint:
    def test_invalid_session_redirects(self, client) -> None:
        resp = client.get("/dashboard/nonexistent", follow_redirects=True)
        assert resp.status_code == 200
        # Should redirect to index
        assert b"Bayesian Network Analyzer" in resp.data

    def test_dashboard_after_analyze(self, client, _sample_csv: str) -> None:
        # First analyze
        resp = client.post("/analyze", data={
            "csvtext": _sample_csv,
            "scoring_method": "bic-d",
            "max_indegree": "3",
            "prior_type": "BDeu",
            "equivalent_sample_size": "5.0",
            "discretize_bins": "5",
            "target_node": "Commute",
        })
        # Should redirect to dashboard
        assert resp.status_code == 302
        dashboard_url = resp.headers["Location"]
        assert "/dashboard/" in dashboard_url

        # Follow redirect
        resp2 = client.get(dashboard_url)
        assert resp2.status_code == 200
        assert b"Network" in resp2.data

    def test_dashboard_has_help_panel(self, client, _sample_csv: str) -> None:
        resp = client.post("/analyze", data={
            "csvtext": _sample_csv,
            "scoring_method": "bic-d",
            "max_indegree": "3",
            "prior_type": "BDeu",
            "equivalent_sample_size": "5.0",
            "discretize_bins": "5",
            "target_node": "Commute",
        })
        dashboard_url = resp.headers["Location"]
        resp2 = client.get(dashboard_url)
        html = resp2.data.decode("utf-8")
        assert "操作ガイド" in html or "Quick Guide" in html

    def test_dashboard_has_japanese_tab_labels(self, client, _sample_csv: str) -> None:
        resp = client.post("/analyze", data={
            "csvtext": _sample_csv,
            "scoring_method": "bic-d",
            "max_indegree": "3",
            "prior_type": "BDeu",
            "equivalent_sample_size": "5.0",
            "discretize_bins": "5",
            "target_node": "Commute",
        })
        dashboard_url = resp.headers["Location"]
        resp2 = client.get(dashboard_url)
        html = resp2.data.decode("utf-8")
        assert "ネットワーク" in html
        assert "感度" in html

    def test_evidence_update(self, client, _sample_csv: str) -> None:
        # Create session
        resp = client.post("/analyze", data={
            "csvtext": _sample_csv,
            "scoring_method": "bic-d",
            "max_indegree": "3",
            "prior_type": "BDeu",
            "equivalent_sample_size": "5.0",
            "discretize_bins": "5",
            "target_node": "Commute",
        })
        dashboard_url = resp.headers["Location"]

        # Update evidence
        resp2 = client.post(dashboard_url, data={
            "action": "update_evidence",
            "ev_Weather": "Sunny",
        }, follow_redirects=True)
        assert resp2.status_code == 200

    def test_change_target(self, client, _sample_csv: str) -> None:
        resp = client.post("/analyze", data={
            "csvtext": _sample_csv,
            "scoring_method": "bic-d",
            "max_indegree": "3",
            "prior_type": "BDeu",
            "equivalent_sample_size": "5.0",
            "discretize_bins": "5",
            "target_node": "Commute",
        })
        dashboard_url = resp.headers["Location"]

        resp2 = client.post(dashboard_url, data={
            "action": "change_target",
            "new_target": "Weather",
        }, follow_redirects=True)
        assert resp2.status_code == 200

    def test_clear_evidence(self, client, _sample_csv: str) -> None:
        resp = client.post("/analyze", data={
            "csvtext": _sample_csv,
            "scoring_method": "bic-d",
            "max_indegree": "3",
            "prior_type": "BDeu",
            "equivalent_sample_size": "5.0",
            "discretize_bins": "5",
            "target_node": "Commute",
        })
        dashboard_url = resp.headers["Location"]

        resp2 = client.post(dashboard_url, data={
            "action": "clear_evidence",
        }, follow_redirects=True)
        assert resp2.status_code == 200


class TestSessionManagement:
    def test_expired_session_redirects(self, client) -> None:
        resp = client.get("/dashboard/abc123", follow_redirects=True)
        assert resp.status_code == 200
        assert b"Analyzer" in resp.data
