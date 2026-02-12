from fastapi.testclient import TestClient
import pytest
from typing import Optional

from web.api.main import app
from web.api.storage import list_runs


client = TestClient(app)


def test_runs_endpoint():
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert "runs" in resp.json()


def test_missing_run_meta():
    resp = client.get("/runs/nonexistent/meta")
    assert resp.status_code in (404, 500)


def _run_id_for_source(source: str) -> Optional[str]:
    for run in reversed(list_runs()):
        if run.get("source") == source:
            return str(run.get("run_id"))
    return None


def test_returns_real_run():
    run_id = _run_id_for_source("real")
    if run_id is None:
        pytest.skip("No real run available for returns endpoint test.")

    resp = client.get(f"/runs/{run_id}/returns")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["units"] == "log_return_pct"
    assert isinstance(body.get("assets"), list)
    for asset in body["assets"]:
        assert asset in body["series"]
        assert asset in body["extreme_points"]


def test_returns_missing_run():
    resp = client.get("/runs/nonexistent/returns")
    assert resp.status_code == 404


def test_returns_generative_supported():
    run_id = _run_id_for_source("generative")
    if run_id is None:
        pytest.skip("No generative run available for returns endpoint test.")

    resp = client.get(f"/runs/{run_id}/returns")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["units"] == "log_return_pct"
    assert body.get("series_mode") in {"generative_imputed", "generative_event_only_fallback"}
    for asset in body["assets"]:
        assert asset in body["series"]
        assert asset in body["extreme_points"]


def test_returns_cached_consistent_response():
    run_id = _run_id_for_source("real")
    if run_id is None:
        pytest.skip("No real run available for cache consistency test.")

    first = client.get(f"/runs/{run_id}/returns")
    second = client.get(f"/runs/{run_id}/returns")
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
