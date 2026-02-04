from fastapi.testclient import TestClient

from web.api.main import app


client = TestClient(app)


def test_runs_endpoint():
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert "runs" in resp.json()


def test_missing_run_meta():
    resp = client.get("/runs/nonexistent/meta")
    assert resp.status_code in (404, 500)
