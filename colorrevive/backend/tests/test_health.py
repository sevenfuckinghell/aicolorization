"""Health endpoint tests — status must reflect real dynamic model state."""

from __future__ import annotations


def test_health_reports_fallback_when_no_checkpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "colorrevive-api"
    # No checkpoint in tests => model_loaded false, fallback true.
    assert body["model_loaded"] is False
    assert body["fallback_mode"] is True
    assert body["device"] in {"cpu", "cuda"}


def test_info_endpoint(client):
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "colorrevive-api"
    assert set(body["supported_formats"]) == {"image/jpeg", "image/png", "image/webp"}
    assert body["max_upload_mb"] == 10
    assert body["fallback_mode"] is True
    assert body["model_name"] == "fallback-heuristic"


def test_model_status_endpoint(client):
    response = client.get("/api/v1/model-status")
    assert response.status_code == 200
    body = response.json()
    assert body["loaded"] is False
    assert body["fallback_mode"] is True
    assert body["precision"] == "float32"
    assert "version" in body
