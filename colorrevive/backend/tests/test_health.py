"""Health and metadata endpoint tests — dynamic service + model status."""

from __future__ import annotations


def test_health_reports_ddcolor_when_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "colorrevive-api"
    assert body["model_loaded"] is True
    assert body["fallback_mode"] is False
    assert body["device"] in {"cpu", "cuda"}


def test_health_reports_fallback_when_explicitly_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_FALLBACK_MODE", "true")
    monkeypatch.setenv("DDCOLOR_MODEL", str(tmp_path / "nonexistent.pt"))
    from fastapi.testclient import TestClient
    from app.config import get_settings
    from app.main import create_app
    from app.ml.inference import reset_model_singleton

    get_settings.cache_clear()
    reset_model_singleton()

    with TestClient(create_app()) as fallback_client:
        response = fallback_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
        assert body["fallback_mode"] is True


def test_info_endpoint(client):
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "colorrevive-api"
    assert set(body["supported_formats"]) == {"image/jpeg", "image/png", "image/webp"}
    assert body["max_upload_mb"] == 10
    assert body["model_name"] == "DDColor"
    assert body["fallback_mode"] is False


def test_model_status_endpoint(client):
    response = client.get("/api/v1/model-status")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "DDColor"
    assert body["loaded"] is True
    assert body["fallback_mode"] is False
    assert body["precision"] == "float32"
    assert "version" in body
