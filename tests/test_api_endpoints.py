import io
import sys
from pathlib import Path
from PIL import Image
import pytest
from starlette.testclient import TestClient

from backend.app.main import app

def test_api_health():
    """Verify health endpoint."""
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

def test_api_verify_text():
    """Verify text verification endpoint with real payload."""
    with TestClient(app) as client:
        payload = {
            "title": "Bipartisan Healthcare Reform Bill Introduced in Senate",
            "text": "Senators from both parties unveiled sweeping legislation aimed at reducing prescription drug costs.",
            "url": "https://www.reuters.com/world/us/healthcare-bill-2026",
            "author": "David Shepardson"
        }
        resp = client.post("/api/verify/text", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "prediction" in data
        assert data["prediction"] in ["REAL", "FAKE", "SUSPICIOUS"]
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert "probabilities" in data
        assert "explanation" in data
        assert "executive_summary" in data["explanation"]

def test_api_verify_image():
    """Verify image verification endpoint with multipart file upload."""
    with TestClient(app) as client:
        # Create small test image buffer
        img = Image.new("RGB", (64, 64), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        files = {"file": ("test_news_photo.jpg", buf, "image/jpeg")}
        data = {"title": "Press Briefing at Capitol"}

        resp = client.post("/api/verify/image", files=files, data=data)
        assert resp.status_code == 200
        res = resp.json()
        assert "id" in res
        assert "prediction" in res
        assert "confidence" in res
        assert "probabilities" in res

def test_api_models_info():
    """Verify model metadata and active benchmark endpoint."""
    with TestClient(app) as client:
        resp = client.get("/api/models/current")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "operational"
        assert "active_models" in data
        assert "text_transformer" in data["active_models"]
        assert "text_baseline" in data["active_models"]
        assert "image_forensics" in data["active_models"]
        assert "video_forensics" in data["active_models"]

def test_api_live_news_feed():
    """Verify live news feed endpoint returns articles."""
    with TestClient(app) as client:
        resp = client.get("/api/news/live?limit=5")
        assert resp.status_code == 200
        articles = resp.json()
        assert isinstance(articles, list)

def test_api_admin_stats_and_feedback():
    """Verify admin platform stats and user feedback submission."""
    with TestClient(app) as client:
        stats_resp = client.get("/api/admin/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert "total_verifications" in stats
        assert "fake_count" in stats
        assert "real_count" in stats
        assert "modality_breakdown" in stats

        # Submit feedback
        fb_payload = {
            "verification_id": "test_verif_123",
            "feedback_type": "agree",
            "user_label": "REAL",
            "comment": "Accurate assessment confirmed by primary source."
        }
        fb_resp = client.post("/api/admin/feedback", json=fb_payload)
        assert fb_resp.status_code == 200
        fb_data = fb_resp.json()
        assert fb_data["status"] == "recorded"
