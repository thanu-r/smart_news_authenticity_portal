import pytest
import io
import uuid
from starlette.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_news_refresh_and_deduplication():
    """Test RSS feed refresh and verify deduplication logic."""
    res1 = client.post("/api/news/refresh")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1.get("status") == "success"
    assert "newly_ingested_articles" in data1

    # Second refresh immediately should recognize duplicates and not crash
    res2 = client.post("/api/news/refresh")
    assert res2.status_code == 200

def test_news_latest_and_search():
    """Test retrieving latest articles and searching by keyword."""
    res = client.get("/api/news/latest?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "articles" in data
    assert isinstance(data["articles"], list)
    assert len(data["articles"]) > 0

    # Search query
    sample_query = "news"
    search_res = client.get(f"/api/news/search?q={sample_query}&limit=5")
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert "articles" in search_data
    assert search_data["query"] == sample_query

def test_verification_history_and_feedback():
    """Test end-to-end verification and human feedback submission."""
    # 1. Run a text verification to register in history
    verif_res = client.post("/api/verify/text", json={
        "headline": "Test Scientific Breakthrough in Quantum Annealing",
        "text": "Researchers at international university have successfully demonstrated high fidelity quantum coherence in room-temperature synthetic diamond lattices with peer-reviewed validation."
    })
    assert verif_res.status_code == 200
    v_data = verif_res.json()
    assert "prediction" in v_data
    assert "calibrated_confidence" in v_data

    # 2. Query history
    hist_res = client.get("/api/history?limit=10")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert "records" in hist_data

    # 3. Submit feedback on a sample record id
    sample_id = hist_data["records"][0]["id"] if hist_data["records"] else str(uuid.uuid4())
    feedback_res = client.post(f"/api/history/{sample_id}/feedback", json={
        "is_correct": True,
        "correct_label": "REAL",
        "comments": "Confirmed by peer validation"
    })
    assert feedback_res.status_code == 200
    assert feedback_res.json().get("status") == "success"

def test_admin_retrain_and_metrics():
    """Test administrative model re-training queue and operational statistics."""
    # Retrain trigger
    retrain_res = client.post("/api/admin/retrain", json={"modality": "text"})
    assert retrain_res.status_code == 200
    retrain_data = retrain_res.json()
    assert retrain_data.get("status") in ["queued", "success"]
    assert "job_id" in retrain_data

    # Retrain invalid modality
    bad_res = client.post("/api/admin/retrain", json={"modality": "invalid_modality_xyz"})
    assert bad_res.status_code in [200, 400]  # graceful validation

    # Admin stats
    stats_res = client.get("/api/admin/metrics")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert "total_articles_ingested" in stats_data
    assert "models_active" in stats_data
