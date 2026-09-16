import io
import pytest
from starlette.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_edge_case_token_overflow_long_article():
    """Verify that articles far exceeding the 512-token Transformer limit are chunked cleanly without crash."""
    # 3,000 words of realistic text
    long_paragraph = (
        "In a groundbreaking scientific symposium held in Geneva, international researchers unveiled comprehensive "
        "measurements detailing quantum entanglement phenomena across unprecedented macroscopic distances. "
        "The experimental setup utilized optical micro-cavities synchronized to atomic clocks with sub-picosecond accuracy. "
    )
    huge_text = (long_paragraph * 40)  # >2,500 tokens
    
    res = client.post("/api/verify/text", json={
        "headline": "Massive Long-Form Investigative Article on Macroscopic Entanglement",
        "text": huge_text,
        "model_type": "transformer"
    })
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert "calibrated_confidence" in data
    assert "explanation" in data
    # Baseline model check on same text
    res_b = client.post("/api/verify/text", json={
        "headline": "Massive Long-Form Investigative Article on Macroscopic Entanglement",
        "text": huge_text,
        "model_type": "baseline"
    })
    assert res_b.status_code == 200

def test_edge_case_unicode_and_emojis():
    """Verify robust handling of emojis, foreign scripts, and HTML fragments."""
    messy_text = (
        "🚨 BREAKING: <html><body>Breaking news alert! <b>Secret leaks</b> revealed! "
        "你好世界 Привет мир 🌍🚀✨ "
        "New legislation passed by committee after heated debate in parliament. "
        "</body></html>"
    )
    res = client.post("/api/verify/text", json={
        "headline": "🔥 Leaked Report: Committee Vote Results!",
        "text": messy_text
    })
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] in ["REAL", "FAKE", "SUSPICIOUS"]

def test_edge_case_malformed_url():
    """Verify that unresolvable or malformed URLs fail gracefully without crashing server."""
    bad_urls = [
        "http://not-a-real-domain-xyz-123456789.org/article.html",
        "not_even_a_url_string",
    ]
    for url in bad_urls:
        res = client.post("/api/verify/url", json={"url": url})
        # Should return either 200 with fallback analysis or 400/422/502 with structured error
        assert res.status_code in [200, 400, 422, 502, 503]

def test_edge_case_corrupted_image_bytes():
    """Verify that corrupted image bytes return structured error or handle safely."""
    corrupted_bytes = b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE_BYTES_123456"
    files = {"image": ("corrupted.jpg", io.BytesIO(corrupted_bytes), "image/jpeg")}
    
    res = client.post("/api/verify/image", files=files)
    assert res.status_code in [200, 400, 422]  # handled safely

def test_edge_case_corrupted_video_bytes():
    """Verify that corrupted video bytes return structured error or handle safely."""
    corrupted_bytes = b"CORRUPTED_MP4_HEADER_INVALID_STREAM"
    files = {"video": ("corrupted.mp4", io.BytesIO(corrupted_bytes), "video/mp4")}
    
    res = client.post("/api/verify/video", files=files)
    assert res.status_code in [200, 400, 422]  # handled safely

def test_edge_case_empty_multimodal_request():
    """Verify that a request with no content fields is rejected cleanly."""
    res = client.post("/api/verify/multimodal", data={})
    assert res.status_code in [400, 422]
