import io
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import pytest
from starlette.testclient import TestClient

# Add ml-service and backend to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml-service"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.main import app
from backend.app.utils.safe_fetcher import is_ip_allowed, validate_url_safety, SSRFSecurityError, safe_fetch_url
from backend.app.utils.article_extractor import extract_article_from_html, ArticleExtractionError
from src.image.ai_image_detector import AIImageDetector
from src.image.image_classifier import ImageForensicClassifier
from src.video.video_classifier import VideoForensicClassifier

def create_sample_video(path: str, num_frames: int = 12, fps: int = 6, is_synthetic: bool = False):
    """Generate a valid MP4 test video for API testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (128, 128))
    for i in range(num_frames):
        if is_synthetic and i % 2 == 1:
            # High-frequency noise simulating generative frame flicker
            frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        else:
            base_col = int(80 + i * 8)
            frame = np.full((128, 128, 3), base_col, dtype=np.uint8)
            cv2.circle(frame, (64 + i * 2, 64), 20, (0, 200, 100), -1)
        out.write(frame)
    out.release()

# -------------------------------------------------------------
# 1. SSRF & Safe URL Fetcher Tests
# -------------------------------------------------------------
def test_ssrf_ip_blocking():
    """Verify safe fetcher rejects all private, loopback, and metadata IP addresses."""
    assert not is_ip_allowed("127.0.0.1"), "Loopback must be blocked"
    assert not is_ip_allowed("127.0.1.1"), "Loopback range must be blocked"
    assert not is_ip_allowed("10.0.0.1"), "RFC 1918 10.x must be blocked"
    assert not is_ip_allowed("192.168.1.1"), "RFC 1918 192.168.x must be blocked"
    assert not is_ip_allowed("172.16.0.1"), "RFC 1918 172.16.x must be blocked"
    assert not is_ip_allowed("169.254.169.254"), "Cloud metadata service IP must be blocked"
    assert not is_ip_allowed("0.0.0.0"), "Unspecified IP must be blocked"
    assert not is_ip_allowed("::1"), "IPv6 loopback must be blocked"
    assert not is_ip_allowed("fe80::1"), "IPv6 link-local must be blocked"

    # Public routable IPs must be allowed
    assert is_ip_allowed("8.8.8.8"), "Public DNS must be allowed"
    assert is_ip_allowed("1.1.1.1"), "Public IP must be allowed"

def test_validate_url_safety_rejects_malicious_schemes_and_hosts():
    """Verify URL validation rejects dangerous protocols and internal hosts."""
    with pytest.raises(SSRFSecurityError):
        validate_url_safety("ftp://example.com/file")

    with pytest.raises(SSRFSecurityError):
        validate_url_safety("file:///etc/passwd")

    with pytest.raises(SSRFSecurityError):
        validate_url_safety("http://localhost:8000/api/admin")

    with pytest.raises(SSRFSecurityError):
        validate_url_safety("http://127.0.0.1:27017")

    with pytest.raises(SSRFSecurityError):
        validate_url_safety("http://169.254.169.254/latest/meta-data/")

# -------------------------------------------------------------
# 2. Article Extraction Tests
# -------------------------------------------------------------
def test_extract_article_from_html_json_ld():
    """Verify article extraction from Schema.org JSON-LD."""
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Default Title</title>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": "Renewable Clean Energy Surpasses Coal in Global Grid Milestone",
        "articleBody": "International energy regulatory agencies confirmed today that combined renewable wind and solar power generation generated more electricity than traditional coal over the past fiscal quarter, representing an unprecedented acceleration towards carbon reduction commitments.",
        "author": {"@type": "Person", "name": "Elena Rostova"},
        "datePublished": "2026-04-12T10:00:00Z",
        "image": "https://example.com/energy-grid.jpg"
      }
      </script>
    </head>
    <body><p>Short paragraph.</p></body>
    </html>
    """
    article = extract_article_from_html(sample_html, "https://reuters.com/clean-energy-milestone")
    assert article["title"] == "Renewable Clean Energy Surpasses Coal in Global Grid Milestone"
    assert "renewable wind and solar power" in article["text"]
    assert article["author"] == "Elena Rostova"
    assert article["domain"] == "reuters.com"
    assert article["lead_image_url"] == "https://example.com/energy-grid.jpg"

def test_extract_article_empty_html_fails():
    """Verify empty or unextractable HTML raises ArticleExtractionError."""
    with pytest.raises(ArticleExtractionError):
        extract_article_from_html("<html><body><p>Under 20 chars</p></body></html>", "https://empty-news.com/page")

# -------------------------------------------------------------
# 3. AI-Generated Image Forensics Tests
# -------------------------------------------------------------
def test_ai_image_detector_provenance_markers():
    """Verify AI image detector catches synthetic generator PNG text metadata."""
    img = Image.new("RGB", (256, 256), color=(220, 220, 220))
    img.info["parameters"] = "A futuristic cyberpunk metropolis at night, Steps: 30, Sampler: DPM++ 2M, CFG scale: 7, Model: StableDiffusion-XL"
    
    detector = AIImageDetector()
    res = detector.predict_ai_generated(img)
    assert res["is_ai_generated"] is True
    assert res["ai_probability"] >= 0.85
    assert any("StableDiffusion" in r or "parameters" in r for r in res["reasons"])

def test_ai_image_detector_natural_photo():
    """Verify natural photo features produce low AI probability."""
    # Create smooth gradient simulating real camera sensor shot
    arr = np.zeros((384, 384, 3), dtype=np.uint8)
    for y in range(384):
        for x in range(384):
            arr[y, x] = [int((x / 384) * 180 + 30), int((y / 384) * 160 + 40), 120]
    img = Image.fromarray(arr)

    detector = AIImageDetector()
    res = detector.predict_ai_generated(img)
    assert res["ai_probability"] < 0.65

def test_image_forensic_classifier_end_to_end():
    """Verify ImageForensicClassifier correctly fuses CASIA2 and AI-generation checks."""
    classifier = ImageForensicClassifier()
    classifier.load()

    # Clean test image
    test_img = Image.new("RGB", (256, 256), color=(150, 180, 210))
    res = classifier.predict_image(test_img)

    assert "prediction" in res
    assert res["prediction"] in ["REAL", "FAKE_TAMPERED", "AI_GENERATED", "SUSPICIOUS"]
    assert "probabilities" in res
    assert "real" in res["probabilities"]
    assert "tampered" in res["probabilities"]
    assert "ai_generated" in res["probabilities"]
    assert "forensic_signals" in res
    assert "ai_generated_analysis" in res["forensic_signals"]

# -------------------------------------------------------------
# 4. Video Forensics Pipeline & API Tests
# -------------------------------------------------------------
def test_video_classifier_end_to_end():
    """Verify VideoForensicClassifier runs keyframe sampling and temporal aggregation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "test_clip.mp4")
        create_sample_video(vid_path, num_frames=12, is_synthetic=True)

        classifier = VideoForensicClassifier(target_fps=2.0, max_frames=8)
        classifier.load()
        res = classifier.predict_video(vid_path)

        assert "prediction" in res
        assert res["prediction"] in ["REAL", "FAKE_MANIPULATED", "DEEPFAKE", "AI_GENERATED", "SUSPICIOUS"]
        assert "confidence" in res
        assert "probabilities" in res
        assert "ai_generated" in res["probabilities"]
        assert "frames_analyzed" in res
        assert "temporal_flicker" in res
        assert "deepfake_risk" in res
        assert "suspicious_timestamps" in res

def test_api_verify_video_endpoint():
    """Verify POST /api/verify/video works end-to-end via FastAPI TestClient."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "news_broadcast.mp4")
        create_sample_video(vid_path, num_frames=10)

        with open(vid_path, "rb") as vf:
            vid_bytes = vf.read()

        with TestClient(app) as client:
            files = {"file": ("news_broadcast.mp4", io.BytesIO(vid_bytes), "video/mp4")}
            data = {"title": "Capitol Press Conference"}

            resp = client.post("/api/verify/video", files=files, data=data)
            assert resp.status_code == 200, f"Failed: {resp.text}"
            res = resp.json()

            assert "id" in res
            assert "prediction" in res
            assert res["prediction"] in ["REAL", "FAKE_MANIPULATED", "DEEPFAKE", "AI_GENERATED", "SUSPICIOUS"]
            assert "confidence" in res
            assert "probabilities" in res
            assert "video_forensics" in res
            assert "modality_breakdown" in res
            assert res["modality_breakdown"] is not None
            assert "video" in res["modality_breakdown"]
            assert res["modality_breakdown"]["video"] is not None

def test_api_verify_url_ssrf_blocked():
    """Verify POST /api/verify/url blocks loopback/internal addresses with HTTP 400."""
    with TestClient(app) as client:
        # Loopback URL
        resp = client.post("/api/verify/url", json={"url": "http://127.0.0.1:8000/api/admin"})
        assert resp.status_code == 400
        assert "restricted" in resp.json()["detail"].lower() or "security" in resp.json()["detail"].lower()

        # Localhost URL
        resp2 = client.post("/api/verify/url", json={"url": "http://localhost:5173"})
        assert resp2.status_code == 400
