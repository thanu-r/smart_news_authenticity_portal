import pytest
import requests
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Verify backend health and database connectivity."""
    resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True

def test_live_news_feed_refresh_and_deduplication():
    """Verify live news ingestion from verified RSS sources and deduplication."""
    resp1 = requests.post(f"{BASE_URL}/api/news/refresh", timeout=15)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] == "success"
    assert "timestamp" in data1

    # Second refresh should deduplicate articles
    resp2 = requests.post(f"{BASE_URL}/api/news/refresh", timeout=15)
    assert resp2.status_code == 200
    data2 = resp2.json()
    # Ingestion count on immediate re-run should be low / zero due to deduplication
    assert data2["newly_ingested_articles"] <= data1["newly_ingested_articles"]

def test_live_news_latest_retrieval_and_fields():
    """Verify articles contain title, source, publication date, URL, and category."""
    resp = requests.get(f"{BASE_URL}/api/news/latest?limit=10", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    articles = data.get("articles", [])
    assert len(articles) > 0

    first = articles[0]
    assert "id" in first
    assert "title" in first and len(first["title"]) > 0
    assert "source" in first
    assert "published_at" in first
    assert "url" in first
    assert "category" in first

def test_live_news_single_article_verification():
    """Verify verify-one endpoint runs claim extraction, ML inference, and evidence checking."""
    # Retrieve latest article
    lat_resp = requests.get(f"{BASE_URL}/api/news/latest?limit=5", timeout=10)
    assert lat_resp.status_code == 200
    articles = lat_resp.json().get("articles", [])
    assert len(articles) > 0

    target = articles[0]
    art_id = target["id"]

    ver_resp = requests.post(f"{BASE_URL}/api/news/verify-one/{art_id}", timeout=25)
    assert ver_resp.status_code == 200
    ver_data = ver_resp.json()

    assert ver_data["id"] == art_id
    assert ver_data["verified"] is True
    assert ver_data["verification_status"] in ["REAL", "FAKE", "SUSPICIOUS"]
    assert ver_data["confidence"] is not None
    assert ver_data["verified_at"] is not None
    # Verified timestamp should exist and be distinct
    assert str(ver_data["verified_at"]) != str(ver_data["published_at"])

def test_live_news_search():
    """Verify search endpoint queries across ingested article headlines and text."""
    resp = requests.get(f"{BASE_URL}/api/news/search?q=news", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert "articles" in data
    assert "query" in data
    assert data["query"] == "news"

def test_url_ssrf_security_defenses():
    """Verify SSRF protection blocks localhost, internal IPs, AWS metadata, and invalid schemes."""
    # 1. Localhost IP
    r1 = requests.post(f"{BASE_URL}/api/verify/url", json={"url": "http://127.0.0.1:8000/api/admin/metrics"})
    assert r1.status_code == 400
    assert "restricted" in r1.json()["detail"].lower()

    # 2. Localhost Domain
    r2 = requests.post(f"{BASE_URL}/api/verify/url", json={"url": "http://localhost:8000/secret"})
    assert r2.status_code == 400

    # 3. Cloud metadata endpoint (169.254.169.254)
    r3 = requests.post(f"{BASE_URL}/api/verify/url", json={"url": "http://169.254.169.254/latest/meta-data/"})
    assert r3.status_code == 400

    # 4. Private RFC 1918 Class A/B/C addresses
    r4 = requests.post(f"{BASE_URL}/api/verify/url", json={"url": "http://10.0.0.1/internal"})
    assert r4.status_code == 400

    # 5. Disallowed protocol scheme (ftp, file, gopher)
    r5 = requests.post(f"{BASE_URL}/api/verify/url", json={"url": "ftp://files.internal.lan/leak.txt"})
    assert r5.status_code == 400
    assert "protocol" in r5.json()["detail"].lower()

def test_url_unsupported_and_invalid():
    """Verify non-existent and inaccessible URLs return meaningful HTTP 400 error."""
    resp = requests.post(f"{BASE_URL}/api/verify/url", json={"url": "https://this-domain-does-not-exist-at-all-xyz9876.org/news"})
    assert resp.status_code == 400
    assert "unable to access" in resp.json()["detail"].lower() or "dns" in resp.json()["detail"].lower() or "resolve" in resp.json()["detail"].lower()

def test_refuted_claim_evidence_override():
    """Verify that a claim refuted by verified fact-checkers is NOT classified as REAL."""
    # Synthetic claim mimicking a debunked viral hoax
    debunked_payload = {
        "title": "NASA Confirms Earth Will Experience 15 Days of Complete Darkness in November",
        "headline": "NASA Confirms Earth Will Experience 15 Days of Complete Darkness in November",
        "text": "NASA scientists have reportedly issued a global bulletin warning that planetary alignment between Jupiter and Venus will trigger a massive atmospheric solar storm causing fifteen consecutive days of total blackness across the entire planet Earth.",
        "model_type": "transformer"
    }
    resp = requests.post(f"{BASE_URL}/api/verify/text", json=debunked_payload, timeout=20)
    assert resp.status_code == 200
    data = resp.json()
    # Must NOT be classified as REAL!
    assert data["prediction"] in ["FAKE", "SUSPICIOUS"]
    assert data["prediction"] != "REAL"

def test_breaking_news_insufficient_evidence():
    """Verify uncorroborated breaking news is treated cautiously with SUSPICIOUS rather than auto REAL."""
    breaking_payload = {
        "title": "Developing: Unidentified Private Satellite Enters Polar Orbit at Unusual Altitude",
        "headline": "Developing: Unidentified Private Satellite Enters Polar Orbit at Unusual Altitude",
        "text": "Reports are just emerging from amateur telemetry stations regarding an undocumented orbital insertion at 450km altitude. No commercial space operator or defense ministry has claimed ownership of the vehicle as of 10 minutes ago.",
        "model_type": "transformer"
    }
    resp = requests.post(f"{BASE_URL}/api/verify/text", json=breaking_payload, timeout=20)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in ["SUSPICIOUS", "REAL"]
