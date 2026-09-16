import sys
from pathlib import Path
import pytest

# Add ml-service to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml-service"))

from src.source.credibility_analyzer import SourceCredibilityAnalyzer
from src.evidence.claim_extractor import ClaimExtractor
from src.evidence.stance_detector import StanceDetector
from src.evidence.evidence_retriever import EvidenceRetriever

def test_source_credibility_analyzer_verified():
    """Verify high-credibility news registry resolution and scoring."""
    analyzer = SourceCredibilityAnalyzer()
    
    # Authoritative outlet
    res = analyzer.evaluate_source(
        url_or_domain="https://www.reuters.com/world/us/presidential-treaty-signed-2026",
        author="David Morgan"
    )
    assert res["category"] == "HIGH_CREDIBILITY"
    assert res["credibility_score"] >= 0.95
    assert res["publisher_name"] == "Reuters"
    assert res["breakdown"]["author_present"] is True

def test_source_credibility_analyzer_satire_and_suspicious():
    """Verify satire detection and high-risk TLD penalties."""
    analyzer = SourceCredibilityAnalyzer()

    # Known satire
    satire_res = analyzer.evaluate_source(url_or_domain="https://theonion.com/world-shocked-by-events")
    assert satire_res["category"] == "SATIRE"
    assert satire_res["breakdown"]["satire_detected"] is True
    assert satire_res["credibility_score"] <= 0.20

    # Suspicious domain with high-risk TLD
    spam_res = analyzer.evaluate_source(url_or_domain="http://breaking-news-shocking-leak.xyz")
    assert spam_res["category"] == "LOW_CREDIBILITY"
    assert spam_res["breakdown"]["tld_trust"] == "suspicious"
    assert spam_res["breakdown"]["is_https"] is False

    # High trust .gov domain
    gov_res = analyzer.evaluate_source(url_or_domain="https://cdc.gov/outbreaks/report")
    assert gov_res["category"] == "HIGH_CREDIBILITY"
    assert gov_res["breakdown"]["tld_trust"] == "high"

def test_claim_extractor():
    """Verify claim sentence extraction and clean search query generation."""
    extractor = ClaimExtractor()
    title = "NASA Announces Discovery of Subsurface Water Ice on Moon's South Pole"
    body = (
        "Scientists confirmed the observation using lunar reconnaissance telemetry. "
        "The discovery was verified during yesterday's press briefing at headquarters. "
        "Click here to subscribe for more space news!"
    )

    claims = extractor.extract_claims(body, title=title)
    assert len(claims) >= 1
    assert "NASA" in claims[0]

    queries = extractor.generate_search_queries(body, title=title)
    assert len(queries) >= 1
    # Check that query preserves core keywords and strips stop words
    assert any("nasa" in q.lower() or "water" in q.lower() or "moon" in q.lower() for q in queries)

def test_stance_detector_supporting_and_refuting():
    """Verify linguistic stance classification between claim and evidence."""
    detector = StanceDetector()
    claim = "President signs landmark clean energy and water infrastructure legislation."

    # Supporting evidence
    supp_evidence = "The president confirmed and signed the bipartisan clean energy infrastructure bill today."
    supp_res = detector.detect_stance(claim, supp_evidence)
    assert supp_res["stance"] == "SUPPORTS"
    assert supp_res["confidence"] > 0.60
    assert supp_res["relevance"] > 0.30

    # Refuting evidence
    contra_evidence = "Fact check: The claim that the president signed this bill is false and has been debunked by congressional records."
    contra_res = detector.detect_stance(claim, contra_evidence)
    assert contra_res["stance"] == "CONTRADICTS"
    assert contra_res["confidence"] > 0.65
    assert len(contra_res["detected_cues"]) > 0

    # Neutral / unrelated evidence
    neutral_evidence = "Local weather forecast shows mild temperatures across the midwest this weekend."
    neutral_res = detector.detect_stance(claim, neutral_evidence)
    assert neutral_res["stance"] == "NEUTRAL"

def test_evidence_retriever_synthesis():
    """Verify overall evidence retrieval structure, ranking, and synthesis."""
    retriever = EvidenceRetriever()
    
    # Test fallback on obscure claim
    result = retriever.retrieve_evidence(
        text="Unique obscure local claim that has no global news coverage xyz_992149",
        title="Obscure Local Event"
    )
    assert "corroboration_status" in result
    assert "overall_stance" in result
    assert "corroboration_score" in result
    assert "evidence_list" in result
    assert result["corroboration_status"] in ["INSUFFICIENT_EVIDENCE", "EVIDENCE_RETRIEVED"]
