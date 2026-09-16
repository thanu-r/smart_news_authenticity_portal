import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import pytest

# Add ml-service to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml-service"))

from src.multimodal.cross_modal_consistency import CrossModalConsistencyEngine

def test_headline_body_consistency_aligned():
    """Verify high consistency score when headline and body share factual context."""
    engine = CrossModalConsistencyEngine()
    headline = "Federal Reserve Announces Interest Rate Decision Following Policy Meeting"
    body = (
        "The Federal Reserve announced on Wednesday that benchmark interest rates will remain unchanged. "
        "Officials cited ongoing progress on inflation following their two-day monetary policy meeting in Washington."
    )
    res = engine.evaluate_headline_body_consistency(headline, body)
    assert res["status"] in ["STRONG_CONSISTENCY", "MODERATE_CONSISTENCY"]
    assert res["score"] >= 0.70
    assert res["alert"] is None

def test_headline_body_consistency_clickbait_mismatch():
    """Verify clickbait detection when sensational headline is not substantiated by body."""
    engine = CrossModalConsistencyEngine()
    headline = "BOMBSHELL EXPOSED: Secret Miracle Cure Banned by Officials!"
    body = (
        "Yesterday in downtown Seattle, a local bakery celebrated its tenth anniversary with free pastries. "
        "Dozens of customers lined up around the block to purchase fresh sourdough loaves and coffee."
    )
    res = engine.evaluate_headline_body_consistency(headline, body)
    assert res["status"] in ["MISMATCH", "EXAGGERATED_CLICKBAIT"]
    assert res["score"] < 0.60
    assert res["alert"] is not None

def test_cross_modal_radar_synthesis():
    """Verify radar dimensions and multi-signal consistency aggregation."""
    engine = CrossModalConsistencyEngine()

    # Create dummy image
    img = Image.new("RGB", (128, 128), color=(120, 150, 180))

    result = engine.analyze_cross_modal_consistency(
        headline="President signs bilateral infrastructure treaty in Geneva",
        body="Diplomats from both nations attended the signing ceremony in Geneva today.",
        image_input=img,
        evidence_corroboration_score=0.88
    )

    assert "overall_consistency_score" in result
    assert 0.0 <= result["overall_consistency_score"] <= 1.0
    assert "consensus_status" in result
    assert "radar_scores" in result
    assert "headline_body" in result["radar_scores"]
    assert "text_image" in result["radar_scores"]
    assert "claim_evidence" in result["radar_scores"]
    assert isinstance(result["mismatch_alerts"], list)

def test_confidence_calibrator():
    """Verify temperature scaling calibration and ECE calculation."""
    from src.confidence.calibrator import ConfidenceCalibrator

    calibrator = ConfidenceCalibrator(default_temperature=1.20)
    raw_probs = np.array([0.90, 0.05, 0.05])
    calibrated = calibrator.calibrate_probabilities(raw_probs)

    assert np.isclose(np.sum(calibrated), 1.0, atol=1e-4)
    # Scaled probability should be slightly softer than raw extreme
    assert calibrated[0] < raw_probs[0]
    assert np.argmax(calibrated) == np.argmax(raw_probs)

    # Test ECE calculation
    test_probs = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]])
    test_labels = np.array([0, 1, 2])
    ece = calibrator.compute_ece(test_probs, test_labels)
    assert 0.0 <= ece <= 1.0

def test_multifactor_explanation_engine():
    """Verify structured explanation generation and driver factor breakdown."""
    from src.explanation.explanation_engine import MultiFactorExplanationEngine

    engine = MultiFactorExplanationEngine()
    explanation = engine.generate_explanation(
        final_prediction="FAKE",
        confidence=0.89,
        text_signals={"detected_cues": ["bombshell", "miracle"], "sensational_score": 0.65},
        image_signals={"anomalies_detected": ["Elevated ELA compression residual"]},
        source_signals={"category": "LOW_CREDIBILITY", "publisher_name": "UnverifiedBlog"}
    )

    assert explanation["prediction"] == "FAKE"
    assert explanation["confidence"] == 0.89
    assert len(explanation["key_factors"]) >= 2
    assert len(explanation["detailed_reasons"]) >= 3
    assert "executive_summary" in explanation
    assert "FAKE" in explanation["executive_summary"]

def test_multimodal_decision_fusion_end_to_end():
    """Verify end-to-end multimodal verification decision and response schema."""
    from src.multimodal.fusion_engine import MultimodalDecisionFusionEngine

    engine = MultimodalDecisionFusionEngine()
    engine.load()

    # Verified clean news sample
    result = engine.verify_content(
        title="Senate Approves Bipartisan Infrastructure Legislation",
        text="Lawmakers voted 68-32 in favor of the federal highways and public transport investment package.",
        url="https://www.reuters.com/world/us/senate-passes-bill",
        author="Patricia Zengerle",
        retrieve_evidence=False
    )

    assert "prediction" in result
    assert result["prediction"] in ["REAL", "FAKE", "SUSPICIOUS"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert "fake" in result["probabilities"]
    assert "real" in result["probabilities"]
    assert "suspicious" in result["probabilities"]
    assert "modalities_analyzed" in result
    assert "text" in result["modalities_analyzed"]
    assert "source" in result["modalities_analyzed"]
    assert "explanation" in result
    assert "executive_summary" in result["explanation"]

