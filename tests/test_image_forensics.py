import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add ml-service directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml-service"))

from src.image.forensic_extractor import compute_ela, compute_noise_inconsistency, extract_metadata_forensics, extract_image_features
from src.image.image_classifier import ImageForensicClassifier

def test_ela_computation():
    """Verify that ELA produces valid enhanced difference map and scalar error score."""
    test_img = Image.new("RGB", (200, 200), color=(120, 150, 180))
    # Draw a distinct patch simulating a spliced region
    for x in range(50, 100):
        for y in range(50, 100):
            test_img.putpixel((x, y), (250, 40, 40))

    enhanced, ela_score = compute_ela(test_img, quality=90)
    assert enhanced is not None
    assert isinstance(ela_score, float)
    assert ela_score >= 0.0

def test_noise_inconsistency_metric():
    """Verify noise variance across patches is bounded between 0.0 and 1.0."""
    test_img = Image.new("RGB", (256, 256), color=(200, 200, 200))
    noise = compute_noise_inconsistency(test_img)
    assert 0.0 <= noise <= 1.0

def test_metadata_extraction():
    """Verify EXIF extraction handles clean image without throwing exception."""
    test_img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    meta = extract_metadata_forensics(test_img)
    assert "has_exif" in meta
    assert "editing_software_detected" in meta
    assert isinstance(meta["editing_software_detected"], bool)

def test_image_classifier_inference():
    """Verify trained CASIA image forensic classifier produces structured predictions."""
    model_path = Path("ml-service/models/image/image_forensic_model.joblib")
    assert model_path.exists(), "Trained image model must exist on disk"

    classifier = ImageForensicClassifier(str(model_path))
    classifier.load()
    assert classifier.is_loaded

    test_img = Image.new("RGB", (256, 256), color=(180, 210, 240))
    result = classifier.predict_image(test_img)

    assert "prediction" in result
    assert result["prediction"] in ["REAL", "FAKE", "FAKE_TAMPERED", "AI_GENERATED", "SUSPICIOUS"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert "fake" in result["probabilities"]
    assert "real" in result["probabilities"]
    assert "forensic_signals" in result
    assert "ela_error_score" in result["forensic_signals"]
    assert "explanation" in result

def test_corrupt_image_graceful_handling():
    """Verify system does not crash on corrupt/unreadable image input."""
    classifier = ImageForensicClassifier()
    classifier.load()

    # Pass invalid path
    res = classifier.predict_image("non_existent_corrupted_file.jpg")
    assert res["prediction"] == "SUSPICIOUS"
    assert "error" in res
