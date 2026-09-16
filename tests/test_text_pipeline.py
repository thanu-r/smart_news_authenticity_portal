import sys
from pathlib import Path
import numpy as np

# Add ml-service directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml-service"))

from src.text.preprocessor import clean_news_text, chunk_text, extract_text_signals
from src.text.baseline_model import BaselineTextModel
from src.text.transformer_model import TransformerTextClassifier

def test_clean_news_text():
    raw_wire = "WASHINGTON (Reuters) - The president met with officials today. https://example.com/live"
    cleaned = clean_news_text(raw_wire)
    assert "WASHINGTON (Reuters) -" not in cleaned
    assert "https://" not in cleaned
    assert "The president met with officials today." in cleaned

def test_chunk_text():
    long_text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_text(long_text, max_words_per_chunk=200, overlap_words=40)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c.split()) <= 200

def test_extract_text_signals():
    text = "SHOCKING BOMBSHELL: Mainstream media won't tell you the secret miracle truth! Is this real?!"
    signals = extract_text_signals(text)
    assert signals["word_count"] > 0
    assert signals["exclamation_count"] >= 1
    assert signals["sensational_score"] > 0.3
    assert len(signals["detected_cues"]) > 0

def test_baseline_model_inference():
    model_path = Path("ml-service/models/text/baseline_tfidf.joblib")
    assert model_path.exists(), "Trained baseline model must exist on disk"
    model = BaselineTextModel.load(str(model_path))
    assert model.is_trained

    samples = [
        "President signs bilateral trade treaty following congressional vote.",
        "BREAKING: Secret alien conspiracy discovered in basement of federal agency!"
    ]
    preds = model.predict(samples)
    probs = model.predict_proba(samples)

    assert len(preds) == 2
    assert probs.shape == (2, 3)
    for p in probs:
        assert np.isclose(np.sum(p), 1.0, atol=1e-3)

def test_transformer_model_inference():
    model_dir = Path("ml-service/models/text/transformer_v1")
    assert model_dir.exists(), "Trained transformer model must exist on disk"
    classifier = TransformerTextClassifier()
    classifier.load(str(model_dir))
    assert classifier.is_loaded

    result = classifier.predict_single(
        title="Senate passes bipartisan infrastructure legislation",
        text="The bill includes funding for federal highways, clean energy, and municipal water systems."
    )
    assert "prediction" in result
    assert result["prediction"] in ["REAL", "FAKE", "SUSPICIOUS"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert "fake" in result["probabilities"]
    assert "real" in result["probabilities"]
    assert "suspicious" in result["probabilities"]
    assert result["calibrated"] is True
    assert result["num_chunks_analyzed"] >= 1
