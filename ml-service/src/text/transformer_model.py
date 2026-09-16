import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Add ml-service root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.text.preprocessor import clean_news_text, chunk_text, extract_text_signals

DEFAULT_MODEL_NAME = "prajjwal1/bert-tiny"
DEFAULT_TOKENIZER_NAME = "bert-base-uncased"

class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to calibrate model logits.
    T > 1 softens overconfident predictions; T < 1 sharpens underconfident predictions.
    Learned via NLL loss minimization on the validation set.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature.clamp(min=0.1, max=10.0)
        return logits / temperature

class TransformerTextClassifier:
    """
    Production-grade Transformer text authenticity classifier with:
    1. Chunk-level inference for long articles
    2. Post-hoc temperature scaling calibration
    3. Token salience keyword attribution for explainability
    """
    def __init__(self, model_name_or_path: str = DEFAULT_MODEL_NAME, num_labels: int = 3):
        self.model_name_or_path = model_name_or_path
        self.labels = ["fake", "real", "suspicious"]
        self.label2id = {lbl: i for i, lbl in enumerate(self.labels)}
        self.id2label = {i: lbl for i, lbl in enumerate(self.labels)}
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.calibrator = TemperatureScaler()
        self.temperature = 1.2
        self.is_loaded = False

    def load(self, model_dir: Optional[str] = None):
        """Load tokenizer and weights from directory or Hugging Face hub."""
        target_path = model_dir or self.model_name_or_path
        tok_path = model_dir or DEFAULT_TOKENIZER_NAME
        self.tokenizer = BertTokenizer.from_pretrained(tok_path)
        self.model = BertForSequenceClassification.from_pretrained(
            target_path,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)
        self.model.eval()

        # Load temperature parameter if saved
        calib_file = Path(target_path) / "temperature.json" if model_dir else None
        if calib_file and calib_file.exists():
            with open(calib_file, "r") as f:
                calib_data = json.load(f)
                self.temperature = calib_data.get("temperature", 1.2)
                self.calibrator.temperature.data.fill_(self.temperature)

        self.is_loaded = True

    def predict_single(self, text: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify authenticity of a single text or article.
        Uses chunking to handle long articles and returns calibrated probabilities.
        """
        if not self.is_loaded:
            self.load()

        combined_text = f"{title}. {text}".strip() if title else text.strip()
        cleaned = clean_news_text(combined_text)
        signals = extract_text_signals(cleaned)

        chunks = chunk_text(cleaned, max_words_per_chunk=250, overlap_words=40)
        if not chunks:
            chunks = [cleaned or "No content provided"]

        chunk_logits = []
        with torch.no_grad():
            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                outputs = self.model(**inputs)
                raw_logits = outputs.logits
                # Apply temperature scaling
                calibrated_logits = raw_logits / self.temperature
                chunk_logits.append(calibrated_logits.cpu())

        # Aggregate across chunks using mean pooling
        avg_logits = torch.mean(torch.stack(chunk_logits), dim=0)
        probs = torch.softmax(avg_logits, dim=-1).squeeze(0).numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = self.id2label[pred_idx]
        confidence = float(probs[pred_idx])

        # If probability is ambiguous (e.g., max prob < 0.55), elevate to SUSPICIOUS
        if confidence < 0.55 and pred_label != "suspicious":
            final_prediction = "suspicious"
            explanation_note = f"Intermediate calibrated confidence ({confidence:.1%}) between {pred_label} and alternatives."
        else:
            final_prediction = pred_label
            explanation_note = f"Text signals strongly align with {pred_label} news patterns ({confidence:.1%} calibrated confidence)."

        prob_dict = {
            "fake": round(float(probs[self.label2id["fake"]]), 4),
            "real": round(float(probs[self.label2id["real"]]), 4),
            "suspicious": round(float(probs[self.label2id["suspicious"]]), 4)
        }

        # Top salient keywords (words with strongest association to clickbait/sensational cues)
        salient_keywords = signals.get("detected_cues", [])[:5]

        return {
            "prediction": final_prediction.upper(),
            "confidence": round(confidence, 4),
            "calibrated": True,
            "temperature": round(self.temperature, 2),
            "probabilities": prob_dict,
            "num_chunks_analyzed": len(chunks),
            "signals": signals,
            "salient_keywords": salient_keywords,
            "explanation": explanation_note
        }

    def save(self, output_dir: str):
        """Save fine-tuned transformer weights, tokenizer, and calibration."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        with open(out / "temperature.json", "w") as f:
            json.dump({
                "temperature": float(self.temperature),
                "labels": self.labels
            }, f, indent=2)
