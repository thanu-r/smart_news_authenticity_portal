import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import joblib
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.image.forensic_extractor import extract_image_features, compute_ela
from src.image.ai_image_detector import AIImageDetector

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "image" / "image_forensic_model.joblib"

class ImageForensicClassifier:
    """
    Forensic image manipulation, splicing, and AI-generated image classifier.
    Combines compression residual analysis (ELA), noise variance,
    Bayer demosaicing, 2D FFT spectral decay, and metadata forensics
    to detect Real, Tampered, AI-Generated, and Suspicious news imagery.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.model = None
        self.classes = ["fake", "real", "suspicious"]
        self.ai_detector = AIImageDetector()
        self.is_loaded = False

    def load(self):
        """Load trained image forensic classifier pipeline."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Image forensic model not found at {self.model_path}. Train it first.")
        saved = joblib.load(self.model_path)
        self.model = saved["model"]
        self.classes = saved.get("classes", ["fake", "real", "suspicious"])
        self.is_loaded = True

    def predict_image(self, image_input) -> Dict[str, Any]:
        """
        Analyze an image (filepath or PIL Image) and return calibrated prediction,
        forensic metrics, and structured explanation across authenticity,
        tampering, and AI generation.
        """
        if not self.is_loaded:
            self.load()

        try:
            if isinstance(image_input, (str, Path)):
                img = Image.open(image_input).convert("RGB")
            else:
                img = image_input.convert("RGB")
        except Exception as e:
            return {
                "prediction": "SUSPICIOUS",
                "confidence": 0.5,
                "error": f"Failed to decode image: {str(e)}",
                "probabilities": {"fake": 0.33, "real": 0.33, "tampered": 0.33, "ai_generated": 0.0, "suspicious": 0.34},
                "forensic_signals": {},
                "explanation": "Image file could not be verified due to decode/format corruption."
            }

        # 1. CASIA2 Tampering & Splicing Feature Extraction
        features, signals = extract_image_features(img)
        features_2d = features.reshape(1, -1)

        probs = self.model.predict_proba(features_2d)[0]
        prob_dict_raw = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes[:len(probs)])}
        casia_fake = float(prob_dict_raw.get("fake", probs[0]))
        casia_real = float(prob_dict_raw.get("real", probs[1] if len(probs) > 1 else 0.0))
        casia_susp = float(prob_dict_raw.get("suspicious", 0.0))

        # 2. AI-Generated Image Forensics (2D FFT, demosaicing, noise moments, provenance)
        ai_result = self.ai_detector.predict_ai_generated(img)
        ai_prob = float(ai_result["ai_probability"])
        ai_reasons = ai_result["reasons"]
        ai_signals = ai_result["forensic_signals"]

        # 3. Multi-Evidence Joint Probability Distribution
        weight_non_ai = max(0.01, 1.0 - ai_prob)
        p_tampered = casia_fake * weight_non_ai
        p_real = casia_real * weight_non_ai
        p_susp = casia_susp * weight_non_ai

        total_p = p_real + p_tampered + ai_prob + p_susp
        norm_ai = round(ai_prob / total_p, 4)
        norm_tampered = round(p_tampered / total_p, 4)
        norm_real = round(p_real / total_p, 4)
        norm_susp = round(p_susp / total_p, 4)

        prob_dict = {
            "real": norm_real,
            "fake": round(min(1.0, norm_tampered + norm_ai), 4),
            "tampered": norm_tampered,
            "ai_generated": norm_ai,
            "suspicious": norm_susp
        }

        # 4. Forensic rules & anomaly compilation
        anomalies = []
        ela_score = signals["ela_error_score"]
        noise_inc = signals["noise_inconsistency"]
        meta = signals["metadata_forensics"]

        if ela_score > 3.5:
            anomalies.append(f"Elevated compression error residual (ELA: {ela_score}) indicates local recompression or splicing.")
        if noise_inc > 0.45:
            anomalies.append(f"High noise inconsistency across visual regions (variance: {noise_inc}), typical of spliced composites.")
        if meta.get("editing_software_detected"):
            anomalies.append(f"Editing software signature found in metadata ({meta.get('software_name')}).")

        # 5. Final Calibrated Decision
        if norm_ai >= 0.48 or ai_result["is_ai_generated"] or (ai_prob >= 0.50 and ai_prob > casia_fake):
            final_pred = "AI_GENERATED"
            confidence = max(norm_ai, ai_prob)
            explanation = (
                "The AI-media detector identified patterns associated with synthetically generated imagery: " +
                ("; ".join(ai_reasons[:2]) if ai_reasons else "anomalous high-frequency Fourier spectral decay and demosaicing absence.")
            )
        elif norm_tampered >= 0.52:
            final_pred = "FAKE_TAMPERED"
            confidence = norm_tampered
            explanation = f"Detected visual manipulation signals: {'; '.join(anomalies) if anomalies else 'significant forensic feature anomalies'}."
        elif norm_real >= 0.55 and not ai_result["is_ai_generated"] and ai_prob < 0.40:
            final_pred = "REAL"
            confidence = norm_real
            explanation = "Image exhibits consistent compression levels, natural optical noise distribution, and no splicing or AI generation artifacts."
        else:
            final_pred = "SUSPICIOUS"
            confidence = max(norm_susp, 0.55)
            explanation = "Forensic signals show ambiguous compression and frequency patterns; content classified as suspicious."

        return {
            "prediction": final_pred,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "forensic_signals": {
                "ela_error_score": ela_score,
                "noise_inconsistency": noise_inc,
                "sharpness_index": signals["sharpness_index"],
                "metadata": meta,
                "anomalies_detected": anomalies,
                "ai_generated_analysis": {
                    "is_ai_detected": ai_result["is_ai_generated"],
                    "ai_score": ai_prob,
                    "spectral_alpha": ai_signals["spectral_alpha"],
                    "high_frequency_ratio": ai_signals["high_frequency_ratio"],
                    "angular_grid_variance": ai_signals["angular_grid_variance"],
                    "demosaic_residual": ai_signals["demosaic_residual"],
                    "noise_excess_kurtosis": ai_signals["noise_excess_kurtosis"],
                    "ai_reasons": ai_reasons
                }
            },
            "explanation": explanation
        }

