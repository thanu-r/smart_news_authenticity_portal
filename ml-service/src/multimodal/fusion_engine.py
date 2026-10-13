import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.text.transformer_model import TransformerTextClassifier
from src.text.baseline_model import BaselineTextModel
from src.image.image_classifier import ImageForensicClassifier
from src.video.video_classifier import VideoForensicClassifier
from src.source.credibility_analyzer import SourceCredibilityAnalyzer
from src.evidence.evidence_retriever import EvidenceRetriever
from src.multimodal.cross_modal_consistency import CrossModalConsistencyEngine
from src.confidence.calibrator import ConfidenceCalibrator
from src.explanation.explanation_engine import MultiFactorExplanationEngine

class MultimodalDecisionFusionEngine:
    """
    Unified Multimodal Decision Engine.
    Fuses predictions across text, image forensics, video forensics,
    source credibility, evidence retrieval, and cross-modal consistency.
    Enforces conflict resolution and outputs calibrated {REAL, FAKE, SUSPICIOUS} decisions.
    """
    def __init__(self):
        self.text_classifier = TransformerTextClassifier()
        self.image_classifier = ImageForensicClassifier()
        self.video_classifier = VideoForensicClassifier()
        self.source_analyzer = SourceCredibilityAnalyzer()
        self.evidence_retriever = EvidenceRetriever()
        self.cross_modal_engine = CrossModalConsistencyEngine()
        self.calibrator = ConfidenceCalibrator()
        self.explanation_engine = MultiFactorExplanationEngine()
        self.is_loaded = False

    def load(self):
        """Pre-load and initialize model weights."""
        if not self.is_loaded:
            model_dir = Path(__file__).resolve().parent.parent.parent / "models" / "text" / "transformer_v1"
            if model_dir.exists():
                self.text_classifier.load(str(model_dir))
            else:
                self.text_classifier.load()
            self.image_classifier.load()
            self.video_classifier.load()
            self.is_loaded = True

    def verify_content(
        self,
        text: Optional[str] = None,
        title: Optional[str] = None,
        url: Optional[str] = None,
        author: Optional[str] = None,
        model_type: str = "transformer",
        image_input: Optional[Union[str, Path, Image.Image]] = None,
        video_path: Optional[str] = None,
        retrieve_evidence: bool = True
    ) -> Dict[str, Any]:
        """
        Execute end-to-end multimodal news authenticity verification.
        """
        if not self.is_loaded:
            self.load()

        modalities_present = []
        modality_probs = []
        modality_weights = []

        # 1. Text Analysis
        text_result = None
        has_text = bool(text and len(text.strip()) > 10) or bool(title and len(title.strip()) > 5)
        if has_text:
            text_result = self.text_classifier.predict_single(text=text or "", title=title, model_type=model_type)
            modalities_present.append("text")
            p_fake = text_result["probabilities"]["fake"]
            p_real = text_result["probabilities"]["real"]
            p_susp = text_result["probabilities"]["suspicious"]
            modality_probs.append(np.array([p_fake, p_real, p_susp]))
            modality_weights.append(1.5)  # Heavy weight for primary semantic text

        # 2. Image Forensic Analysis
        image_result = None
        if image_input is not None:
            image_result = self.image_classifier.predict_image(image_input)
            modalities_present.append("image")
            p_ai = float(image_result["probabilities"].get("ai_generated", 0.0))
            p_tampered = float(image_result["probabilities"].get("tampered", 0.0))
            p_fake = float(image_result["probabilities"].get("fake", p_tampered + p_ai))
            p_real = float(image_result["probabilities"].get("real", 0.5))
            p_susp = float(image_result["probabilities"].get("suspicious", 0.0))
            if image_result.get("prediction") == "SUSPICIOUS" and p_susp == 0.0:
                p_susp = round(1.0 - abs(p_fake - p_real), 4)
            total = p_fake + p_real + p_susp
            if total > 0:
                p_fake, p_real, p_susp = p_fake / total, p_real / total, p_susp / total
            modality_probs.append(np.array([p_fake, p_real, p_susp]))
            modality_weights.append(1.3)

        # 3. Video Forensic Analysis
        video_result = None
        if video_path and os.path.exists(video_path):
            video_result = self.video_classifier.predict_video(video_path)
            modalities_present.append("video")
            p_fake = video_result["probabilities"].get("fake", 0.33)
            p_real = video_result["probabilities"].get("real", 0.33)
            p_susp = video_result["probabilities"].get("suspicious", 0.34)
            modality_probs.append(np.array([p_fake, p_real, p_susp]))
            modality_weights.append(1.3)

        # 4. Source Credibility Analysis
        source_result = None
        if url or author:
            source_result = self.source_analyzer.evaluate_source(url_or_domain=url, author=author)
            modalities_present.append("source")
            cred = source_result["credibility_score"]
            # Convert source credibility into probability distribution
            if source_result["category"] in ["LOW_CREDIBILITY", "SATIRE"]:
                s_fake, s_real, s_susp = 0.70, 0.10, 0.20
            elif cred >= 0.85:
                s_fake, s_real, s_susp = 0.05, 0.85, 0.10
            else:
                s_fake, s_real, s_susp = 0.25, 0.45, 0.30
            modality_probs.append(np.array([s_fake, s_real, s_susp]))
            modality_weights.append(1.0)

        # 5. External Evidence Retrieval
        evidence_result = None
        if retrieve_evidence and has_text:
            evidence_result = self.evidence_retriever.retrieve_evidence(text=text or "", title=title)
            modalities_present.append("evidence")
            corroboration = evidence_result["corroboration_score"]
            if evidence_result["overall_stance"] == "REFUTED":
                e_fake, e_real, e_susp = 0.80, 0.05, 0.15
            elif evidence_result["overall_stance"] == "CORROBORATED":
                e_fake, e_real, e_susp = 0.08, 0.82, 0.10
            else:
                e_fake, e_real, e_susp = 0.25, 0.35, 0.40
            modality_probs.append(np.array([e_fake, e_real, e_susp]))
            modality_weights.append(1.2)

        # 6. Cross-Modal Consistency Evaluation
        cross_modal_result = self.cross_modal_engine.analyze_cross_modal_consistency(
            headline=title,
            body=text,
            image_input=image_input,
            video_path=video_path,
            evidence_corroboration_score=evidence_result.get("corroboration_score") if evidence_result else None
        )

        # Default fallback if empty input
        if not modality_probs:
            return {
                "prediction": "SUSPICIOUS",
                "confidence": 0.50,
                "probabilities": {"fake": 0.33, "real": 0.33, "suspicious": 0.34},
                "modalities_analyzed": [],
                "explanation": {"executive_summary": "No verifiable content provided."}
            }

        # Dedicated Pure Image Verification (preserves exact trained classifier probabilities and signals)
        if len(modalities_present) == 1 and modalities_present[0] == "image" and image_result:
            final_pred = image_result["prediction"]
            conf = float(image_result["confidence"])
            p_dict = image_result["probabilities"]
            p_fake = float(p_dict.get("fake", 0.0))
            p_real = float(p_dict.get("real", 0.0))
            p_susp = float(p_dict.get("suspicious", 0.0))
            p_ai = float(p_dict.get("ai_generated", 0.0))
            p_tampered = float(p_dict.get("tampered", p_fake))
            
            anomalies = image_result.get("forensic_signals", {}).get("anomalies_detected", [])
            ela_score = image_result.get("forensic_signals", {}).get("ela_error_score", "N/A")
            noise_inc = image_result.get("forensic_signals", {}).get("noise_inconsistency", "N/A")
            ai_analysis = image_result.get("forensic_signals", {}).get("ai_generated_analysis", {})
            
            drivers = [
                f"Visual Forensics Verdict: {final_pred} ({round(conf * 100, 1)}% confidence)",
                f"ELA Residual Error: {ela_score}",
                f"Noise Inconsistency: {noise_inc}"
            ]
            if ai_analysis.get("ai_score") is not None:
                drivers.append(f"AI Generative Score: {round(ai_analysis['ai_score'] * 100, 1)}%")
            if anomalies:
                drivers.extend(anomalies)
            if ai_analysis.get("ai_reasons"):
                drivers.extend(ai_analysis["ai_reasons"][:2])

            explanation = {
                "summary": image_result["explanation"],
                "detailed_reasons": anomalies if anomalies else ["Compression levels, frequency decay, and noise variance patterns analyzed."],
                "driver_factors": drivers
            }
            return {
                "prediction": final_pred,
                "confidence": round(conf, 4),
                "calibrated_confidence": round(conf, 4),
                "probabilities": {
                    "fake": round(p_fake, 4),
                    "real": round(p_real, 4),
                    "tampered": round(p_tampered, 4),
                    "ai_generated": round(p_ai, 4),
                    "suspicious": round(p_susp, 4)
                },
                "modalities_analyzed": ["image"],
                "text_analysis": None,
                "image_forensics": image_result,
                "video_forensics": None,
                "source_credibility": None,
                "evidence_retrieval": None,
                "cross_modal_consistency": None,
                "modality_breakdown": {"image": image_result},
                "modality_weights": {"image": 1.0},
                "explanation": explanation
            }

        # Dedicated Pure Video Verification (preserves exact video classifier probabilities and timestamps)
        if len(modalities_present) == 1 and modalities_present[0] == "video" and video_result:
            final_pred = video_result["prediction"]
            conf = float(video_result["confidence"])
            p_dict = video_result["probabilities"]
            susp_times = video_result.get("suspicious_timestamps", [])
            flicker = video_result.get("temporal_flicker", 0.0)
            deepfake_risk = video_result.get("deepfake_risk", "LOW")
            
            drivers = [
                f"Video Verdict: {final_pred} ({round(conf * 100, 1)}% confidence)",
                f"Temporal Flicker Index: {flicker}",
                f"Deepfake Risk Rating: {deepfake_risk}",
                f"Sampled Frames Analyzed: {video_result.get('frames_analyzed', 0)}"
            ]
            for st in susp_times[:2]:
                drivers.append(f"Discontinuity at {st['timestamp']}s: {', '.join(st.get('reasons', []))}")

            explanation = {
                "summary": video_result["explanation"],
                "detailed_reasons": [video_result["explanation"]] + [
                    f"Timestamp {s['timestamp']}s: {'; '.join(s.get('reasons', []))}" for s in susp_times[:3]
                ],
                "driver_factors": drivers
            }
            return {
                "prediction": final_pred,
                "confidence": round(conf, 4),
                "calibrated_confidence": round(conf, 4),
                "probabilities": p_dict,
                "modalities_analyzed": ["video"],
                "text_analysis": None,
                "image_forensics": None,
                "video_forensics": video_result,
                "source_credibility": None,
                "evidence_retrieval": None,
                "cross_modal_consistency": None,
                "modality_breakdown": {"video": video_result},
                "modality_weights": {"video": 1.0},
                "explanation": explanation
            }

        # Weighted Logit / Probability Fusion
        total_w = sum(modality_weights)
        fused_raw_probs = sum(p * (w / total_w) for p, w in zip(modality_probs, modality_weights))
        
        # Apply Confidence Calibration (Platt / Temperature Scaling)
        calibrated_probs = self.calibrator.calibrate_probabilities(fused_raw_probs)

        # Conflict Detection: Check if modalities disagree sharply
        # e.g., Text says REAL (> 0.70) while Image/Video says FAKE (> 0.70)
        has_cross_modal_conflict = False
        conflict_note = None

        if text_result and image_result:
            t_pred = text_result["prediction"]
            i_pred = image_result["prediction"]
            if t_pred == "REAL" and i_pred in ["FAKE", "AI_GENERATED", "FAKE_TAMPERED", "TAMPERED"]:
                has_cross_modal_conflict = True
                conflict_note = f"Text content appears authentic, but accompanying imagery exhibits {'synthetic AI generative patterns' if i_pred == 'AI_GENERATED' else 'digital tampering/splicing artifacts'}."
            elif t_pred == "FAKE" and i_pred == "REAL":
                has_cross_modal_conflict = True
                conflict_note = "Accompanying image appears authentic, but text content exhibits disinformation markers."

        if text_result and video_result:
            t_pred = text_result["prediction"]
            v_pred = video_result["prediction"]
            if t_pred == "REAL" and v_pred in ["FAKE", "DEEPFAKE", "FAKE_MANIPULATED", "AI_GENERATED"]:
                has_cross_modal_conflict = True
                conflict_note = f"Text content appears authentic, but accompanying video exhibits {'deepfake facial manipulation' if v_pred == 'DEEPFAKE' else 'generative/temporal anomalies'}."

        # Make Final Prediction
        labels = ["fake", "real", "suspicious"]
        best_idx = int(np.argmax(calibrated_probs))
        pred_label = labels[best_idx].upper()
        confidence = float(calibrated_probs[best_idx])

        # Enforce SUSPICIOUS on cross-modal conflict or ambiguous probability
        if has_cross_modal_conflict:
            final_pred = "SUSPICIOUS"
            # Elevate suspicious probability
            calibrated_probs[2] = max(calibrated_probs[2], 0.55)
            calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
            confidence = float(calibrated_probs[2])
        elif confidence < 0.55 and pred_label != "SUSPICIOUS":
            final_pred = "SUSPICIOUS"
            confidence = float(calibrated_probs[2])
        else:
            final_pred = pred_label

        # Multi-Factor Explanation
        explanation = self.explanation_engine.generate_explanation(
            final_prediction=final_pred,
            confidence=confidence,
            text_signals=text_result.get("signals") if text_result else None,
            image_signals=image_result.get("forensic_signals") if image_result else None,
            video_signals=video_result if video_result else None,
            cross_modal_signals=cross_modal_result,
            source_signals=source_result,
            evidence_signals=evidence_result
        )

        if conflict_note:
            explanation["detailed_reasons"].insert(0, f"Cross-Modal Conflict: {conflict_note}")

        return {
            "prediction": final_pred,
            "confidence": round(confidence, 4),
            "probabilities": {
                "fake": round(float(calibrated_probs[0]), 4),
                "real": round(float(calibrated_probs[1]), 4),
                "suspicious": round(float(calibrated_probs[2]), 4)
            },
            "modalities_analyzed": modalities_present,
            "text_analysis": text_result,
            "image_forensics": image_result,
            "video_forensics": video_result,
            "source_credibility": source_result,
            "evidence_retrieval": evidence_result,
            "cross_modal_consistency": cross_modal_result,
            "explanation": explanation
        }
