import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.video.frame_sampler import VideoFrameSampler
from src.image.image_classifier import ImageForensicClassifier

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "video"

class VideoForensicClassifier:
    """
    Multiframe video deepfake and manipulation detector.
    Samples keyframes, evaluates frame-level forensic artifacts,
    aggregates temporal flicker and continuity, and pinpoints suspicious timestamps.
    """
    def __init__(self, target_fps: float = 1.0, max_frames: int = 24):
        self.sampler = VideoFrameSampler(target_fps=target_fps, max_frames=max_frames)
        self.frame_classifier = ImageForensicClassifier()
        self.is_loaded = False

    def load(self):
        """Ensure underlying forensic feature extractors and classifiers are ready."""
        self.frame_classifier.load()
        self.is_loaded = True

    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze a video file and return video-level authenticity prediction,
        per-frame breakdown, suspicious timestamps, and forensic summary.
        """
        if not self.is_loaded:
            self.load()

        try:
            frames, meta = self.sampler.sample_frames(video_path)
        except Exception as e:
            return {
                "prediction": "SUSPICIOUS",
                "confidence": 0.5,
                "error": f"Failed to decode video: {str(e)}",
                "probabilities": {"fake": 0.33, "real": 0.33, "suspicious": 0.34},
                "suspicious_timestamps": [],
                "explanation": "Video file could not be verified due to format or codec decoding error."
            }

        if not frames:
            return {
                "prediction": "SUSPICIOUS",
                "confidence": 0.5,
                "probabilities": {"fake": 0.33, "real": 0.33, "suspicious": 0.34},
                "suspicious_timestamps": [],
                "explanation": "Video has 0 readable frames."
            }

        frame_results = []
        frame_results = []
        suspicious_timestamps = []
        fake_probs = []
        real_probs = []
        ai_probs = []
        temporal_diffs = []
        faces_detected_count = 0

        for f in frames:
            pil_img = f["image"]
            t_sec = f["timestamp"]
            temporal_diff = f["temporal_diff"]
            temporal_diffs.append(temporal_diff)
            if f.get("face_detected"):
                faces_detected_count += 1

            # Analyze frame through enhanced forensic classifier
            res = self.frame_classifier.predict_image(pil_img)
            f_prob = res["probabilities"].get("fake", 0.0)
            r_prob = res["probabilities"].get("real", 0.0)
            ai_prob = res["probabilities"].get("ai_generated", 0.0)

            fake_probs.append(f_prob)
            real_probs.append(r_prob)
            ai_probs.append(ai_prob)

            # Anomaly conditions: AI generation, splicing, or optical flicker discontinuity
            is_ai_frame = ai_prob > 0.50
            is_tampered_frame = f_prob > 0.60
            is_flicker = temporal_diff > 0.35

            if is_ai_frame or is_tampered_frame or is_flicker:
                reasons = []
                if is_ai_frame:
                    reasons.append(f"Synthetic AI generative frame artifacts (prob {ai_prob:.2f})")
                if is_tampered_frame:
                    reasons.append(f"Visual manipulation/splicing artifacts (prob {f_prob:.2f})")
                if is_flicker:
                    reasons.append(f"Inter-frame optical flicker discontinuity ({temporal_diff:.2f})")

                suspicious_timestamps.append({
                    "timestamp": t_sec,
                    "frame_index": f["frame_index"],
                    "anomaly_score": round(max(f_prob, ai_prob, temporal_diff), 3),
                    "face_detected": f["face_detected"],
                    "reasons": reasons
                })

            frame_results.append({
                "timestamp": t_sec,
                "fake_prob": round(f_prob, 3),
                "ai_prob": round(ai_prob, 3),
                "temporal_diff": temporal_diff,
                "face_detected": f["face_detected"]
            })

        # Temporal & Multimodal Frame Aggregation
        avg_fake_prob = float(np.mean(fake_probs)) if fake_probs else 0.5
        avg_real_prob = float(np.mean(real_probs)) if real_probs else 0.5
        avg_ai_prob = float(np.mean(ai_probs)) if ai_probs else 0.0
        max_fake_prob = float(np.max(fake_probs)) if fake_probs else 0.5
        max_ai_prob = float(np.max(ai_probs)) if ai_probs else 0.0
        flicker_index = float(np.std(temporal_diffs)) if temporal_diffs else 0.0
        manipulated_ratio = len(suspicious_timestamps) / max(len(frames), 1)
        ai_frame_ratio = sum(1 for p in ai_probs if p > 0.50) / max(len(frames), 1)

        # Video-level calibrated multi-label decision
        if ai_frame_ratio >= 0.40 or (avg_ai_prob >= 0.48 and avg_ai_prob > avg_fake_prob):
            final_pred = "AI_GENERATED"
            confidence = min(0.96, round(avg_ai_prob * 0.6 + ai_frame_ratio * 0.4, 4))
            deepfake_risk = "CRITICAL"
            explanation = (
                f"Video displays synthetic generative patterns across {int(ai_frame_ratio * 100)}% of sampled frames "
                f"(spectral power deviations, demosaicing anomalies, and synthetic diffusion residual noise)."
            )
        elif manipulated_ratio > 0.30 or (max_fake_prob > 0.85 and len(suspicious_timestamps) >= 2):
            if faces_detected_count > 0:
                final_pred = "DEEPFAKE"
                deepfake_risk = "HIGH"
                explanation = (
                    f"Facial manipulation and temporal boundary flicker detected across {len(suspicious_timestamps)} sampled intervals "
                    f"({manipulated_ratio:.1%} of video) at timestamps: {', '.join([str(s['timestamp']) + 's' for s in suspicious_timestamps[:3]])}."
                )
            else:
                final_pred = "FAKE_MANIPULATED"
                deepfake_risk = "HIGH"
                explanation = (
                    f"Video exhibits visual manipulation artifacts across {len(suspicious_timestamps)} sampled intervals "
                    f"({manipulated_ratio:.1%} of video) at timestamps: {', '.join([str(s['timestamp']) + 's' for s in suspicious_timestamps[:3]])}."
                )
            confidence = min(0.95, round(avg_fake_prob * 0.5 + manipulated_ratio * 0.5, 4))
        elif manipulated_ratio > 0.10 or flicker_index > 0.15 or (0.35 <= avg_ai_prob < 0.48):
            final_pred = "SUSPICIOUS"
            confidence = 0.66
            deepfake_risk = "MEDIUM"
            explanation = (
                f"Temporal consistency and frequency checking identified isolated anomalies at "
                f"{len(suspicious_timestamps)} timestamps, but insufficient overall consensus to classify as definitely fake."
            )
        else:
            final_pred = "REAL"
            confidence = min(0.94, round(avg_real_prob, 4))
            deepfake_risk = "LOW"
            explanation = "Sampled video frames show stable temporal continuity, natural optical camera motion, and no deepfake or synthetic generation artifacts."

        suspicious_prob = round(max(0.04, 1.0 - (avg_real_prob + avg_fake_prob + avg_ai_prob) / 3.0), 4)
        total_p = avg_real_prob + avg_fake_prob + avg_ai_prob + suspicious_prob
        normalized_probs = {
            "real": round(avg_real_prob / total_p, 4),
            "fake": round(avg_fake_prob / total_p, 4),
            "tampered": round(avg_fake_prob / total_p, 4),
            "ai_generated": round(avg_ai_prob / total_p, 4),
            "suspicious": round(suspicious_prob / total_p, 4)
        }

        return {
            "prediction": final_pred,
            "confidence": confidence,
            "probabilities": normalized_probs,
            "frames_analyzed": len(frames),
            "temporal_flicker": round(flicker_index, 4),
            "deepfake_risk": deepfake_risk,
            "video_metadata": meta,
            "suspicious_timestamps": suspicious_timestamps,
            "temporal_signals": {
                "flicker_index": round(flicker_index, 4),
                "manipulated_ratio": round(manipulated_ratio, 4),
                "ai_frame_ratio": round(ai_frame_ratio, 4),
                "max_frame_fake_prob": round(max_fake_prob, 4),
                "max_frame_ai_prob": round(max_ai_prob, 4),
                "total_frames_sampled": len(frames)
            },
            "explanation": explanation
        }
