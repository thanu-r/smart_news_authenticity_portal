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
    Multiframe video deepfake, AI-generated, and manipulation detector.
    Samples keyframes across MP4, MOV, AVI, and WEBM video streams,
    evaluates frame-level forensic artifacts, tracks facial continuity,
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
        Supports MP4, MOV, AVI, WEBM.
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
                "probabilities": {"fake": 0.33, "real": 0.33, "tampered": 0.33, "ai_generated": 0.0, "suspicious": 0.34},
                "suspicious_timestamps": [],
                "explanation": f"Video file could not be verified due to format or codec decoding error: {str(e)}"
            }

        if not frames:
            return {
                "prediction": "SUSPICIOUS",
                "confidence": 0.5,
                "probabilities": {"fake": 0.33, "real": 0.33, "tampered": 0.33, "ai_generated": 0.0, "suspicious": 0.34},
                "suspicious_timestamps": [],
                "explanation": "Video has 0 readable frames."
            }

        frame_results = []
        suspicious_timestamps = []
        fake_probs = []
        real_probs = []
        ai_probs = []
        deepfake_probs = []
        temporal_diffs = []
        face_jitters = []
        faces_detected_count = 0

        for f in frames:
            pil_img = f["image"]
            t_sec = f["timestamp"]
            temporal_diff = f["temporal_diff"]
            face_jitter = f.get("face_jitter", 0.0)
            has_face = f.get("face_detected", False)
            
            temporal_diffs.append(temporal_diff)
            face_jitters.append(face_jitter)
            if has_face:
                faces_detected_count += 1

            # Frame-level deep analysis through enhanced forensic classifier
            res = self.frame_classifier.predict_image(pil_img, is_video_frame=True)
            f_prob = res["probabilities"].get("fake", 0.0)
            r_prob = res["probabilities"].get("real", 0.0)
            ai_prob = res["probabilities"].get("ai_generated", 0.0)
            pred_type = res.get("prediction", "")

            is_df_frame = bool(pred_type == "DEEPFAKE" or (has_face and face_jitter > 0.22))
            df_score = 0.85 if is_df_frame else 0.05
            deepfake_probs.append(df_score)

            fake_probs.append(f_prob)
            real_probs.append(r_prob)
            ai_probs.append(ai_prob)

            # Anomaly checks
            is_ai_frame = ai_prob > 0.48
            is_tampered_frame = f_prob > 0.55
            is_flicker = temporal_diff > 0.25

            if is_ai_frame or is_tampered_frame or is_df_frame or is_flicker:
                reasons = []
                if is_df_frame:
                    reasons.append(f"Facial boundary warping / temporal jitter ({face_jitter:.2f})")
                if is_ai_frame:
                    reasons.append(f"Synthetic AI generative frame artifacts (prob {ai_prob:.2f})")
                if is_tampered_frame:
                    reasons.append(f"Visual manipulation/splicing ELA residual (prob {f_prob:.2f})")
                if is_flicker:
                    reasons.append(f"Inter-frame optical flicker discontinuity ({temporal_diff:.2f})")

                anomaly_score = max(f_prob, ai_prob, df_score, min(1.0, temporal_diff * 1.5))
                suspicious_timestamps.append({
                    "timestamp": t_sec,
                    "frame_index": f["frame_index"],
                    "anomaly_score": round(float(anomaly_score), 3),
                    "face_detected": has_face,
                    "reasons": reasons
                })

            frame_results.append({
                "timestamp": t_sec,
                "fake_prob": round(f_prob, 3),
                "ai_prob": round(ai_prob, 3),
                "deepfake_score": round(df_score, 3),
                "temporal_diff": temporal_diff,
                "face_jitter": face_jitter,
                "face_detected": has_face
            })

        # Aggregation across all sampled frames
        num_frames = max(len(frames), 1)
        avg_fake_prob = float(np.mean(fake_probs)) if fake_probs else 0.5
        avg_real_prob = float(np.mean(real_probs)) if real_probs else 0.5
        avg_ai_prob = float(np.mean(ai_probs)) if ai_probs else 0.0
        avg_df_prob = float(np.mean(deepfake_probs)) if deepfake_probs else 0.0
        max_fake_prob = float(np.max(fake_probs)) if fake_probs else 0.5
        max_ai_prob = float(np.max(ai_probs)) if ai_probs else 0.0
        raw_mean_flicker = meta.get("mean_optical_flicker", 0.0) / 255.0
        flicker_index = float(max(np.std(temporal_diffs), np.max(temporal_diffs) * 0.5, raw_mean_flicker)) if temporal_diffs else raw_mean_flicker
        max_face_jitter = float(np.max(face_jitters)) if face_jitters else 0.0

        ai_frame_ratio = sum(1 for p in ai_probs if p > 0.48) / num_frames
        df_frame_ratio = sum(1 for p in deepfake_probs if p > 0.50) / num_frames
        tampered_ratio = sum(1 for p in fake_probs if p > 0.55) / num_frames
        manipulated_ratio = len(suspicious_timestamps) / num_frames

        optical_flicker = meta.get("mean_optical_flicker", 0.0)
        face_ratio_var = meta.get("face_motion_ratio_variance", 0.0)

        # Video-level multi-label decision
        if ai_frame_ratio >= 0.35 or (avg_ai_prob >= 0.46 and avg_ai_prob > avg_fake_prob):
            final_pred = "AI_GENERATED"
            confidence = min(0.95, round(avg_ai_prob * 0.5 + ai_frame_ratio * 0.5, 4))
            deepfake_risk = "CRITICAL"
            explanation = (
                f"Video displays synthetic generative patterns across {int(ai_frame_ratio * 100)}% of sampled frames "
                f"(spectral power deviations, demosaicing absence, and synthetic diffusion residual noise)."
            )
        elif (df_frame_ratio >= 0.35 and faces_detected_count > 0) or (face_ratio_var > 0.0018 and optical_flicker > 20.0) or (faces_detected_count > 0 and optical_flicker > 25.0 and max_face_jitter > 0.20):
            final_pred = "DEEPFAKE"
            confidence = min(0.94, max(0.72, 0.60 + min(0.30, face_ratio_var * 50.0 + (optical_flicker / 100.0))))
            deepfake_risk = "CRITICAL"
            ts_summary = ", ".join([f"{s['timestamp']}s" for s in suspicious_timestamps[:3]])
            explanation = (
                f"Facial manipulation and boundary blending jitter detected across human subjects "
                f"(facial motion variance: {face_ratio_var:.4f}, optical flicker: {optical_flicker:.1f}) "
                f"at timestamps: {ts_summary if ts_summary else 'throughout duration'}."
            )
        elif tampered_ratio >= 0.30 or (max_fake_prob > 0.80 and len(suspicious_timestamps) >= 2) or optical_flicker > 30.0:
            final_pred = "FAKE_MANIPULATED"
            confidence = min(0.92, round(avg_fake_prob * 0.5 + tampered_ratio * 0.5, 4))
            deepfake_risk = "HIGH"
            ts_summary = ", ".join([f"{s['timestamp']}s" for s in suspicious_timestamps[:3]])
            explanation = (
                f"Video exhibits visual manipulation/splicing artifacts across {len(suspicious_timestamps)} sampled intervals "
                f"({int(manipulated_ratio * 100)}% of duration) at timestamps: {ts_summary if ts_summary else 'multiple cuts'}."
            )
        elif manipulated_ratio >= 0.15 or flicker_index > 0.16 or (0.32 <= avg_ai_prob < 0.46):
            final_pred = "SUSPICIOUS"
            confidence = 0.65
            deepfake_risk = "MEDIUM"
            explanation = (
                f"Temporal consistency and frequency analysis identified isolated anomalies at "
                f"{len(suspicious_timestamps)} timestamps, but insufficient consensus to confirm deliberate manipulation."
            )
        else:
            final_pred = "REAL"
            confidence = min(0.94, round(avg_real_prob, 4))
            deepfake_risk = "LOW"
            explanation = (
                f"Sampled video frames ({len(frames)} frames across {meta.get('duration_seconds', 0)}s) show stable temporal continuity, "
                f"natural optical camera motion (flicker index: {optical_flicker:.1f}), and no deepfake or synthetic generation artifacts."
            )

        suspicious_prob = round(max(0.04, 1.0 - (avg_real_prob + avg_fake_prob + avg_ai_prob) / 3.0), 4)
        total_p = avg_real_prob + avg_fake_prob + avg_ai_prob + suspicious_prob
        normalized_probs = {
            "real": round(avg_real_prob / total_p, 4),
            "fake": round(min(1.0, (avg_fake_prob + avg_ai_prob + avg_df_prob) / total_p), 4),
            "tampered": round(avg_fake_prob / total_p, 4),
            "ai_generated": round(avg_ai_prob / total_p, 4),
            "deepfake": round(avg_df_prob / total_p, 4),
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
                "df_frame_ratio": round(df_frame_ratio, 4),
                "max_face_jitter": round(max_face_jitter, 4),
                "max_frame_fake_prob": round(max_fake_prob, 4),
                "max_frame_ai_prob": round(max_ai_prob, 4),
                "total_frames_sampled": len(frames)
            },
            "explanation": explanation
        }
