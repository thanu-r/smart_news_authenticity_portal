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
        tampered_probs = []

        for f in frames:
            pil_img = f["image"]
            t_sec = f["timestamp"]
            temporal_diff = f["temporal_diff"]
            face_jitter = f.get("face_jitter", 0.0)
            has_face = f.get("face_detected", False)
            
            temporal_diffs.append(temporal_diff)
            face_jitters.append(face_jitter)

            # Frame-level deep analysis through enhanced forensic classifier
            res = self.frame_classifier.predict_image(pil_img, is_video_frame=True)
            t_prob = res["probabilities"].get("tampered", 0.0)
            f_prob = res["probabilities"].get("fake", 0.0)
            r_prob = res["probabilities"].get("real", 0.0)
            ai_prob = res["probabilities"].get("ai_generated", 0.0)
            pred_type = res.get("prediction", "")
            
            face_info = res.get("forensic_signals", {}).get("face_forensics", {})
            has_face_forensics = bool(face_info.get("face_detected", 0.0) == 1.0)
            if has_face or has_face_forensics:
                faces_detected_count += 1

            face_seam = float(face_info.get("boundary_seam_discontinuity", 0.0))
            face_noise_r = float(face_info.get("face_noise_ratio", 1.0))
            is_df_frame = bool(
                pred_type == "DEEPFAKE" or
                (has_face_forensics and (face_seam > 1.85 or face_noise_r > 2.4 or face_noise_r < 0.38))
            )
            df_score = 0.85 if is_df_frame else 0.05
            deepfake_probs.append(df_score)

            tampered_probs.append(t_prob)
            fake_probs.append(f_prob)
            real_probs.append(r_prob)
            ai_probs.append(ai_prob)

            # Anomaly checks
            is_ai_frame = ai_prob > 0.40
            is_tampered_frame = t_prob > 0.65 or (t_prob > 0.55 and len(res.get("forensic_signals", {}).get("anomalies_detected", [])) > 0)
            is_flicker = temporal_diff > 0.25

            if is_ai_frame or is_tampered_frame or is_df_frame or is_flicker:
                reasons = []
                if is_df_frame:
                    reasons.append(f"Facial boundary blending seam discontinuity ({face_seam:.2f})")
                if is_ai_frame:
                    reasons.append(f"Synthetic AI generative frame artifacts (prob {ai_prob:.2f})")
                if is_tampered_frame:
                    reasons.append(f"Visual manipulation/splicing ELA residual (prob {t_prob:.2f})")
                if is_flicker:
                    reasons.append(f"Inter-frame optical flicker discontinuity ({temporal_diff:.2f})")

                anomaly_score = max(t_prob, ai_prob, df_score, min(1.0, temporal_diff * 1.5))
                suspicious_timestamps.append({
                    "timestamp": t_sec,
                    "frame_index": f["frame_index"],
                    "anomaly_score": round(float(anomaly_score), 3),
                    "face_detected": has_face or has_face_forensics,
                    "reasons": reasons
                })

            frame_results.append({
                "timestamp": t_sec,
                "fake_prob": round(f_prob, 3),
                "tampered_prob": round(t_prob, 3),
                "ai_prob": round(ai_prob, 3),
                "deepfake_score": round(df_score, 3),
                "temporal_diff": temporal_diff,
                "face_jitter": face_jitter,
                "face_detected": has_face or has_face_forensics
            })

        # Aggregation across all sampled frames
        num_frames = max(len(frames), 1)
        avg_fake_prob = float(np.mean(fake_probs)) if fake_probs else 0.5
        avg_tampered_prob = float(np.mean(tampered_probs)) if tampered_probs else 0.5
        avg_real_prob = float(np.mean(real_probs)) if real_probs else 0.5
        avg_ai_prob = float(np.mean(ai_probs)) if ai_probs else 0.0
        avg_df_prob = float(np.mean(deepfake_probs)) if deepfake_probs else 0.0
        max_tampered_prob = float(np.max(tampered_probs)) if tampered_probs else 0.5
        max_ai_prob = float(np.max(ai_probs)) if ai_probs else 0.0
        raw_mean_flicker = meta.get("mean_optical_flicker", 0.0) / 255.0
        flicker_index = float(max(np.std(temporal_diffs), np.max(temporal_diffs) * 0.5, raw_mean_flicker)) if temporal_diffs else raw_mean_flicker
        max_face_jitter = float(np.max(face_jitters)) if face_jitters else 0.0

        ai_frame_ratio = sum(1 for p in ai_probs if p >= 0.48) / num_frames
        df_frame_ratio = sum(1 for p in deepfake_probs if p > 0.50) / num_frames
        tampered_ratio = sum(1 for p in tampered_probs if p > 0.65) / num_frames
        manipulated_ratio = len(suspicious_timestamps) / num_frames

        optical_flicker = meta.get("mean_optical_flicker", 0.0)
        face_ratio_var = meta.get("face_motion_ratio_variance", 0.0)
        mean_warp = meta.get("mean_warp_residual", 0.0)
        mean_prnu = meta.get("mean_temporal_prnu_corr", 0.0)
        container_analysis = meta.get("container_analysis", {})

        # Video-level multi-label decision hierarchy

        has_ai_meta = bool(container_analysis.get("has_ai_metadata", False))
        is_generative_warp = bool(mean_warp > 2.2 and mean_prnu < 0.25)
        is_generative_frames = bool(ai_frame_ratio >= 0.30 or (avg_ai_prob >= 0.44 and avg_ai_prob > avg_real_prob) or (max_ai_prob > 0.65 and mean_warp > 1.8))
        is_deepfake = bool(faces_detected_count > 0 and (df_frame_ratio >= 0.25 or (face_ratio_var > 0.0018 and optical_flicker > 20.0)))

        # 1. AI-Generated Video via Container Metadata
        if has_ai_meta:
            final_pred = "AI_GENERATED"
            confidence = 0.96
            deepfake_risk = "CRITICAL"
            explanation = f"Video identified as AI-GENERATED: AI generator signatures identified in container metadata ({', '.join(container_analysis.get('detected_signatures', []))})."

        # 2. Deepfake / Face-Swap Video Detection (Human subjects with boundary seam / motion jitter)
        elif is_deepfake:
            final_pred = "DEEPFAKE"
            confidence = min(0.94, max(0.72, 0.60 + min(0.30, face_ratio_var * 50.0 + (optical_flicker / 100.0))))
            deepfake_risk = "CRITICAL"
            ts_summary = ", ".join([f"{s['timestamp']}s" for s in suspicious_timestamps[:3]])
            explanation = (
                f"Facial manipulation and boundary blending jitter detected across human subjects "
                f"(facial motion variance: {face_ratio_var:.4f}, optical flicker: {optical_flicker:.1f}) "
                f"at timestamps: {ts_summary if ts_summary else 'throughout duration'}."
            )

        # 3. AI-Generated Video Detection (Non-rigid generative optical flow warping or diffusion latents)
        elif is_generative_warp or is_generative_frames:
            final_pred = "AI_GENERATED"
            confidence = min(0.96, max(0.75, round(avg_ai_prob * 0.4 + ai_frame_ratio * 0.4 + 0.15, 4)))
            deepfake_risk = "CRITICAL"
            reasons = []
            if is_generative_warp:
                reasons.append(f"non-rigid generative motion warping across frames (optical flow warp residual: {mean_warp:.2f})")
            if is_generative_frames:
                reasons.append(f"synthetic generative latent artifacts across {int(ai_frame_ratio * 100)}% of sampled frames (heavy-tailed noise kurtosis, spectral power deviation)")
            explanation = f"Video identified as AI-GENERATED: {'; '.join(reasons)}."

        # 3. Tampered / Spliced Cut Video Detection
        elif tampered_ratio >= 0.30 or (max_tampered_prob > 0.75 and len(suspicious_timestamps) >= 2) or optical_flicker > 30.0:
            final_pred = "FAKE_MANIPULATED"
            confidence = min(0.92, round(avg_tampered_prob * 0.5 + tampered_ratio * 0.5, 4))
            deepfake_risk = "HIGH"
            ts_summary = ", ".join([f"{s['timestamp']}s" for s in suspicious_timestamps[:3]])
            explanation = (
                f"Video exhibits visual manipulation/splicing artifacts across {len(suspicious_timestamps)} sampled intervals "
                f"({int(manipulated_ratio * 100)}% of duration) at timestamps: {ts_summary if ts_summary else 'multiple cuts'}."
            )

        # 4. Authentic Camera Video (Requires positive evidence of physical camera recording)
        elif (mean_warp < 1.5 and mean_prnu >= 0.04 and avg_ai_prob < 0.32 and not has_ai_meta and df_frame_ratio < 0.25 and tampered_ratio < 0.25):
            final_pred = "REAL"
            confidence = min(0.95, max(0.70, round(avg_real_prob, 4)))
            deepfake_risk = "LOW"
            explanation = (
                f"Sampled video frames ({len(frames)} frames across {meta.get('duration_seconds', 0)}s) show stable temporal continuity, "
                f"natural optical camera motion (flicker: {optical_flicker:.1f}, warp residual: {mean_warp:.2f}), "
                f"consistent temporal sensor pattern noise (correlation: {mean_prnu:.2f}), and no synthetic generation or splicing artifacts."
            )

        # 5. Borderline / Ambiguous / Compression Uncertainty
        else:
            final_pred = "SUSPICIOUS"
            confidence = 0.62
            deepfake_risk = "MEDIUM"
            explanation = (
                f"Video analysis identified subtle temporal or compression anomalies across {len(suspicious_timestamps)} intervals. "
                f"Evidence is insufficient to confirm deliberate manipulation, but lacks definitive markers of pristine camera capture."
            )

        # Normalized Probability Distribution
        if final_pred == "AI_GENERATED":
            p_ai = max(0.72, avg_ai_prob)
            p_df = 0.10 if faces_detected_count > 0 else 0.03
            p_tampered = 0.10
            p_real = min(0.12, avg_real_prob * 0.2)
            p_susp = 0.05
        elif final_pred == "DEEPFAKE":
            p_ai = 0.10
            p_df = max(0.72, avg_df_prob)
            p_tampered = 0.12
            p_real = min(0.10, avg_real_prob * 0.2)
            p_susp = 0.05
        elif final_pred == "FAKE_MANIPULATED":
            p_ai = 0.10
            p_df = 0.08
            p_tampered = max(0.70, avg_fake_prob)
            p_real = min(0.15, avg_real_prob * 0.2)
            p_susp = 0.07
        elif final_pred == "REAL":
            p_ai = min(0.08, avg_ai_prob)
            p_df = 0.04
            p_tampered = min(0.08, avg_fake_prob)
            p_real = max(0.75, avg_real_prob)
            p_susp = 0.05
        else:
            p_ai = max(0.20, avg_ai_prob)
            p_df = 0.15
            p_tampered = max(0.25, avg_fake_prob)
            p_real = max(0.25, avg_real_prob)
            p_susp = 0.40

        total_p = p_real + p_tampered + p_ai + p_df + p_susp
        normalized_probs = {
            "real": round(p_real / total_p, 4),
            "fake": round(min(1.0, (p_tampered + p_ai + p_df) / total_p), 4),
            "tampered": round(p_tampered / total_p, 4),
            "ai_generated": round(p_ai / total_p, 4),
            "deepfake": round(p_df / total_p, 4),
            "suspicious": round(p_susp / total_p, 4)
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
                "mean_warp_residual": round(mean_warp, 4),
                "mean_temporal_prnu_corr": round(mean_prnu, 4),
                "manipulated_ratio": round(manipulated_ratio, 4),
                "ai_frame_ratio": round(ai_frame_ratio, 4),
                "df_frame_ratio": round(df_frame_ratio, 4),
                "max_face_jitter": round(max_face_jitter, 4),
                "max_frame_fake_prob": round(max_tampered_prob, 4),
                "max_frame_ai_prob": round(max_ai_prob, 4),
                "has_ai_container_metadata": has_ai_meta,
                "total_frames_sampled": len(frames)
            },
            "explanation": explanation
        }
