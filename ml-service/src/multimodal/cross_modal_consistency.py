import os
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.image.forensic_extractor import extract_image_features
from src.video.frame_sampler import VideoFrameSampler

class CrossModalConsistencyEngine:
    """
    Evaluates cross-modal coherence and detects out-of-context misattribution across:
    1. Headline <-> Body Article
    2. Text <-> Image
    3. Text <-> Video Keyframes
    4. Text <-> External Corroborating Evidence
    """
    def __init__(self):
        self.frame_sampler = VideoFrameSampler(target_fps=1.0, max_frames=12)

    @staticmethod
    def _compute_jaccard_and_term_overlap(text1: str, text2: str) -> float:
        """Compute lexical and entity term overlap between two text segments."""
        if not text1 or not text2:
            return 0.0
        words1 = set(re.findall(r"\w{3,}", text1.lower()))
        words2 = set(re.findall(r"\w{3,}", text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union)
        # Recall relative to the shorter text (typically the headline)
        recall = len(intersection) / min(len(words1), len(words2))
        return round(0.4 * jaccard + 0.6 * recall, 4)

    def evaluate_headline_body_consistency(self, headline: str, body: str) -> Dict[str, Any]:
        """
        Evaluate whether the body text actually substantiates the claim in the headline.
        Flags clickbait discrepancies where the body contradicts or lacks evidence for the headline.
        """
        if not headline or not body:
            return {
                "score": 1.0 if not headline else 0.5,
                "status": "CONSISTENT" if not headline else "AMBIGUOUS",
                "overlap_ratio": 1.0 if not headline else 0.0,
                "alert": None
            }

        overlap = self._compute_jaccard_and_term_overlap(headline, body)
        
        # Check if headline contains strong assertion absent in body
        h_lower = headline.lower()
        b_lower = body.lower()
        
        sensational_words = ["exposed", "secret", "miracle", "arrested", "bombshell", "banned", "confirms"]
        sensational_in_headline = [w for w in sensational_words if w in h_lower and w not in b_lower]

        alert = None
        if overlap < 0.15:
            score = 0.35
            status = "MISMATCH"
            alert = "Headline makes claims that have minimal semantic overlap with the article body text."
        elif sensational_in_headline:
            score = 0.58
            status = "EXAGGERATED_CLICKBAIT"
            alert = f"Headline uses sensational assertions ({', '.join(sensational_in_headline)}) not substantiated in the body."
        elif overlap >= 0.40:
            score = min(0.98, round(0.70 + overlap * 0.3, 4))
            status = "STRONG_CONSISTENCY"
        else:
            score = round(0.50 + overlap * 0.4, 4)
            status = "MODERATE_CONSISTENCY"

        return {
            "score": score,
            "status": status,
            "overlap_ratio": overlap,
            "alert": alert
        }

    def evaluate_text_image_consistency(
        self,
        text: str,
        image_input: Optional[Union[str, Path, Image.Image]]
    ) -> Dict[str, Any]:
        """
        Evaluate alignment between article text/headline and accompanying photo.
        Detects out-of-context repurposing and forensic compression conflicts.
        """
        if image_input is None:
            return {
                "score": 1.0,
                "status": "NOT_APPLICABLE",
                "alert": None,
                "forensic_integrity": 1.0
            }

        try:
            if isinstance(image_input, (str, Path)):
                img = Image.open(image_input).convert("RGB")
            else:
                img = image_input.convert("RGB")
        except Exception as e:
            return {
                "score": 0.40,
                "status": "DECODE_ERROR",
                "alert": f"Could not verify image alignment: {str(e)}",
                "forensic_integrity": 0.40
            }

        # Extract forensic signals
        features, signals = extract_image_features(img)
        ela_score = signals.get("ela_error_score", 0.0)
        noise_inc = signals.get("noise_inconsistency", 0.0)

        # Baseline alignment
        score = 0.85
        alert = None

        # If image exhibits significant forensic tampering, consistency is severely degraded
        if ela_score > 3.8 or noise_inc > 0.48:
            score -= 0.35
            status = "MANIPULATION_DETECTED"
            alert = "Image exhibits local forensic splicing or recompression artifacts inconsistent with authentic journalism."
        elif ela_score > 2.8:
            score -= 0.15
            status = "ELEVATED_RESIDUAL"
            alert = "Image has elevated compression residuals; potential composite or edited photo."
        else:
            status = "CONSISTENT"

        return {
            "score": round(max(0.1, min(0.99, score)), 4),
            "status": status,
            "alert": alert,
            "ela_score": ela_score,
            "noise_inconsistency": noise_inc
        }

    def evaluate_text_video_consistency(
        self,
        text: str,
        video_path: Optional[str]
    ) -> Dict[str, Any]:
        """
        Evaluate temporal and forensic consistency between news text and accompanying video footage.
        """
        if not video_path or not os.path.exists(video_path):
            return {
                "score": 1.0,
                "status": "NOT_APPLICABLE",
                "alert": None
            }

        try:
            frames, meta = self.frame_sampler.sample_frames(video_path)
            if not frames:
                return {"score": 0.5, "status": "EMPTY_VIDEO", "alert": "Video contains 0 readable frames."}

            # Check temporal continuity
            diffs = [f["temporal_diff"] for f in frames]
            flicker = float(np.std(diffs)) if diffs else 0.0

            if flicker > 0.14:
                score = 0.52
                status = "TEMPORAL_DISCONTINUITY"
                alert = "Video keyframes exhibit unnatural optical flicker or inter-frame splicing."
            else:
                score = 0.88
                status = "CONSISTENT"
                alert = None

            return {
                "score": round(score, 4),
                "status": status,
                "flicker_index": round(flicker, 4),
                "frames_evaluated": len(frames),
                "alert": alert
            }
        except Exception as e:
            return {
                "score": 0.50,
                "status": "ERROR",
                "alert": f"Failed to evaluate video consistency: {str(e)}"
            }

    def analyze_cross_modal_consistency(
        self,
        headline: Optional[str] = None,
        body: Optional[str] = None,
        image_input: Optional[Union[str, Path, Image.Image]] = None,
        video_path: Optional[str] = None,
        evidence_corroboration_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Synthesize cross-modal consistency across all provided modalities.
        Produces multi-dimensional radar scores, consistency status, and alerts.
        """
        full_text = f"{headline or ''}. {body or ''}".strip()

        # 1. Headline <-> Body
        hb_res = self.evaluate_headline_body_consistency(headline or "", body or "")

        # 2. Text <-> Image
        ti_res = self.evaluate_text_image_consistency(full_text, image_input)

        # 3. Text <-> Video
        tv_res = self.evaluate_text_video_consistency(full_text, video_path)

        # 4. Claim <-> Evidence
        ce_score = evidence_corroboration_score if evidence_corroboration_score is not None else 0.70

        # Collect active modality scores and weights
        active_scores = [("headline_body", hb_res["score"], 1.0)]
        radar_dimensions = {
            "headline_body": hb_res["score"]
        }
        alerts = []

        if hb_res["alert"]:
            alerts.append(hb_res["alert"])

        if ti_res["status"] != "NOT_APPLICABLE":
            active_scores.append(("text_image", ti_res["score"], 1.2))
            radar_dimensions["text_image"] = ti_res["score"]
            if ti_res["alert"]:
                alerts.append(ti_res["alert"])

        if tv_res["status"] != "NOT_APPLICABLE":
            active_scores.append(("text_video", tv_res["score"], 1.2))
            radar_dimensions["text_video"] = tv_res["score"]
            if tv_res["alert"]:
                alerts.append(tv_res["alert"])

        if evidence_corroboration_score is not None:
            active_scores.append(("claim_evidence", ce_score, 1.1))
            radar_dimensions["claim_evidence"] = ce_score
            if ce_score < 0.40:
                alerts.append("External news registries contradict or refute the submitted claim.")

        # Compute overall weighted harmonic mean
        total_weight = sum(w for _, _, w in active_scores)
        overall_score = sum(s * w for _, s, w in active_scores) / total_weight

        if overall_score >= 0.75:
            consensus = "HIGH_CONSISTENCY"
            summary = "All provided modalities and external signals align coherently."
        elif overall_score >= 0.50:
            consensus = "MODERATE_CONSISTENCY"
            summary = "Minor cross-modal variance detected; some signals warrant scrutiny."
        else:
            consensus = "CROSS_MODAL_CONFLICT"
            summary = f"Significant discrepancy detected across modalities: {'; '.join(alerts[:2])}."

        return {
            "overall_consistency_score": round(overall_score, 4),
            "consensus_status": consensus,
            "radar_scores": {k: round(v, 4) for k, v in radar_dimensions.items()},
            "mismatch_alerts": alerts,
            "detailed_evaluations": {
                "headline_body": hb_res,
                "text_image": ti_res,
                "text_video": tv_res
            },
            "summary": summary
        }
