from typing import Dict, Any, List, Optional

class MultiFactorExplanationEngine:
    """
    Generates transparent, structured natural-language explanations for verification decisions.
    Synthesizes signals across text linguistics, media forensics, cross-modal coherence,
    and external fact-checking evidence into human-interpretable reasoning.
    """

    def generate_explanation(
        self,
        final_prediction: str,
        confidence: float,
        text_signals: Optional[Dict[str, Any]] = None,
        image_signals: Optional[Dict[str, Any]] = None,
        video_signals: Optional[Dict[str, Any]] = None,
        cross_modal_signals: Optional[Dict[str, Any]] = None,
        source_signals: Optional[Dict[str, Any]] = None,
        evidence_signals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assemble comprehensive multi-factor explanation report.
        """
        key_factors = []
        detailed_bullets = []

        # 1. Text & Linguistic Signals
        if text_signals:
            detected_cues = text_signals.get("detected_cues", [])
            sensational = text_signals.get("sensational_score", 0.0)
            if detected_cues:
                bullet = f"Sensational or clickbait lexical markers detected: {', '.join(detected_cues[:4])}."
                detailed_bullets.append(bullet)
                key_factors.append("Sensational language patterns")
            elif sensational > 0.40:
                detailed_bullets.append("Elevated stylistic sensationalism index detected in vocabulary and phrasing.")
            else:
                detailed_bullets.append("Text exhibits neutral, journalistic tone without clickbait cues.")

        # 2. Image Forensic Signals
        if image_signals:
            anomalies = image_signals.get("anomalies_detected", [])
            ela_score = image_signals.get("ela_error_score", 0.0)
            if anomalies:
                for a in anomalies:
                    detailed_bullets.append(f"Image Forensic Analysis: {a}")
                key_factors.append("Visual manipulation artifacts")
            elif ela_score > 3.0:
                detailed_bullets.append(f"Image exhibits moderate compression residual anomalies (ELA score: {ela_score:.2f}).")
            else:
                detailed_bullets.append("Image forensics: compression levels and noise variance are consistent with authentic capture.")

        # 3. Video Forensic Signals
        if video_signals:
            susp_timestamps = video_signals.get("suspicious_timestamps", [])
            flicker = video_signals.get("temporal_signals", {}).get("flicker_index", 0.0)
            if susp_timestamps:
                ts_str = ", ".join([f"{s['timestamp']}s" for s in susp_timestamps[:3]])
                detailed_bullets.append(f"Video Forensic Analysis: Temporal discontinuity or manipulation detected at timestamps {ts_str}.")
                key_factors.append("Video temporal discontinuities")
            elif flicker > 0.12:
                detailed_bullets.append(f"Video exhibits elevated inter-frame optical flicker (index: {flicker:.3f}).")
            else:
                detailed_bullets.append("Video forensics: smooth temporal continuity and stable inter-frame motion across all sampled frames.")

        # 4. Cross-Modal Consistency
        if cross_modal_signals:
            alerts = cross_modal_signals.get("mismatch_alerts", [])
            consensus = cross_modal_signals.get("consensus_status", "HIGH_CONSISTENCY")
            if alerts:
                for alt in alerts:
                    detailed_bullets.append(f"Cross-Modal Alert: {alt}")
                key_factors.append("Cross-modal discrepancy")
            elif consensus == "HIGH_CONSISTENCY":
                detailed_bullets.append("Cross-modal evaluation: headline, body text, and media elements demonstrate high mutual coherence.")

        # 5. Source Credibility
        if source_signals:
            cred_score = source_signals.get("credibility_score", 0.5)
            publisher = source_signals.get("publisher_name", "Unknown")
            cat = source_signals.get("category", "MEDIUM_CREDIBILITY")
            if cat == "HIGH_CREDIBILITY":
                detailed_bullets.append(f"Source Verification: Published by {publisher}, a verified authoritative news organization with established editorial standards.")
            elif cat in ["LOW_CREDIBILITY", "SATIRE"]:
                detailed_bullets.append(f"Source Verification: Published by {publisher}, flagged as {cat.lower()} or unreliable.")
                key_factors.append(f"Unreliable source ({publisher})")
            else:
                detailed_bullets.append(f"Source Verification: Independent or unindexed domain ({publisher}); reliability score: {cred_score:.1%}.")

        # 6. External Evidence & Fact-Checking
        if evidence_signals:
            overall_stance = evidence_signals.get("overall_stance", "UNKNOWN")
            supp_count = evidence_signals.get("supporting_count", 0)
            contra_count = evidence_signals.get("contradicting_count", 0)
            if overall_stance == "CORROBORATED":
                detailed_bullets.append(f"External Evidence: Corroborated by {supp_count} independent verified news report{'s' if supp_count > 1 else ''}.")
            elif overall_stance == "REFUTED":
                detailed_bullets.append(f"External Evidence: Directly contradicted or debunked by {contra_count} verified fact-checking source{'s' if contra_count > 1 else ''}.")
                key_factors.append("Contradicted by external evidence")
            else:
                detailed_bullets.append("External Evidence: Limited independent fact-checking records found; content treated with prudence.")

        # Synthesize Executive Summary
        if final_prediction == "FAKE":
            executive_summary = (
                f"Content classified as FAKE ({confidence:.1%} confidence). "
                f"Primary drivers: {'; '.join(key_factors) if key_factors else 'negative linguistic and forensic indicators'}."
            )
        elif final_prediction == "REAL":
            executive_summary = (
                f"Content verified as REAL ({confidence:.1%} confidence). "
                f"Corroborated by reputable reporting, natural media forensics, and coherent cross-modal consistency."
            )
        else:
            executive_summary = (
                f"Content flagged as SUSPICIOUS ({confidence:.1%} confidence). "
                f"Ambiguous signals or cross-modal discrepancies detected: {'; '.join(key_factors) if key_factors else 'requires manual inspection'}."
            )

        return {
            "prediction": final_prediction,
            "confidence": round(confidence, 4),
            "summary": executive_summary,
            "executive_summary": executive_summary,
            "key_factors": key_factors,
            "detailed_reasons": detailed_bullets
        }
