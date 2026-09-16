import re
from typing import Dict, Any, List, Optional
import numpy as np

CONTRADICTION_CUES = {
    "debunked", "false", "hoax", "fabricated", "unfounded", "denies", "denied",
    "disputed", "refutes", "refuted", "misleading", "fact check", "no evidence",
    "incorrect", "inaccurate", "untrue", "contrary to", "disproven", "fake",
    "falsely claimed", "never happened", "contradicts", "debunking"
}

SUPPORT_CUES = {
    "confirms", "confirmed", "verified", "authenticates", "reported", "corroborates",
    "corroborated", "official statement", "agreed", "proves", "documents show",
    "announced", "according to officials", "evidence shows", "substantiated"
}

class StanceDetector:
    """
    Determines whether an external evidence snippet SUPPORTS, CONTRADICTS,
    or is NEUTRAL towards a given news claim or headline.
    """

    def detect_stance(self, claim: str, evidence_snippet: str) -> Dict[str, Any]:
        """
        Analyze claim vs evidence snippet and output stance, confidence, and linguistic rationale.
        """
        claim_clean = claim.lower().strip()
        snippet_clean = evidence_snippet.lower().strip()

        # Token overlap for relevance estimation
        claim_words = set(re.findall(r"\w{3,}", claim_clean))
        snippet_words = set(re.findall(r"\w{3,}", snippet_clean))

        if not claim_words or not snippet_words:
            return {
                "stance": "NEUTRAL",
                "confidence": 0.5,
                "relevance": 0.0,
                "detected_cues": [],
                "rationale": "Insufficient lexical overlap between claim and evidence snippet."
            }

        overlap = claim_words.intersection(snippet_words)
        relevance = len(overlap) / max(len(claim_words), 1)

        # Look for refutation cues
        contra_found = [c for c in CONTRADICTION_CUES if c in snippet_clean]
        support_found = [s for s in SUPPORT_CUES if s in snippet_clean]

        # Stance logic
        if contra_found:
            stance = "CONTRADICTS"
            confidence = min(0.95, round(0.65 + (len(contra_found) * 0.1) + (min(relevance, 0.5) * 0.4), 4))
            rationale = f"Evidence contains explicit refutation/debunking markers: {', '.join(contra_found[:3])}."
            cues = contra_found
        elif support_found and relevance >= 0.25:
            stance = "SUPPORTS"
            confidence = min(0.92, round(0.60 + (len(support_found) * 0.08) + (min(relevance, 0.5) * 0.4), 4))
            rationale = f"Evidence corroborates claim with affirmative reporting cues: {', '.join(support_found[:3])}."
            cues = support_found
        elif relevance >= 0.40:
            # High topical alignment without refutation cues
            stance = "SUPPORTS"
            confidence = min(0.85, round(0.50 + relevance * 0.45, 4))
            rationale = f"High topical and entity alignment ({len(overlap)} matching terms) without contradictory markers."
            cues = list(overlap)[:5]
        else:
            stance = "NEUTRAL"
            confidence = 0.60
            rationale = f"Snippet provides contextual background but lacks decisive confirmatory or refuting stance."
            cues = []

        return {
            "stance": stance,
            "confidence": confidence,
            "relevance": round(min(1.0, relevance), 4),
            "detected_cues": cues,
            "rationale": rationale
        }
