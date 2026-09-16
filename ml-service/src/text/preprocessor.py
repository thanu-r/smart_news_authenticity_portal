import re
import unicodedata
from typing import Dict, Any, Optional

def clean_news_text(text: str) -> str:
    """
    Clean and normalize news text for model training and inference.
    Strips trivial wire identifiers (e.g. 'WASHINGTON (Reuters) - ') to prevent
    the model from memorizing publisher boilerplate instead of semantic content.
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Remove wire prefixes like "WASHINGTON (Reuters) - " or "LONDON (Reuters) -- "
    text = re.sub(r"^[A-Z\s,]{2,30}\s*\([A-Za-z\s,]+\)\s*[-–—]+\s*", "", text)
    text = re.sub(r"^([A-Z\s]{2,20}\s*[-–—]\s*)", "", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove HTML tags if present
    text = re.sub(r"<.*?>", "", text)

    # Normalize multiple whitespaces and newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, max_words_per_chunk: int = 250, overlap_words: int = 50) -> list[str]:
    """
    Split long news articles into overlapping chunks for chunk-level transformer inference.
    Preserves context across chunk boundaries without simple hard truncation.
    """
    words = text.split()
    if len(words) <= max_words_per_chunk:
        return [text] if text else []

    chunks = []
    step = max_words_per_chunk - overlap_words
    for i in range(0, len(words), step):
        chunk_words = words[i:i + max_words_per_chunk]
        if len(chunk_words) > 20:  # Ignore tiny trailing fragments
            chunks.append(" ".join(chunk_words))
        if i + max_words_per_chunk >= len(words):
            break
    return chunks

def extract_text_signals(text: str) -> Dict[str, Any]:
    """
    Extract linguistic and stylistic signals contributing to authenticity detection.
    Useful for structured explanation generation.
    """
    if not text:
        return {"word_count": 0, "exclamation_count": 0, "capital_word_ratio": 0.0, "sensational_score": 0.0}

    words = text.split()
    word_count = len(words)
    exclamation_count = text.count("!")
    question_count = text.count("?")

    # Count uppercase words (excluding common acronyms)
    capital_words = [w for w in words if w.isupper() and len(w) > 2]
    capital_ratio = len(capital_words) / max(word_count, 1)

    # Sensational / emotional clickbait indicators
    clickbait_cues = [
        "shocking", "bombshell", "unbelievable", "secret", "exposed",
        "conspiracy", "mainstream media won't tell you", "miracle", "furious",
        "devastating", "proof", "you won't believe", "destroy", "annihilate"
    ]
    lower_text = text.lower()
    cues_found = [cue for cue in clickbait_cues if cue in lower_text]
    sensational_score = min(1.0, (len(cues_found) * 0.25) + (capital_ratio * 0.5) + (min(exclamation_count, 4) * 0.1))

    return {
        "word_count": word_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "capital_word_ratio": round(capital_ratio, 4),
        "sensational_score": round(sensational_score, 4),
        "detected_cues": cues_found
    }
