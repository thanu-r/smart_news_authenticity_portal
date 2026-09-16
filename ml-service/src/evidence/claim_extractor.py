import re
from typing import List, Dict, Any, Optional

class ClaimExtractor:
    """
    Extracts central factual claims, named entities, and search queries from news texts.
    Optimizes query construction for evidence retrieval across RSS and fact-checking registries.
    """

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
        "which", "this", "that", "these", "those", "then", "just", "so", "than",
        "such", "both", "through", "about", "for", "is", "of", "while", "during",
        "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "don",
        "should", "now", "says", "said", "reported", "breaking", "shocking"
    }

    def extract_claims(self, text: str, title: Optional[str] = None) -> List[str]:
        """
        Extract primary factual assertion statements from title and body text.
        """
        claims = []
        if title and len(title.strip()) > 10:
            claims.append(title.strip())

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text.strip()) if text else []
        for s in sentences:
            s_clean = s.strip()
            # Select declarative factual assertions with sufficient length
            if 30 <= len(s_clean) <= 200:
                # Discard purely conversational / clickbait imperatives
                if not re.match(r"^(click|subscribe|watch|share|read more|comment below)", s_clean.lower()):
                    claims.append(s_clean)
            if len(claims) >= 4:
                break

        return claims[:3] if claims else ([title] if title else [text[:150]])

    def generate_search_queries(self, text: str, title: Optional[str] = None) -> List[str]:
        """
        Generate prioritized keyword search queries for fact-checking and news retrieval.
        """
        queries = []

        # 1. Headline query (cleaned of punctuation)
        if title:
            clean_title = re.sub(r"[^\w\s]", "", title).strip()
            # Remove filler words while preserving key entities
            words = [w for w in clean_title.split() if w.lower() not in self.STOP_WORDS]
            if len(words) >= 3:
                queries.append(" ".join(words[:8]))
            else:
                queries.append(clean_title)

        # 2. Quotation / named entity query
        quotes = re.findall(r'"([^"]{10,80})"', text)
        for q in quotes[:2]:
            queries.append(q.strip())

        # 3. Key entity & noun phrases from body text
        first_para = text.split("\n")[0] if text else ""
        cleaned_body = re.sub(r"[^\w\s]", "", first_para)
        body_words = [w for w in cleaned_body.split() if len(w) > 3 and w.lower() not in self.STOP_WORDS]
        if len(body_words) >= 4:
            queries.append(" ".join(body_words[:6]))

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for q in queries:
            q_norm = q.lower().strip()
            if q_norm and q_norm not in seen and len(q_norm) > 8:
                seen.add(q_norm)
                deduped.append(q.strip())

        return deduped[:3] if deduped else [title or text[:60]]
