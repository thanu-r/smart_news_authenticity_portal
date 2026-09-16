import re
from urllib.parse import urlparse
from typing import Dict, Any, Optional, List

# Curated registry of verified, high-credibility news organizations and wire services
HIGH_CREDIBILITY_REGISTRY = {
    "reuters.com": {"name": "Reuters", "tier": 1, "bias": "least-biased", "credibility": 0.98},
    "apnews.com": {"name": "Associated Press", "tier": 1, "bias": "least-biased", "credibility": 0.98},
    "bbc.com": {"name": "BBC News", "tier": 1, "bias": "left-center", "credibility": 0.94},
    "bbc.co.uk": {"name": "BBC News UK", "tier": 1, "bias": "left-center", "credibility": 0.94},
    "npr.org": {"name": "NPR", "tier": 1, "bias": "least-biased", "credibility": 0.92},
    "pbs.org": {"name": "PBS NewsHour", "tier": 1, "bias": "least-biased", "credibility": 0.93},
    "bloomberg.com": {"name": "Bloomberg News", "tier": 1, "bias": "least-biased", "credibility": 0.93},
    "wsj.com": {"name": "Wall Street Journal", "tier": 1, "bias": "right-center", "credibility": 0.91},
    "nytimes.com": {"name": "The New York Times", "tier": 2, "bias": "left-center", "credibility": 0.88},
    "washingtonpost.com": {"name": "The Washington Post", "tier": 2, "bias": "left-center", "credibility": 0.87},
    "theguardian.com": {"name": "The Guardian", "tier": 2, "bias": "left-center", "credibility": 0.86},
    "economist.com": {"name": "The Economist", "tier": 1, "bias": "least-biased", "credibility": 0.92},
    "nature.com": {"name": "Nature Publishing", "tier": 1, "bias": "scientific", "credibility": 0.99},
    "science.org": {"name": "Science / AAAS", "tier": 1, "bias": "scientific", "credibility": 0.99},
    "afp.com": {"name": "Agence France-Presse", "tier": 1, "bias": "least-biased", "credibility": 0.95},
    "dw.com": {"name": "Deutsche Welle", "tier": 1, "bias": "least-biased", "credibility": 0.92},
    "aljazeera.com": {"name": "Al Jazeera English", "tier": 2, "bias": "left-center", "credibility": 0.82},
    "politico.com": {"name": "Politico", "tier": 2, "bias": "least-biased", "credibility": 0.85},
    "thehill.com": {"name": "The Hill", "tier": 2, "bias": "least-biased", "credibility": 0.85},
    "factcheck.org": {"name": "FactCheck.org", "tier": 1, "bias": "fact-checker", "credibility": 0.97},
    "politifact.com": {"name": "PolitiFact", "tier": 1, "bias": "fact-checker", "credibility": 0.96},
    "snopes.com": {"name": "Snopes", "tier": 1, "bias": "fact-checker", "credibility": 0.95}
}

# Known deceptive, satirical, or conspiracy registries
LOW_CREDIBILITY_REGISTRY = {
    "theonion.com": {"name": "The Onion", "type": "satire", "credibility": 0.10},
    "babylonbee.com": {"name": "The Babylon Bee", "type": "satire", "credibility": 0.15},
    "infowars.com": {"name": "InfoWars", "type": "conspiracy/fake", "credibility": 0.05},
    "naturalnews.com": {"name": "Natural News", "type": "health-misinformation", "credibility": 0.08},
    "worldnewsdailyreport.com": {"name": "World News Daily Report", "type": "fake/satire", "credibility": 0.05},
    "beforeitsnews.com": {"name": "Before It's News", "type": "unvetted-blog", "credibility": 0.15},
    "breitbart.com": {"name": "Breitbart News", "type": "hyperpartisan", "credibility": 0.40},
    "dailykos.com": {"name": "Daily Kos", "type": "hyperpartisan", "credibility": 0.45}
}

SUSPICIOUS_TLDS = {".xyz", ".top", ".click", ".buzz", ".work", ".link", ".gq", ".cf", ".tk", ".ml", ".ga"}
HIGH_TRUST_TLDS = {".gov", ".edu", ".mil"}

class SourceCredibilityAnalyzer:
    """
    Evaluates news source trustworthiness using domain reputation, TLD security,
    verified registry cross-referencing, and attribution integrity.
    """

    @staticmethod
    def extract_clean_domain(url_or_domain: str) -> str:
        """Extract canonical registered domain from arbitrary URL or text."""
        if not url_or_domain:
            return ""
        raw = url_or_domain.strip().lower()
        if not (raw.startswith("http://") or raw.startswith("https://")):
            raw = "https://" + raw

        try:
            parsed = urlparse(raw)
            hostname = parsed.hostname or ""
            # Strip leading 'www.'
            if hostname.startswith("www."):
                hostname = hostname[4:]
            return hostname
        except Exception:
            return ""

    def evaluate_source(
        self,
        url_or_domain: Optional[str] = None,
        author: Optional[str] = None,
        publisher: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze credibility signals for a given news source.
        Returns a composite credibility score [0.0, 1.0], category, and transparent breakdown.
        """
        domain = self.extract_clean_domain(url_or_domain or publisher or "")
        is_https = (url_or_domain or "").lower().startswith("https://") if url_or_domain else True

        breakdown = {
            "domain": domain,
            "is_https": is_https,
            "registry_match": None,
            "tld_trust": "standard",
            "author_present": bool(author and author.lower() not in ["unknown", "admin", "staff", ""]),
            "satire_detected": False,
            "indicators": []
        }

        # 1. Check known high-credibility news registry
        if domain in HIGH_CREDIBILITY_REGISTRY:
            info = HIGH_CREDIBILITY_REGISTRY[domain]
            score = info["credibility"]
            breakdown["registry_match"] = info
            breakdown["indicators"].append(f"Verified authoritative news source ({info['name']})")
            if breakdown["author_present"]:
                score = min(0.99, score + 0.02)
                breakdown["indicators"].append(f"Attributed to journalist: {author}")
            
            category = "HIGH_CREDIBILITY"
            return {
                "credibility_score": round(score, 4),
                "category": category,
                "publisher_name": info["name"],
                "breakdown": breakdown,
                "summary": f"Authoritative source with established editorial standards ({info['name']})."
            }

        # 2. Check known low-credibility or satire registry
        if domain in LOW_CREDIBILITY_REGISTRY:
            info = LOW_CREDIBILITY_REGISTRY[domain]
            score = info["credibility"]
            breakdown["registry_match"] = info
            breakdown["satire_detected"] = (info["type"] == "satire")
            breakdown["indicators"].append(f"Source flagged in low-credibility registry: {info['type']}")
            category = "LOW_CREDIBILITY" if not breakdown["satire_detected"] else "SATIRE"
            return {
                "credibility_score": round(score, 4),
                "category": category,
                "publisher_name": info["name"],
                "breakdown": breakdown,
                "summary": f"Known {info['type']} outlet ({info['name']}). Claims require independent verification."
            }

        # 3. Dynamic heuristic evaluation for unindexed domains
        score = 0.55  # Default neutral baseline for unknown domains
        
        # TLD check
        tld = "." + domain.split(".")[-1] if "." in domain else ""
        if tld in HIGH_TRUST_TLDS:
            score += 0.30
            breakdown["tld_trust"] = "high"
            breakdown["indicators"].append(f"Official government or educational top-level domain ({tld})")
        elif tld in SUSPICIOUS_TLDS:
            score -= 0.30
            breakdown["tld_trust"] = "suspicious"
            breakdown["indicators"].append(f"High-risk low-cost top-level domain frequently associated with spam ({tld})")

        # Deceptive lookalike / hyphenated domain checks (e.g. 'cnn-news-breaking.xyz')
        if domain.count("-") >= 2:
            score -= 0.15
            breakdown["indicators"].append("Multiple hyphens in domain name (common in deceptive spoof sites)")

        if not is_https:
            score -= 0.10
            breakdown["indicators"].append("Missing HTTPS encryption")

        if breakdown["author_present"]:
            score += 0.08
            breakdown["indicators"].append(f"Author byline detected: {author}")
        else:
            breakdown["indicators"].append("No verified individual journalist attribution found")

        # Bound score [0.05, 0.95]
        score = max(0.05, min(0.95, score))

        if score >= 0.75:
            category = "HIGH_CREDIBILITY"
            summary = "Domain exhibits standard institutional credibility signals."
        elif score >= 0.45:
            category = "MEDIUM_CREDIBILITY"
            summary = "Unregistered or independent web domain with standard web infrastructure."
        else:
            category = "LOW_CREDIBILITY"
            summary = "Domain exhibits structural or domain-level risk factors."

        return {
            "credibility_score": round(score, 4),
            "category": category,
            "publisher_name": domain or "Independent Source",
            "breakdown": breakdown,
            "summary": summary
        }
