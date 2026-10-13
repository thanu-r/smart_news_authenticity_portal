import os
import sys
import re
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import feedparser
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent) if "__file__" in locals() else ".")
from src.evidence.claim_extractor import ClaimExtractor
from src.evidence.stance_detector import StanceDetector
from src.source.credibility_analyzer import SourceCredibilityAnalyzer

# Curated high-credibility news and fact-checking RSS feeds for live cross-referencing
VERIFIED_RSS_FEEDS = [
    {"name": "PolitiFact", "url": "https://www.politifact.com/rss/factchecks/", "credibility": 0.98, "is_fact_checker": True},
    {"name": "FactCheck.org", "url": "https://www.factcheck.org/feed/", "credibility": 0.98, "is_fact_checker": True},
    {"name": "NYT World", "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "credibility": 0.96, "is_fact_checker": False},
    {"name": "BBC News", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "credibility": 0.95, "is_fact_checker": False},
    {"name": "The Guardian", "url": "https://www.theguardian.com/world/rss", "credibility": 0.94, "is_fact_checker": False},
    {"name": "NPR News", "url": "https://feeds.npr.org/1001/rss.xml", "credibility": 0.93, "is_fact_checker": False},
    {"name": "AP News", "url": "https://feedx.net/rss/ap.xml", "credibility": 0.96, "is_fact_checker": False},
    {"name": "PBS NewsHour", "url": "https://www.pbs.org/newshour/feeds/rss/headlines", "credibility": 0.94, "is_fact_checker": False}
]

class EvidenceRetriever:
    """
    Retrieves and ranks external corroborating or refuting evidence
    from verified news registries, RSS feeds, and web search.
    """
    def __init__(self):
        self.claim_extractor = ClaimExtractor()
        self.stance_detector = StanceDetector()
        self.credibility_analyzer = SourceCredibilityAnalyzer()

    def retrieve_evidence(
        self,
        text: str,
        title: Optional[str] = None,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Execute multi-source evidence retrieval for a given news story or claim.
        Returns ranked evidence citations, overall evidence stance, and corroboration score.
        """
        queries = self.claim_extractor.generate_search_queries(text, title=title)
        primary_claim = title or (text[:120] + "..." if len(text) > 120 else text)

        all_candidates = []

        # 1. Search high-trust RSS feeds
        for feed_info in VERIFIED_RSS_FEEDS:
            try:
                parsed_feed = feedparser.parse(feed_info["url"])
                for entry in parsed_feed.entries[:8]:
                    entry_title = entry.get("title", "")
                    entry_summary = entry.get("summary", "")
                    entry_link = entry.get("link", "")
                    combined_entry = f"{entry_title}. {entry_summary}"

                    # Multi-query topical keyword matching
                    all_query_words = set()
                    for q in queries:
                        all_query_words.update([w.lower() for w in q.split() if len(w) > 3])
                    claim_words = set([w.lower() for w in re.findall(r"\w{4,}", primary_claim)])
                    target_words = all_query_words.union(claim_words)

                    entry_words = set(re.findall(r"\w{4,}", combined_entry.lower()))
                    match_count = len(target_words.intersection(entry_words))

                    if match_count >= 2 or (feed_info.get("is_fact_checker") and match_count >= 1):
                        all_candidates.append({
                            "title": entry_title,
                            "snippet": entry_summary[:300],
                            "url": entry_link,
                            "publisher": feed_info["name"],
                            "published_at": entry.get("published", "Recent"),
                            "source_credibility": feed_info["credibility"],
                            "is_fact_checker": feed_info.get("is_fact_checker", False)
                        })
            except Exception:
                continue

        # 2. Fallback to DuckDuckGo HTML / instant search if candidates are low
        if len(all_candidates) < 2 and queries:
            try:
                query_str = queries[0]
                ddg_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query_str + ' news')}"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                resp = requests.get(ddg_url, headers=headers, timeout=4)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for result in soup.find_all("div", class_="result")[:4]:
                        title_tag = result.find("a", class_="result__a")
                        snippet_tag = result.find("a", class_="result__snippet")
                        if title_tag:
                            res_title = title_tag.get_text(strip=True)
                            res_url = title_tag.get("href", "")
                            res_snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                            
                            # Clean DuckDuckGo redirect url if present
                            if "uddg=" in res_url:
                                res_url = urllib.parse.unquote(res_url.split("uddg=")[-1].split("&")[0])

                            cred = self.credibility_analyzer.evaluate_source(res_url)
                            all_candidates.append({
                                "title": res_title,
                                "snippet": res_snippet,
                                "url": res_url,
                                "publisher": cred["publisher_name"],
                                "published_at": "Recent",
                                "source_credibility": cred["credibility_score"]
                            })
            except Exception:
                pass

        # 3. Analyze stance and score each candidate
        scored_evidence = []
        for cand in all_candidates:
            snippet_text = f"{cand['title']}. {cand['snippet']}"
            stance_res = self.stance_detector.detect_stance(primary_claim, snippet_text)
            
            # Weighted evidence score combining stance confidence, topical relevance, and publisher credibility
            evidence_weight = (
                stance_res["confidence"] * 0.4 +
                stance_res["relevance"] * 0.3 +
                cand["source_credibility"] * 0.3
            )

            scored_evidence.append({
                "title": cand["title"],
                "snippet": cand["snippet"],
                "url": cand["url"],
                "publisher": cand["publisher"],
                "published_at": cand["published_at"],
                "stance": stance_res["stance"],
                "stance_confidence": stance_res["confidence"],
                "relevance": stance_res["relevance"],
                "rationale": stance_res["rationale"],
                "evidence_score": round(evidence_weight, 4)
            })

        # Rank evidence by score
        scored_evidence.sort(key=lambda x: x["evidence_score"], reverse=True)
        top_evidence = scored_evidence[:max_results]

        # Overall synthesis
        if not top_evidence:
            return {
                "corroboration_status": "INSUFFICIENT_EVIDENCE",
                "evidence_count": 0,
                "evidence_list": [],
                "supporting_count": 0,
                "contradicting_count": 0,
                "neutral_count": 0,
                "overall_stance": "UNKNOWN",
                "corroboration_score": 0.50,
                "summary": "Insufficient external evidence found across verified news registries and fact-checking archives."
            }

        supp_count = sum(1 for e in top_evidence if e["stance"] == "SUPPORTS")
        contra_count = sum(1 for e in top_evidence if e["stance"] == "CONTRADICTS")
        neut_count = sum(1 for e in top_evidence if e["stance"] == "NEUTRAL")

        if contra_count > supp_count:
            overall_stance = "REFUTED"
            corroboration_score = round(max(0.1, 0.5 - (contra_count * 0.15)), 4)
            summary = f"External fact-checking evidence contradicts claim ({contra_count} refuting source{'s' if contra_count > 1 else ''})."
        elif supp_count > contra_count:
            overall_stance = "CORROBORATED"
            corroboration_score = round(min(0.95, 0.5 + (supp_count * 0.15)), 4)
            summary = f"Claim corroborated by {supp_count} independent verified news source{'s' if supp_count > 1 else ''}."
        else:
            overall_stance = "MIXED_OR_AMBIGUOUS"
            corroboration_score = 0.50
            summary = "External coverage presents mixed or neutral reporting regarding this assertion."

        return {
            "corroboration_status": "EVIDENCE_RETRIEVED",
            "evidence_count": len(top_evidence),
            "evidence_list": top_evidence,
            "supporting_count": supp_count,
            "contradicting_count": contra_count,
            "neutral_count": neut_count,
            "overall_stance": overall_stance,
            "corroboration_score": corroboration_score,
            "search_queries_used": queries,
            "summary": summary
        }
