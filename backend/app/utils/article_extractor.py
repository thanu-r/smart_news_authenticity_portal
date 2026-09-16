import json
import re
import urllib.parse
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger("smartnews.article_extractor")

class ArticleExtractionError(ValueError):
    """Raised when an article cannot be extracted from HTML content."""
    pass

def extract_article_from_html(html: str, source_url: str) -> Dict[str, Any]:
    """
    Extract structured news article metadata and text body from raw HTML.
    Extracts:
      - title
      - text (main body)
      - domain
      - author (if available)
      - publication_date (if available)
      - lead_image_url (if available)
    """
    if not html or not html.strip():
        raise ArticleExtractionError("Webpage returned empty content.")

    soup = BeautifulSoup(html, "html.parser")
    parsed_url = urllib.parse.urlparse(source_url)
    domain = parsed_url.netloc.lower().replace("www.", "")

    # Remove script, style, nav, footer, header elements to prevent boilerplate pollution
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"]):
        # Keep application/ld+json for metadata extraction
        if tag.name == "script" and tag.get("type") == "application/ld+json":
            continue
        tag.decompose()

    title = ""
    text_body = ""
    author = None
    publication_date = None
    lead_image_url = None

    # 1. Try Schema.org JSON-LD extraction
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            items = data if isinstance(data, list) else [data]
            for item in items:
                # Handle @graph wrapper
                graph = item.get("@graph", [item]) if isinstance(item, dict) else []
                for node in graph:
                    if not isinstance(node, dict):
                        continue
                    node_type = str(node.get("@type", ""))
                    if any(t in node_type for t in ["NewsArticle", "Article", "BlogPosting", "Report"]):
                        if not title and node.get("headline"):
                            title = str(node["headline"]).strip()
                        if not text_body and node.get("articleBody"):
                            text_body = str(node["articleBody"]).strip()
                        if not author and node.get("author"):
                            auth_val = node["author"]
                            if isinstance(auth_val, list) and auth_val:
                                author = auth_val[0].get("name") if isinstance(auth_val[0], dict) else str(auth_val[0])
                            elif isinstance(auth_val, dict):
                                author = auth_val.get("name")
                            else:
                                author = str(auth_val)
                        if not publication_date and node.get("datePublished"):
                            publication_date = str(node["datePublished"])
                        if not lead_image_url and node.get("image"):
                            img_val = node["image"]
                            if isinstance(img_val, dict):
                                lead_image_url = img_val.get("url")
                            elif isinstance(img_val, list) and img_val:
                                lead_image_url = img_val[0] if isinstance(img_val[0], str) else img_val[0].get("url")
                            elif isinstance(img_val, str):
                                lead_image_url = img_val
        except Exception:
            pass

    # 2. Extract OpenGraph & Meta Tags as fallback/supplement
    if not title:
        og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()

    if not author:
        meta_author = (
            soup.find("meta", attrs={"name": "author"}) or
            soup.find("meta", property="article:author") or
            soup.find("meta", attrs={"name": "twitter:creator"})
        )
        if meta_author and meta_author.get("content"):
            author = meta_author["content"].strip()

    if not publication_date:
        meta_date = (
            soup.find("meta", property="article:published_time") or
            soup.find("meta", attrs={"name": "pubdate"}) or
            soup.find("meta", attrs={"name": "date"})
        )
        if meta_date and meta_date.get("content"):
            publication_date = meta_date["content"].strip()

    if not lead_image_url:
        og_img = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "twitter:image"})
        if og_img and og_img.get("content"):
            lead_image_url = urllib.parse.urljoin(source_url, og_img["content"].strip())

    # 3. Fallback for Title from HTML tags
    if not title:
        h1_tag = soup.find("h1")
        if h1_tag and h1_tag.get_text(strip=True):
            title = h1_tag.get_text(strip=True)
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        else:
            title = f"News report from {domain}"

    # 4. Extract Main Article Body Text if not already retrieved via JSON-LD
    if not text_body or len(text_body) < 100:
        # Search for high-signal content containers
        article_elem = soup.find("article") or soup.find("main") or soup.find(class_=re.compile(r"article[-_]body|story[-_]content|post[-_]content|entry[-_]content", re.I))
        target_container = article_elem if article_elem else soup

        # Extract meaningful paragraphs
        paragraphs = []
        for p in target_container.find_all("p"):
            p_text = p.get_text(separator=" ", strip=True)
            # Filter out tiny boilerplate, cookie warnings, copyright notices
            if len(p_text) >= 30 and not any(bp in p_text.lower() for bp in [
                "cookie policy", "terms of use", "privacy policy", "all rights reserved",
                "subscribe now", "advertisement", "sign up for our newsletter"
            ]):
                paragraphs.append(p_text)

        if paragraphs:
            text_body = "\n\n".join(paragraphs)

    # 5. Check if meaningful text was obtained
    if not text_body or len(text_body.strip()) < 50:
        # Attempt fallback to meta description if body is empty (e.g. paywall / JS splash)
        meta_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content") and len(meta_desc["content"].strip()) >= 50:
            text_body = meta_desc["content"].strip()
        else:
            raise ArticleExtractionError(
                f"Could not extract sufficient article text from {domain}. "
                "The webpage may require JavaScript execution, have an active paywall, or contain anti-scraping protections."
            )

    return {
        "title": title,
        "text": text_body,
        "domain": domain,
        "author": author,
        "publication_date": publication_date,
        "lead_image_url": lead_image_url,
        "source_url": source_url,
        "text_length": len(text_body)
    }
