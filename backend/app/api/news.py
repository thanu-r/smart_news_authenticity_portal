import re
import uuid
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status
from bson import ObjectId
import feedparser

from backend.app.database.connection import MongoDBManager
from backend.app.schemas.verification_schemas import LiveNewsItem
from backend.app.api.verify import get_fusion_engine
from backend.app.utils.safe_fetcher import safe_fetch_url, SafeFetchError, SSRFSecurityError
from backend.app.utils.article_extractor import extract_article_from_html, ArticleExtractionError

logger = logging.getLogger("smartnews.news")

router = APIRouter(prefix="/api/news", tags=["Live News"])

CURATED_FEEDS = [
    {"source": "NYT World", "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "category": "World"},
    {"source": "BBC News", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "category": "World"},
    {"source": "The Guardian", "url": "https://www.theguardian.com/world/rss", "category": "World"},
    {"source": "NPR", "url": "https://feeds.npr.org/1001/rss.xml", "category": "General"},
    {"source": "AP News", "url": "https://feedx.net/rss/ap.xml", "category": "General"},
    {"source": "TechCrunch", "url": "https://techcrunch.com/feed/", "category": "Science & Tech"},
]

def generate_headline_hash(title: str, url: str = "") -> str:
    """Deterministic hash for deduplicating incoming live news articles."""
    norm = (title or "").lower().strip()
    return hashlib.sha256(norm.encode('utf-8')).hexdigest()

def extract_entry_image(entry: Any) -> Optional[str]:
    """Extract lead/thumbnail image from feed entry across multiple RSS formats."""
    # 1. media_content
    media_content = entry.get("media_content")
    if isinstance(media_content, list) and media_content:
        for m in media_content:
            if isinstance(m, dict) and m.get("url"):
                return m["url"]
    elif isinstance(media_content, dict) and media_content.get("url"):
        return media_content["url"]

    # 2. media_thumbnail
    media_thumb = entry.get("media_thumbnail")
    if isinstance(media_thumb, list) and media_thumb:
        for t in media_thumb:
            if isinstance(t, dict) and t.get("url"):
                return t["url"]
    elif isinstance(media_thumb, dict) and media_thumb.get("url"):
        return media_thumb["url"]

    # 3. enclosures
    enclosures = entry.get("enclosures")
    if isinstance(enclosures, list):
        for enc in enclosures:
            if isinstance(enc, dict) and ("image" in enc.get("type", "") or enc.get("url", "").endswith((".jpg", ".png", ".webp"))):
                return enc.get("url")

    # 4. links with image type
    links = entry.get("links", [])
    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict) and "image" in link.get("type", "") and link.get("href"):
                return link["href"]

    # 5. Extract <img> from summary / description HTML
    summary_html = entry.get("summary", "") or entry.get("description", "")
    if "<img" in summary_html:
        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', summary_html, re.IGNORECASE)
        if img_match:
            return img_match.group(1)

    return None

def clean_summary_text(summary_html: str) -> str:
    """Strip HTML tags from summary string."""
    clean = re.sub(r'<[^>]+>', ' ', summary_html or '')
    return re.sub(r'\s+', ' ', clean).strip()

@router.get("/live", response_model=List[LiveNewsItem])
async def get_live_news(
    limit: int = Query(20, ge=1, le=50),
    verified_only: bool = False
):
    """Retrieve indexed live news articles from database or RSS feeds."""
    articles = []

    # 1. Check MongoDB first
    if MongoDBManager.db is not None:
        try:
            query = {"verified": True} if verified_only else {}
            cursor = MongoDBManager.db.live_news.find(query).sort("published_at", -1).limit(limit)
            async for doc in cursor:
                verif_res = doc.get("verification_result") or {}
                evidence_items = verif_res.get("evidence_retrieval", {}).get("evidence_list") if isinstance(verif_res.get("evidence_retrieval"), dict) else None
                articles.append(LiveNewsItem(
                    id=str(doc.get("_id", doc.get("id", ""))),
                    title=doc.get("title", ""),
                    summary=doc.get("summary", ""),
                    url=doc.get("url", ""),
                    source=doc.get("source", "Live News"),
                    category=doc.get("category", "General"),
                    image_url=doc.get("image_url"),
                    published_at=str(doc.get("published_at", "Recent")),
                    verified_at=str(doc.get("verified_at")) if doc.get("verified_at") else None,
                    verified=doc.get("verified", False),
                    verification_status=verif_res.get("prediction") or ("REAL" if doc.get("verified") else None),
                    confidence=verif_res.get("calibrated_confidence") or verif_res.get("confidence"),
                    verification_result=verif_res if doc.get("verified") else None,
                    evidence=evidence_items
                ))
        except Exception as e:
            logger.warning(f"Error reading live news from DB: {e}")

    # 2. If database empty, dynamically fetch recent headlines from trusted feeds
    if not articles:
        for feed_info in CURATED_FEEDS:
            try:
                parsed = feedparser.parse(feed_info["url"])
                for entry in parsed.entries[:5]:
                    item_id = str(uuid.uuid4())[:8]
                    img_url = extract_entry_image(entry)
                    clean_sum = clean_summary_text(entry.get("summary", ""))[:250]
                    articles.append(LiveNewsItem(
                        id=item_id,
                        title=entry.get("title", "Headline"),
                        summary=clean_sum,
                        url=entry.get("link", ""),
                        source=feed_info["source"],
                        category=feed_info["category"],
                        image_url=img_url,
                        published_at=entry.get("published", "Recent"),
                        verified_at=None,
                        verified=False,
                        verification_status=None,
                        confidence=None,
                        verification_result=None,
                        evidence=None
                    ))
            except Exception as e:
                logger.warning(f"Error fetching feed {feed_info['source']}: {e}")
                continue

    return articles[:limit]

@router.get("/latest")
async def get_latest_news(
    limit: int = Query(20, ge=1, le=50),
    category: Optional[str] = Query(None)
):
    """Retrieve indexed live news with category filtering for frontend portal."""
    articles = []
    
    if MongoDBManager.db is not None:
        try:
            query = {}
            if category and category.lower() != 'all':
                query["category"] = {"$regex": f"^{category}", "$options": "i"}
            cursor = MongoDBManager.db.live_news.find(query).sort("published_at", -1).limit(limit)
            async for doc in cursor:
                verif_res = doc.get("verification_result") or {}
                evidence_items = verif_res.get("evidence_retrieval", {}).get("evidence_list") if isinstance(verif_res.get("evidence_retrieval"), dict) else None
                articles.append({
                    "id": str(doc.get("_id", doc.get("id", ""))),
                    "title": doc.get("title", ""),
                    "summary": doc.get("summary", ""),
                    "url": doc.get("url", ""),
                    "source": doc.get("source", "Live News"),
                    "category": doc.get("category", "General"),
                    "image_url": doc.get("image_url"),
                    "published_at": str(doc.get("published_at", "Recent")),
                    "verified_at": str(doc.get("verified_at")) if doc.get("verified_at") else None,
                    "verified": doc.get("verified", False),
                    "verification_status": verif_res.get("prediction") or ("REAL" if doc.get("verified") else None),
                    "confidence": verif_res.get("calibrated_confidence") or verif_res.get("confidence"),
                    "verification_result": verif_res if doc.get("verified") else None,
                    "evidence": evidence_items
                })
        except Exception as e:
            logger.warning(f"Error fetching latest news: {e}")

    if not articles:
        # Provide populated curated default items dynamically from feeds
        live_list = await get_live_news(limit=limit)
        for item in live_list:
            articles.append({
                "id": item.id,
                "title": item.title,
                "summary": item.summary,
                "url": item.url,
                "source": item.source,
                "category": item.category or "World",
                "image_url": item.image_url,
                "published_at": item.published_at,
                "verified_at": item.verified_at,
                "verified": item.verified,
                "verification_status": item.verification_status,
                "confidence": item.confidence,
                "verification_result": item.verification_result,
                "evidence": item.evidence
            })

    return {"articles": articles}

@router.get("/search")
async def search_live_news(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=50)
):
    """Search ingested articles across title and summary text."""
    articles = []
    if MongoDBManager.db is not None:
        try:
            regex_query = {"$regex": q, "$options": "i"}
            cursor = MongoDBManager.db.live_news.find({
                "$or": [
                    {"title": regex_query},
                    {"summary": regex_query},
                    {"source": regex_query}
                ]
            }).sort("published_at", -1).limit(limit)

            async for doc in cursor:
                verif_res = doc.get("verification_result") or {}
                evidence_items = verif_res.get("evidence_retrieval", {}).get("evidence_list") if isinstance(verif_res.get("evidence_retrieval"), dict) else None
                articles.append({
                    "id": str(doc.get("_id", doc.get("id", ""))),
                    "title": doc.get("title", ""),
                    "summary": doc.get("summary", ""),
                    "url": doc.get("url", ""),
                    "source": doc.get("source", "Live News"),
                    "category": doc.get("category", "General"),
                    "image_url": doc.get("image_url"),
                    "published_at": str(doc.get("published_at", "Recent")),
                    "verified_at": str(doc.get("verified_at")) if doc.get("verified_at") else None,
                    "verified": doc.get("verified", False),
                    "verification_status": verif_res.get("prediction"),
                    "confidence": verif_res.get("calibrated_confidence") or verif_res.get("confidence"),
                    "verification_result": verif_res,
                    "evidence": evidence_items
                })
        except Exception as e:
            logger.warning(f"Error searching news: {e}")

    return {"articles": articles, "query": q, "count": len(articles)}

@router.post("/fetch")
@router.post("/refresh")
async def trigger_news_fetch():
    """Trigger background ingestion and deduplication from curated RSS feeds."""
    ingested_count = 0
    now = datetime.now(timezone.utc)

    for feed_info in CURATED_FEEDS:
        try:
            parsed = feedparser.parse(feed_info["url"])
            for entry in parsed.entries[:12]:
                link = entry.get("link", "")
                title = entry.get("title", "")
                if not link or not title:
                    continue

                headline_hash = generate_headline_hash(title, link)
                img_url = extract_entry_image(entry)
                clean_sum = clean_summary_text(entry.get("summary", ""))[:350]
                doc = {
                    "title": title,
                    "summary": clean_sum,
                    "url": link,
                    "source": feed_info["source"],
                    "category": feed_info.get("category", "General"),
                    "image_url": img_url,
                    "headline_hash": headline_hash,
                    "published_at": entry.get("published", str(now)),
                    "verified": False,
                    "verified_at": None,
                    "verification_result": None
                }

                if MongoDBManager.db is not None:
                    try:
                        # Deduplicate by URL or headline hash
                        res = await MongoDBManager.db.live_news.update_one(
                            {"$or": [{"url": link}, {"headline_hash": headline_hash}]},
                            {"$setOnInsert": doc},
                            upsert=True
                        )
                        if res.upserted_id:
                            ingested_count += 1
                    except Exception as e:
                        logger.warning(f"Error upserting live news doc: {e}")
        except Exception as e:
            logger.warning(f"Error parsing feed {feed_info['source']}: {e}")
            continue

    return {
        "status": "success",
        "newly_ingested_articles": ingested_count,
        "timestamp": now
    }

@router.post("/verify-one/{article_id}", response_model=LiveNewsItem)
async def verify_single_live_news_article(article_id: str):
    """Run full verification pipeline on a specific live news article."""
    if MongoDBManager.db is None:
        raise HTTPException(status_code=503, detail="Database service not available")

    # Find article by ObjectId or string id
    query_conditions = [{"id": article_id}]
    if ObjectId.is_valid(article_id):
        query_conditions.append({"_id": ObjectId(article_id)})
    query_conditions.append({"_id": article_id})

    doc = await MongoDBManager.db.live_news.find_one({"$or": query_conditions})
    if not doc:
        raise HTTPException(status_code=404, detail="Article not found")

    title = doc.get("title", "")
    summary = doc.get("summary", "")
    url = doc.get("url", "")
    img_url = doc.get("image_url")

    # Attempt to fetch full article body from URL for rich textual verification
    content_to_verify = summary
    if url:
        try:
            html, final_url, status_code = safe_fetch_url(url, timeout=5.0)
            if html and status_code == 200:
                extracted = extract_article_from_html(html, final_url)
                if extracted.get("text") and len(extracted["text"]) > len(summary):
                    content_to_verify = extracted["text"]
                    if not img_url and extracted.get("lead_image_url"):
                        img_url = extracted["lead_image_url"]
        except Exception as e:
            logger.info(f"Could not fetch full article text for live news {url}, using RSS summary: {e}")

    engine = get_fusion_engine()
    verif_res = engine.verify_content(
        text=content_to_verify,
        title=title,
        url=url,
        retrieve_evidence=True
    )

    verif_timestamp = datetime.now(timezone.utc)
    evidence_items = verif_res.get("evidence_retrieval", {}).get("evidence_list") if isinstance(verif_res.get("evidence_retrieval"), dict) else None

    # Update in live_news collection
    await MongoDBManager.db.live_news.update_one(
        {"_id": doc["_id"]},
        {
            "$set": {
                "verified": True,
                "verified_at": verif_timestamp,
                "image_url": img_url,
                "verification_result": verif_res
            }
        }
    )

    # Also persist to verifications history collection so it shows in verification history
    try:
        verif_record = {
            "_id": str(uuid.uuid4()),
            "title": title,
            "text_snippet": content_to_verify[:300],
            "source_url": url,
            "user_id": None,
            "created_at": verif_timestamp,
            **verif_res
        }
        await MongoDBManager.db.verifications.insert_one(verif_record)
    except Exception as e:
        logger.warning(f"Could not persist live news to verifications collection: {e}")

    return LiveNewsItem(
        id=str(doc["_id"]),
        title=title,
        summary=summary,
        url=url,
        source=doc.get("source", "Live News"),
        category=doc.get("category", "General"),
        image_url=img_url,
        published_at=str(doc.get("published_at", "Recent")),
        verified_at=str(verif_timestamp),
        verified=True,
        verification_status=verif_res.get("prediction"),
        confidence=verif_res.get("calibrated_confidence") or verif_res.get("confidence"),
        verification_result=verif_res,
        evidence=evidence_items
    )
