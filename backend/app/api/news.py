import uuid
import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status
import feedparser

from backend.app.database.connection import MongoDBManager
from backend.app.schemas.verification_schemas import LiveNewsItem
from backend.app.api.verify import get_fusion_engine

router = APIRouter(prefix="/api/news", tags=["Live News"])

CURATED_FEEDS = [
    {"source": "Reuters", "url": "https://feeds.reuters.com/reuters/worldNews", "category": "World"},
    {"source": "BBC News", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "category": "World"},
    {"source": "NPR", "url": "https://feeds.npr.org/1001/rss.xml", "category": "General"},
    {"source": "AP News", "url": "https://feedx.net/rss/ap.xml", "category": "General"},
    {"source": "TechCrunch", "url": "https://techcrunch.com/feed/", "category": "Science & Tech"},
]

def generate_headline_hash(title: str, url: str = "") -> str:
    """Deterministic hash for deduplicating incoming live news articles."""
    norm = (title or "").lower().strip()
    return hashlib.sha256(norm.encode('utf-8')).hexdigest()

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
                articles.append(LiveNewsItem(
                    id=str(doc.get("_id", doc.get("id", ""))),
                    title=doc.get("title", ""),
                    summary=doc.get("summary", ""),
                    url=doc.get("url", ""),
                    source=doc.get("source", "Live News"),
                    published_at=str(doc.get("published_at", "Recent")),
                    verified=doc.get("verified", False),
                    verification_result=doc.get("verification_result")
                ))
        except Exception:
            pass

    # 2. If database empty, dynamically fetch recent headlines from trusted feeds
    if not articles:
        for feed_info in CURATED_FEEDS:
            try:
                parsed = feedparser.parse(feed_info["url"])
                for entry in parsed.entries[:5]:
                    item_id = str(uuid.uuid4())[:8]
                    articles.append(LiveNewsItem(
                        id=item_id,
                        title=entry.get("title", "Headline"),
                        summary=entry.get("summary", "")[:250],
                        url=entry.get("link", ""),
                        source=feed_info["source"],
                        published_at=entry.get("published", "Recent"),
                        verified=False,
                        verification_result=None
                    ))
            except Exception:
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
                articles.append({
                    "id": str(doc.get("_id", doc.get("id", ""))),
                    "title": doc.get("title", ""),
                    "summary": doc.get("summary", ""),
                    "url": doc.get("url", ""),
                    "source": doc.get("source", "Live News"),
                    "category": doc.get("category", "General"),
                    "published_at": str(doc.get("published_at", "Recent")),
                    "verification_status": verif_res.get("prediction") or ("REAL" if doc.get("verified") else None),
                    "confidence": verif_res.get("calibrated_confidence") or verif_res.get("confidence"),
                })
        except Exception:
            pass

    if not articles:
        # Provide populated curated default items
        live_list = await get_live_news(limit=limit)
        for item in live_list:
            articles.append({
                "id": item.id,
                "title": item.title,
                "summary": item.summary,
                "url": item.url,
                "source": item.source,
                "category": "World",
                "published_at": item.published_at,
                "verification_status": None,
                "confidence": None,
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
                articles.append({
                    "id": str(doc.get("_id", doc.get("id", ""))),
                    "title": doc.get("title", ""),
                    "summary": doc.get("summary", ""),
                    "url": doc.get("url", ""),
                    "source": doc.get("source", "Live News"),
                    "category": doc.get("category", "General"),
                    "published_at": str(doc.get("published_at", "Recent")),
                    "verification_status": verif_res.get("prediction"),
                    "confidence": verif_res.get("calibrated_confidence") or verif_res.get("confidence"),
                })
        except Exception:
            pass

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
            for entry in parsed.entries[:10]:
                link = entry.get("link", "")
                title = entry.get("title", "")
                if not link or not title:
                    continue

                headline_hash = generate_headline_hash(title, link)
                doc = {
                    "title": title,
                    "summary": entry.get("summary", "")[:350],
                    "url": link,
                    "source": feed_info["source"],
                    "category": feed_info.get("category", "General"),
                    "headline_hash": headline_hash,
                    "published_at": now,
                    "verified": False,
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
                    except Exception:
                        pass
        except Exception:
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

    # Find article
    doc = await MongoDBManager.db.live_news.find_one({
        "$or": [
            {"id": article_id},
            {"_id": article_id}
        ]
    })
    if not doc:
        raise HTTPException(status_code=404, detail="Article not found")

    title = doc.get("title", "")
    summary = doc.get("summary", "")
    url = doc.get("url", "")

    engine = get_fusion_engine()
    verif_res = engine.verify_content(
        text=summary,
        title=title,
        url=url,
        retrieve_evidence=True
    )

    # Update in database
    await MongoDBManager.db.live_news.update_one(
        {"_id": doc["_id"]},
        {
            "$set": {
                "verified": True,
                "verification_result": verif_res
            }
        }
    )

    return LiveNewsItem(
        id=str(doc["_id"]),
        title=title,
        summary=summary,
        url=url,
        source=doc.get("source", "Live News"),
        published_at=str(doc.get("published_at", "Recent")),
        verified=True,
        verification_result=verif_res
    )
