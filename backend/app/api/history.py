import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from bson import ObjectId

from backend.app.database.connection import MongoDBManager
from backend.app.auth.jwt_handler import get_current_active_user, get_optional_user
from backend.app.schemas.user_schemas import UserResponse
from backend.app.schemas.verification_schemas import VerificationResponse, FeedbackResponse

router = APIRouter(prefix="/api/history", tags=["History"])

@router.get("")
async def get_verification_history(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    prediction: Optional[str] = None,
    modality: Optional[str] = None,
    current_user: Optional[UserResponse] = Depends(get_optional_user)
) -> Dict[str, Any]:
    """Retrieve past verification records with filtering, pagination, and feedback status."""
    records = []
    if MongoDBManager.db is None:
        return {"records": [], "count": 0}

    query = {}
    if current_user:
        query["user_id"] = current_user.id
    if prediction:
        query["prediction"] = prediction.upper()
    if modality:
        query["modalities_analyzed"] = modality.lower()

    try:
        cursor = MongoDBManager.db.verifications.find(query).sort("created_at", -1).skip(skip).limit(limit)
        async for doc in cursor:
            records.append({
                "id": str(doc.get("_id", "")),
                "headline": doc.get("title") or (doc.get("text_snippet", "")[:80] if doc.get("text_snippet") else "Media Verification"),
                "modality": doc.get("modalities_analyzed", ["multimodal"])[0] if doc.get("modalities_analyzed") else "multimodal",
                "prediction": doc.get("prediction", "SUSPICIOUS"),
                "confidence": doc.get("confidence", 0.5),
                "calibrated_confidence": doc.get("calibrated_confidence", doc.get("confidence", 0.5)),
                "probabilities": doc.get("probabilities", {"real": 0.33, "fake": 0.33, "suspicious": 0.34}),
                "modalities_analyzed": doc.get("modalities_analyzed", []),
                "title": doc.get("title"),
                "text_snippet": doc.get("text_snippet"),
                "source_url": doc.get("source_url"),
                "explanation": doc.get("explanation", {"summary": "Archived verification."}),
                "user_feedback": doc.get("user_feedback"),
                "created_at": str(doc.get("created_at", datetime.now(timezone.utc)))
            })
    except Exception:
        pass

    return {"records": records, "count": len(records)}

@router.get("/{verification_id}", response_model=VerificationResponse)
async def get_single_verification(verification_id: str):
    """Retrieve complete forensic details for a specific past verification."""
    if MongoDBManager.db is None:
        raise HTTPException(status_code=503, detail="Database service not available")

    doc = await MongoDBManager.db.verifications.find_one({"_id": verification_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Verification record not found")

    return VerificationResponse(
        id=str(doc["_id"]),
        prediction=doc.get("prediction", "SUSPICIOUS"),
        confidence=doc.get("confidence", 0.5),
        calibrated_confidence=doc.get("calibrated_confidence", doc.get("confidence", 0.5)),
        probabilities=doc.get("probabilities", {}),
        modalities_analyzed=doc.get("modalities_analyzed", []),
        title=doc.get("title"),
        text_snippet=doc.get("text_snippet"),
        source_url=doc.get("source_url"),
        text_analysis=doc.get("text_analysis"),
        image_forensics=doc.get("image_forensics"),
        video_forensics=doc.get("video_forensics"),
        source_credibility=doc.get("source_credibility"),
        evidence_retrieval=doc.get("evidence_retrieval"),
        cross_modal_consistency=doc.get("cross_modal_consistency"),
        explanation=doc.get("explanation", {}),
        created_at=doc.get("created_at")
    )

@router.post("/{verification_id}/feedback")
async def submit_history_feedback(
    verification_id: str,
    feedback: Dict[str, Any],
    current_user: Optional[UserResponse] = Depends(get_optional_user)
):
    """Record human-in-the-loop accuracy rating or dispute on a historical verification."""
    is_correct = feedback.get("is_correct", True)
    correct_label = feedback.get("correct_label")
    comments = feedback.get("comments", "")
    now = datetime.now(timezone.utc)
    fb_id = str(uuid.uuid4())

    feedback_doc = {
        "_id": fb_id,
        "verification_id": verification_id,
        "is_correct": is_correct,
        "correct_label": correct_label,
        "comments": comments,
        "user_id": current_user.id if current_user else None,
        "created_at": now
    }

    if MongoDBManager.db is not None:
        try:
            # Store in feedback collection
            await MongoDBManager.db.feedback.insert_one(feedback_doc)
            # Also annotate verification doc
            await MongoDBManager.db.verifications.update_one(
                {"_id": verification_id},
                {
                    "$set": {
                        "user_feedback": {
                            "is_correct": is_correct,
                            "correct_label": correct_label,
                            "recorded_at": now
                        },
                        "disputed": not is_correct
                    }
                }
            )
        except Exception:
            pass

    return {
        "status": "success",
        "feedback_id": fb_id,
        "verification_id": verification_id,
        "is_correct": is_correct,
        "timestamp": now
    }
