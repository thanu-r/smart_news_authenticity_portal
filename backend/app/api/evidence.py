from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from backend.app.database.connection import MongoDBManager

router = APIRouter(prefix="/api/evidence", tags=["Evidence"])

@router.get("/{verification_id}")
async def get_verification_evidence(verification_id: str) -> Dict[str, Any]:
    """Retrieve external corroboration citations, publisher metadata, and stances for a verification."""
    if MongoDBManager.db is None:
        raise HTTPException(status_code=503, detail="Database service not available")

    doc = await MongoDBManager.db.verifications.find_one({"_id": verification_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Verification record not found")

    evidence_data = doc.get("evidence_retrieval")
    if not evidence_data:
        return {
            "verification_id": verification_id,
            "corroboration_status": "NO_EVIDENCE_RECORDED",
            "evidence_list": [],
            "summary": "No external corroboration queries were recorded for this verification."
        }

    return {
        "verification_id": verification_id,
        **evidence_data
    }
