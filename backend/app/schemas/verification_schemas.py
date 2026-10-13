from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class TextVerificationRequest(BaseModel):
    title: Optional[str] = Field(None, description="Headline or title of the news article")
    headline: Optional[str] = Field(None, description="Headline alias")
    text: str = Field(..., min_length=10, description="Body text or claim to verify")
    url: Optional[str] = Field(None, description="Source URL if available")
    author: Optional[str] = Field(None, description="Author attribution if available")
    model_type: Optional[str] = Field("transformer", description="transformer or baseline")

class UrlVerificationRequest(BaseModel):
    url: str = Field(..., description="Web article URL to scrape and verify")
    model_type: Optional[str] = Field("transformer", description="transformer or baseline")

class VerificationResponse(BaseModel):
    id: str
    prediction: str
    confidence: float
    calibrated_confidence: Optional[float] = None
    probabilities: Dict[str, float]
    modalities_analyzed: List[str]
    title: Optional[str] = None
    text_snippet: Optional[str] = None
    source_url: Optional[str] = None
    text_analysis: Optional[Dict[str, Any]] = None
    image_forensics: Optional[Dict[str, Any]] = None
    video_forensics: Optional[Dict[str, Any]] = None
    source_credibility: Optional[Dict[str, Any]] = None
    evidence_retrieval: Optional[Dict[str, Any]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    cross_modal_consistency: Optional[Dict[str, Any]] = None
    modality_breakdown: Optional[Dict[str, Any]] = None
    modality_weights: Optional[Dict[str, float]] = None
    explanation: Dict[str, Any]
    created_at: datetime

class FeedbackRequest(BaseModel):
    verification_id: str
    feedback_type: str = Field(..., description="'agree', 'disagree', or 'flag'")
    user_label: Optional[str] = Field(None, description="User suggested label: 'REAL', 'FAKE', or 'SUSPICIOUS'")
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    id: str
    verification_id: str
    status: str
    message: str
    created_at: datetime

class LiveNewsItem(BaseModel):
    id: str
    title: str
    summary: str
    url: str
    source: str
    published_at: str
    verified: bool = False
    verification_result: Optional[Dict[str, Any]] = None

class AdminStatsResponse(BaseModel):
    total_verifications: int
    total_users: int
    fake_count: int
    real_count: int
    suspicious_count: int
    modality_breakdown: Dict[str, int]
    recent_verifications: List[Dict[str, Any]]
