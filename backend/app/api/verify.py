import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from bson import ObjectId
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
import logging

logger = logging.getLogger("smartnews.verify")

from backend.app.database.connection import MongoDBManager
from backend.app.auth.jwt_handler import get_optional_user
from backend.app.schemas.user_schemas import UserResponse
from backend.app.schemas.verification_schemas import (
    TextVerificationRequest, UrlVerificationRequest, VerificationResponse
)
from backend.app.utils.safe_fetcher import safe_fetch_url, SSRFSecurityError, SafeFetchError
from backend.app.utils.article_extractor import extract_article_from_html, ArticleExtractionError

# Import ML service engines
from src.multimodal.fusion_engine import MultimodalDecisionFusionEngine
from src.image.image_classifier import ImageForensicClassifier
from src.video.video_classifier import VideoForensicClassifier

router = APIRouter(prefix="/api/verify", tags=["Verification"])

# Global singleton engine instance
_fusion_engine: Optional[MultimodalDecisionFusionEngine] = None

def get_fusion_engine() -> MultimodalDecisionFusionEngine:
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = MultimodalDecisionFusionEngine()
        _fusion_engine.load()
    return _fusion_engine

UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

async def _persist_verification(record: dict, current_user: Optional[UserResponse]) -> str:
    """Save verification record to MongoDB if database is available."""
    verification_id = str(uuid.uuid4())
    record["_id"] = verification_id
    record["user_id"] = current_user.id if current_user else None
    record["created_at"] = datetime.now(timezone.utc)

    if MongoDBManager.db is not None:
        try:
            await MongoDBManager.db.verifications.insert_one(record)
        except Exception:
            pass
    return verification_id

@router.post("/text", response_model=VerificationResponse)
async def verify_text(
    req: TextVerificationRequest,
    current_user: Optional[UserResponse] = Depends(get_optional_user)
):
    """Verify raw news text or statement against text transformer and evidence retrieval."""
    engine = get_fusion_engine()
    title = req.title or req.headline
    model_type = getattr(req, "model_type", None) or "transformer"
    result = engine.verify_content(
        text=req.text,
        title=title,
        url=req.url,
        author=req.author,
        model_type=model_type,
        retrieve_evidence=True
    )

    verif_id = await _persist_verification({
        "title": title,
        "text_snippet": req.text[:300],
        "source_url": req.url,
        "author": req.author,
        **result
    }, current_user)

    evidence_items = result.get("evidence_retrieval", {}).get("evidence_list") if isinstance(result.get("evidence_retrieval"), dict) else None

    return VerificationResponse(
        id=verif_id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        calibrated_confidence=result.get("calibrated_confidence", result["confidence"]),
        probabilities=result["probabilities"],
        modalities_analyzed=result["modalities_analyzed"],
        title=title,
        text_snippet=req.text[:300],
        source_url=req.url,
        text_analysis=result.get("text_analysis"),
        image_forensics=None,
        video_forensics=None,
        source_credibility=result.get("source_credibility"),
        evidence_retrieval=result.get("evidence_retrieval"),
        evidence=evidence_items,
        cross_modal_consistency=result.get("cross_modal_consistency"),
        explanation=result["explanation"],
        created_at=datetime.now(timezone.utc)
    )

@router.post("/url", response_model=VerificationResponse)
async def verify_url(
    req: UrlVerificationRequest,
    current_user: Optional[UserResponse] = Depends(get_optional_user)
):
    """Fetch article content from URL safely, evaluate source credibility, and run verification."""
    url = req.url.strip()
    logger.info(f"verify_url: Received request for URL: '{url}'")

    # Safe SSRF-protected fetch
    try:
        html, final_url, status_code = safe_fetch_url(url)
        logger.info(f"verify_url: Successfully fetched '{final_url}', HTTP {status_code}, length={len(html)} bytes")
    except SSRFSecurityError as e:
        logger.warning(f"verify_url: SSRF violation or restricted target blocked: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Restricted URL destination: {str(e)}"
        )
    except SafeFetchError as e:
        logger.error(f"verify_url: Unable to fetch URL '{url}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to access URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"verify_url: Unexpected network error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch content from URL: {str(e)}"
        )

    # Structured article extraction
    try:
        article = extract_article_from_html(html, final_url)
        title = article["title"]
        text_content = article["text"]
        author = article.get("author")
        logger.info(
            f"verify_url: Extracted article title='{title[:60]}...', length={len(text_content)} chars, "
            f"author={author}, domain='{article['domain']}'"
        )
    except ArticleExtractionError as e:
        logger.warning(f"verify_url: Article extraction failed for '{url}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

    engine = get_fusion_engine()
    model_type = getattr(req, "model_type", None) or "transformer"
    result = engine.verify_content(
        text=text_content,
        title=title,
        url=final_url,
        author=author,
        model_type=model_type,
        retrieve_evidence=True
    )

    logger.info(
        f"verify_url: Pipeline complete -> prediction={result.get('prediction')}, "
        f"confidence={result.get('confidence')}"
    )

    verif_id = await _persist_verification({
        "title": title,
        "text_snippet": text_content[:300],
        "source_url": final_url,
        "author": author,
        **result
    }, current_user)

    modality_breakdown = {
        "text": result.get("text_analysis"),
        "source": result.get("source_credibility"),
        "evidence": result.get("evidence_retrieval"),
        "image": None,
        "video": None
    }

    evidence_items = result.get("evidence_retrieval", {}).get("evidence_list") if isinstance(result.get("evidence_retrieval"), dict) else None

    return VerificationResponse(
        id=verif_id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        calibrated_confidence=result.get("calibrated_confidence", result["confidence"]),
        probabilities=result["probabilities"],
        modalities_analyzed=result["modalities_analyzed"],
        title=title,
        text_snippet=text_content[:300],
        source_url=final_url,
        text_analysis=result.get("text_analysis"),
        image_forensics=None,
        video_forensics=None,
        source_credibility=result.get("source_credibility"),
        evidence_retrieval=result.get("evidence_retrieval"),
        evidence=evidence_items,
        cross_modal_consistency=result.get("cross_modal_consistency"),
        modality_breakdown=modality_breakdown,
        modality_weights=result.get("modality_weights", {"text": 0.6, "source": 0.4}),
        explanation=result["explanation"],
        created_at=datetime.now(timezone.utc)
    )

@router.post("/image", response_model=VerificationResponse)
async def verify_image(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    headline: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    current_user: Optional[UserResponse] = Depends(get_optional_user)
):
    """Upload and verify news image for digital tampering, ELA compression residuals, and noise consistency."""
    upload_file = file or image
    if upload_file is None or not getattr(upload_file, "filename", None):
        logger.error("verify_image: No image file received in request (expected 'file' or 'image' field)")
        raise HTTPException(
            status_code=400,
            detail="No image file provided in request. Please upload an image using field 'file' or 'image'."
        )

    # Validate file extension
    ext = Path(upload_file.filename).suffix.lower()
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
    if ext not in allowed_exts:
        logger.warning(f"verify_image: Unsupported file extension '{ext}' for file {upload_file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Supported formats: {', '.join(sorted(allowed_exts))}"
        )

    contents = await upload_file.read()
    if not contents or len(contents) == 0:
        logger.error("verify_image: Empty file content (0 bytes)")
        raise HTTPException(status_code=400, detail="Uploaded image file is empty (0 bytes).")

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.error(f"verify_image: Failed to decode image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image format: {str(e)}")

    logger.info(
        f"verify_image: Received image '{upload_file.filename}', size={len(contents)} bytes, "
        f"dimensions={pil_image.size}, mode={pil_image.mode}"
    )

    final_title = title or headline
    engine = get_fusion_engine()
    result = engine.verify_content(
        text=text,
        title=final_title,
        image_input=pil_image,
        retrieve_evidence=bool(text or final_title)
    )

    logger.info(
        f"verify_image: Inference complete -> prediction={result.get('prediction')}, "
        f"confidence={result.get('confidence')}, probabilities={result.get('probabilities')}"
    )

    verif_id = await _persist_verification({
        "title": final_title or upload_file.filename,
        "text_snippet": (text[:300] if text else "Image Forensic Analysis"),
        "media_type": "image",
        "filename": upload_file.filename,
        **result
    }, current_user)

    modality_breakdown = {
        "image": result.get("image_forensics"),
        "text": result.get("text_analysis"),
        "video": result.get("video_forensics"),
        "source": result.get("source_credibility")
    }

    evidence_items = result.get("evidence_retrieval", {}).get("evidence_list") if isinstance(result.get("evidence_retrieval"), dict) else None

    return VerificationResponse(
        id=verif_id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        calibrated_confidence=result.get("calibrated_confidence", result["confidence"]),
        probabilities=result["probabilities"],
        modalities_analyzed=result["modalities_analyzed"],
        title=final_title or upload_file.filename,
        text_snippet=text[:300] if text else None,
        source_url=None,
        text_analysis=result.get("text_analysis"),
        image_forensics=result.get("image_forensics"),
        video_forensics=None,
        source_credibility=result.get("source_credibility"),
        evidence_retrieval=result.get("evidence_retrieval"),
        evidence=evidence_items,
        cross_modal_consistency=result.get("cross_modal_consistency"),
        modality_breakdown=modality_breakdown,
        modality_weights={"image": 1.0},
        explanation=result["explanation"],
        created_at=datetime.now(timezone.utc)
    )

@router.post("/video", response_model=VerificationResponse)
async def verify_video(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    headline: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    current_user: Optional[UserResponse] = Depends(get_optional_user)
):
    """Upload and verify news video for deepfake manipulation, synthetic generation, and temporal flicker."""
    upload_file = file or video
    if upload_file is None or not getattr(upload_file, "filename", None):
        logger.error("verify_video: No video file received in request (expected 'file' or 'video' field)")
        raise HTTPException(
            status_code=400,
            detail="No video file provided in request. Please upload a video using field 'file' or 'video'."
        )

    # Validate file extension
    ext = Path(upload_file.filename).suffix.lower()
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if ext not in allowed_exts:
        logger.warning(f"verify_video: Unsupported video extension '{ext}' for file {upload_file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported formats: {', '.join(sorted(allowed_exts))}"
        )

    contents = await upload_file.read()
    if not contents or len(contents) == 0:
        logger.error("verify_video: Empty video file content (0 bytes)")
        raise HTTPException(status_code=400, detail="Uploaded video file is empty (0 bytes).")

    # Important: Preserve extension for OpenCV container decoding on Windows
    temp_filename = f"{uuid.uuid4()}{ext}"
    temp_path = UPLOAD_DIR / temp_filename
    with open(temp_path, "wb") as f:
        f.write(contents)

    logger.info(f"verify_video: Received video '{upload_file.filename}', size={len(contents)} bytes, saved to '{temp_path}'")

    final_title = title or headline
    try:
        engine = get_fusion_engine()
        result = engine.verify_content(
            text=text,
            title=final_title,
            video_path=str(temp_path),
            retrieve_evidence=bool(text or final_title)
        )
    except Exception as e:
        logger.error(f"verify_video: Video pipeline inference failure: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze video: {str(e)}")
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass

    logger.info(
        f"verify_video: Completed video analysis -> prediction={result.get('prediction')}, "
        f"confidence={result.get('confidence')}, probabilities={result.get('probabilities')}"
    )

    verif_id = await _persist_verification({
        "title": final_title or upload_file.filename,
        "text_snippet": (text[:300] if text else "Video Deepfake Forensic Analysis"),
        "media_type": "video",
        "filename": upload_file.filename,
        **result
    }, current_user)

    modality_breakdown = {
        "video": result.get("video_forensics"),
        "text": result.get("text_analysis"),
        "image": None,
        "source": result.get("source_credibility")
    }

    evidence_items = result.get("evidence_retrieval", {}).get("evidence_list") if isinstance(result.get("evidence_retrieval"), dict) else None

    return VerificationResponse(
        id=verif_id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        calibrated_confidence=result.get("calibrated_confidence", result["confidence"]),
        probabilities=result["probabilities"],
        modalities_analyzed=result["modalities_analyzed"],
        title=final_title or upload_file.filename,
        text_snippet=text[:300] if text else None,
        image_forensics=None,
        video_forensics=result.get("video_forensics"),
        source_credibility=result.get("source_credibility"),
        evidence_retrieval=result.get("evidence_retrieval"),
        evidence=evidence_items,
        cross_modal_consistency=result.get("cross_modal_consistency"),
        modality_breakdown=modality_breakdown,
        modality_weights={"video": 1.0},
        explanation=result["explanation"],
        created_at=datetime.now(timezone.utc)
    )

@router.post("/multimodal", response_model=VerificationResponse)
async def verify_multimodal(
    title: Optional[str] = Form(None),
    headline: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    model_type: Optional[str] = Form("transformer"),
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    current_user: Optional[UserResponse] = Depends(get_optional_user)
):
    """Verify arbitrary combinations of text, image, video, and source attribution in one unified request."""
    final_title = title or headline
    actual_image = image or file
    has_text = bool(text and text.strip())
    has_title = bool(final_title and final_title.strip())
    has_url = bool(url and url.strip())
    has_img = bool(actual_image is not None and getattr(actual_image, 'filename', None))
    has_vid = bool(video is not None and getattr(video, 'filename', None))

    if not (has_text or has_title or has_url or has_img or has_vid):
        raise HTTPException(
            status_code=400,
            detail="At least one input modality (text, title, url, image, or video) must be provided"
        )

    pil_image = None
    if actual_image is not None and getattr(actual_image, 'filename', None):
        img_bytes = await actual_image.read()
        try:
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            pass

    temp_video_path = None
    if video is not None and getattr(video, 'filename', None):
        vid_bytes = await video.read()
        v_ext = Path(video.filename).suffix.lower() or ".mp4"
        temp_video_path = str(UPLOAD_DIR / f"{uuid.uuid4()}{v_ext}")
        with open(temp_video_path, "wb") as f:
            f.write(vid_bytes)

    try:
        engine = get_fusion_engine()
        result = engine.verify_content(
            text=text,
            title=final_title,
            url=url,
            author=author,
            model_type=model_type or "transformer",
            image_input=pil_image,
            video_path=temp_video_path,
            retrieve_evidence=True
        )
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

    verif_id = await _persist_verification({
        "title": final_title or "Multimodal Verification Submission",
        "text_snippet": (text[:300] if text else "Multimodal Media Analysis"),
        "source_url": url,
        **result
    }, current_user)

    modality_breakdown = {
        "image": result.get("image_forensics"),
        "text": result.get("text_analysis"),
        "video": result.get("video_forensics"),
        "source": result.get("source_credibility")
    }

    evidence_items = result.get("evidence_retrieval", {}).get("evidence_list") if isinstance(result.get("evidence_retrieval"), dict) else None

    return VerificationResponse(
        id=verif_id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        calibrated_confidence=result.get("calibrated_confidence", result["confidence"]),
        probabilities=result["probabilities"],
        modalities_analyzed=result["modalities_analyzed"],
        title=final_title,
        text_snippet=text[:300] if text else None,
        source_url=url,
        text_analysis=result.get("text_analysis"),
        image_forensics=result.get("image_forensics"),
        video_forensics=result.get("video_forensics"),
        source_credibility=result.get("source_credibility"),
        evidence_retrieval=result.get("evidence_retrieval"),
        evidence=evidence_items,
        cross_modal_consistency=result.get("cross_modal_consistency"),
        modality_breakdown=modality_breakdown,
        modality_weights=result.get("modality_weights"),
        explanation=result["explanation"],
        created_at=datetime.now(timezone.utc)
    )
