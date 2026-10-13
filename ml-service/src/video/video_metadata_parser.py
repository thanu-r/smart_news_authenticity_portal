import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("smartnews.video_metadata")

KNOWN_AI_VIDEO_SIGNATURES = [
    "runway", "pika", "sora", "openai", "kling", "luma", "dream machine",
    "comfyui", "stablediffusion", "stable diffusion", "svd", "animatediff",
    "gen-2", "gen-3", "haiper", "deforum", "midjourney", "cogvideo",
    "moonvalley", "kaiber", "morph studio", "viggle", "hedra"
]

KNOWN_CAMERA_HARDWARE_SIGNATURES = [
    "apple", "iphone", "samsung", "galaxy", "sony", "canon", "nikon",
    "gopro", "dji", "panasonic", "fujifilm", "olympus", "pixel", "xiaomi"
]

def inspect_video_container_metadata(video_path: str) -> Dict[str, Any]:
    """
    Inspect the raw binary header and metadata boxes (MP4/MOV atoms, WEBM EBML)
    of a video file without requiring external ffprobe dependencies.
    
    Detects:
      1. Known AI video generator tags and watermark/comment strings.
      2. Camera hardware encoder signatures (Apple, Samsung, Sony, GoPro, DJI).
      3. Software creation libraries (e.g., Lavf, FFmpeg web encoders).
      4. Audio track multiplexing indicators.
    """
    if not os.path.exists(video_path):
        return {
            "has_ai_metadata": False,
            "has_camera_hardware": False,
            "detected_signatures": [],
            "encoder_tool": None,
            "has_audio_track": False
        }

    file_size = os.path.getsize(video_path)
    read_size = min(file_size, 1024 * 512)  # First 512 KB

    detected_signatures = []
    has_camera_hardware = False
    encoder_tool = None
    has_audio_track = False

    try:
        with open(video_path, "rb") as f:
            header_bytes = f.read(read_size)
            # Also read the tail 128 KB where MP4 moov atom is sometimes placed
            if file_size > read_size:
                f.seek(max(0, file_size - 128 * 1024))
                tail_bytes = f.read()
            else:
                tail_bytes = b""

        combined_bytes = header_bytes + tail_bytes
        lower_bytes = combined_bytes.lower()

        # 1. Check for AI generator signatures
        for sig in KNOWN_AI_VIDEO_SIGNATURES:
            if sig.encode("utf-8") in lower_bytes:
                detected_signatures.append(f"AI generator token identified in container metadata: '{sig}'")

        # 2. Check for camera hardware signatures
        for cam in KNOWN_CAMERA_HARDWARE_SIGNATURES:
            if cam.encode("utf-8") in lower_bytes:
                has_camera_hardware = True
                break

        # 3. Detect audio track presence via MP4/MOV 'soun' handler atom or audio codec tags
        if b"soun" in combined_bytes or b"mp4a" in combined_bytes or b"aac " in lower_bytes or b"ac-3" in lower_bytes:
            has_audio_track = True

        # 4. Extract encoder tool strings
        if b"lavf" in lower_bytes:
            encoder_tool = "FFmpeg/Lavf"
        elif b"quicktime" in lower_bytes:
            encoder_tool = "Apple QuickTime"
        elif b"x264" in lower_bytes:
            encoder_tool = "x264"

    except Exception as e:
        logger.warning(f"inspect_video_container_metadata error for {video_path}: {e}")

    return {
        "has_ai_metadata": len(detected_signatures) > 0,
        "has_camera_hardware": has_camera_hardware,
        "detected_signatures": detected_signatures,
        "encoder_tool": encoder_tool,
        "has_audio_track": has_audio_track,
        "file_size_bytes": file_size
    }
