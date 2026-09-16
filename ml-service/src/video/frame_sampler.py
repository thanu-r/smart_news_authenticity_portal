import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

class VideoFrameSampler:
    """
    Decodes video files using OpenCV and extracts keyframes at a configurable rate.
    Computes inter-frame temporal continuity and optical consistency signals.
    """
    def __init__(self, target_fps: float = 1.0, max_frames: int = 30):
        self.target_fps = target_fps
        self.max_frames = max_frames
        # Load OpenCV Haar Cascade for face detection in frames if available
        self.face_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                try:
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                except Exception:
                    self.face_cascade = None

    def sample_frames(self, video_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Sample frames from a video file and compute temporal discontinuity metrics.
        Returns:
          frames: List of dicts containing frame_index, timestamp, PIL image, face_boxes, diff_score
          video_meta: Dict containing duration, total_frames, original_fps, width, height
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / orig_fps if orig_fps > 0 else 0.0

        # Determine frame skip step
        frame_interval = max(1, int(orig_fps / self.target_fps))
        
        frames = []
        prev_gray = None
        current_frame_idx = 0

        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx % frame_interval == 0:
                timestamp = round(current_frame_idx / orig_fps, 2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                # Compute inter-frame temporal difference (flicker/discontinuity)
                diff_score = 0.0
                if prev_gray is not None:
                    # Normalized Mean Absolute Difference between adjacent sampled frames
                    diff = cv2.absdiff(gray, prev_gray)
                    diff_score = float(np.mean(diff)) / 255.0

                # Detect faces in frame if present
                face_boxes = []
                if self.face_cascade is not None:
                    try:
                        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                        face_boxes = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
                    except Exception:
                        face_boxes = []

                frames.append({
                    "frame_index": current_frame_idx,
                    "timestamp": timestamp,
                    "image": pil_img,
                    "face_detected": len(face_boxes) > 0,
                    "face_boxes": face_boxes,
                    "temporal_diff": round(diff_score, 4)
                })

                prev_gray = gray

            current_frame_idx += 1

        cap.release()

        video_meta = {
            "duration_seconds": round(duration, 2),
            "total_frames": total_frames,
            "fps": round(orig_fps, 2),
            "resolution": f"{width}x{height}",
            "frames_sampled": len(frames)
        }

        return frames, video_meta
