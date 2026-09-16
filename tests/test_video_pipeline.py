import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest

# Add ml-service to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml-service"))

from src.video.frame_sampler import VideoFrameSampler
from src.video.video_classifier import VideoForensicClassifier

def create_dummy_video(path: str, num_frames: int = 15, is_manipulated: bool = False):
    """Generate a lightweight temporary video for unit testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 10, (128, 128))
    for i in range(num_frames):
        if is_manipulated and i in [5, 6]:
            # Discontinuous / spliced frame
            frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        else:
            frame = np.full((128, 128, 3), 100 + i * 2, dtype=np.uint8)
        out.write(frame)
    out.release()

def test_frame_sampler():
    """Verify video keyframe decoding, timestamp generation, and metadata extraction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "test_sample.mp4")
        create_dummy_video(vid_path, num_frames=20)

        sampler = VideoFrameSampler(target_fps=2.0, max_frames=10)
        frames, meta = sampler.sample_frames(vid_path)

        assert len(frames) > 0
        assert len(frames) <= 10
        assert "duration_seconds" in meta
        assert "fps" in meta
        assert "resolution" in meta

        first_frame = frames[0]
        assert "frame_index" in first_frame
        assert "timestamp" in first_frame
        assert "image" in first_frame
        assert "temporal_diff" in first_frame
        assert hasattr(first_frame["image"], "size")

def test_video_classifier_inference():
    """Verify end-to-end video classification and suspicious timestamp detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "test_manip.mp4")
        create_dummy_video(vid_path, num_frames=20, is_manipulated=True)

        classifier = VideoForensicClassifier(target_fps=2.0, max_frames=10)
        classifier.load()

        res = classifier.predict_video(vid_path)
        assert "prediction" in res
        assert res["prediction"] in ["REAL", "FAKE", "DEEPFAKE", "FAKE_MANIPULATED", "AI_GENERATED", "SUSPICIOUS"]
        assert "confidence" in res
        assert 0.0 <= res["confidence"] <= 1.0
        assert "probabilities" in res
        assert "fake" in res["probabilities"]
        assert "real" in res["probabilities"]
        assert "suspicious" in res["probabilities"]
        assert "suspicious_timestamps" in res
        assert "explanation" in res
        assert len(res["explanation"]) > 10

def test_corrupt_video_handling():
    """Ensure non-existent or corrupted video inputs fail gracefully with SUSPICIOUS status."""
    classifier = VideoForensicClassifier()
    classifier.load()

    res = classifier.predict_video("nonexistent_corrupt_path_9921.mp4")
    assert res["prediction"] == "SUSPICIOUS"
    assert "error" in res
    assert res["confidence"] == 0.5
