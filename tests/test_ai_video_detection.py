import os
import sys
import cv2
import numpy as np
import pytest
from pathlib import Path

# Add project root and ml-service/src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ml-service" / "src"))

from video.video_classifier import VideoForensicClassifier
from video.frame_sampler import VideoFrameSampler
from video.video_metadata_parser import inspect_video_container_metadata

DEMO_MEDIA_DIR = r"C:\Users\priya\Downloads\ml-service-efficientnet-upgrade\ml-service\data\demo_media"
SYNTH_VIDEO_1 = r"C:\Users\priya\.vscode\extensions\openai.chatgpt-26.908.40401-win32-x64\webview\assets\codex-home-hero-dark-3f0460c4764f.mp4"
SYNTH_VIDEO_2 = r"C:\Users\priya\.vscode\extensions\openai.chatgpt-26.908.40401-win32-x64\webview\assets\codex-home-hero-light-b28a9df8176a.mp4"
CAMERA_VIDEO_1 = r"C:\Users\priya\Downloads\16546143_1080_1920_30fps.mp4"

@pytest.fixture(scope="module")
def classifier():
    clf = VideoForensicClassifier(target_fps=2.0, max_frames=10)
    clf.load()
    return clf

def test_container_metadata_ai_detector():
    """Verify video container inspection identifies AI generator tokens and camera hardware."""
    if os.path.exists(SYNTH_VIDEO_1):
        meta = inspect_video_container_metadata(SYNTH_VIDEO_1)
        assert meta["has_ai_metadata"] is True
        assert any("openai" in s.lower() for s in meta["detected_signatures"])

    if os.path.exists(CAMERA_VIDEO_1):
        meta_cam = inspect_video_container_metadata(CAMERA_VIDEO_1)
        assert meta_cam["has_ai_metadata"] is False

def test_ai_generated_video_classification(classifier):
    """Verify synthetic / AI-generated videos are classified as AI_GENERATED and never as REAL."""
    target_vid = None
    if os.path.exists(SYNTH_VIDEO_1):
        target_vid = SYNTH_VIDEO_1
    elif os.path.exists(SYNTH_VIDEO_2):
        target_vid = SYNTH_VIDEO_2

    if not target_vid:
        pytest.skip("Synthetic demo video not available")

    res = classifier.predict_video(target_vid)
    assert res["prediction"] == "AI_GENERATED", f"Expected AI_GENERATED, got {res['prediction']}"
    assert res["prediction"] != "REAL", "CRITICAL ERROR: AI-generated video was classified as REAL!"
    assert res["confidence"] >= 0.70
    assert res["deepfake_risk"] == "CRITICAL"
    assert "AI_GENERATED" in res["explanation"] or "synthetic" in res["explanation"].lower() or "ai" in res["explanation"].lower()

def test_generative_motion_warping_detection(classifier):
    """Verify non-rigid optical flow warping from diffusion video generation is flagged."""
    temp_warp_vid = "test_generative_warp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_warp_vid, fourcc, 10.0, (192, 192))
    
    # Generate non-rigid fluid morphing across frames (diffusion latent warping simulation)
    base_frame = np.random.randint(40, 220, (192, 192, 3), dtype=np.uint8)
    for i in range(20):
        # Local sinusoidal texture morphing with dynamically evolving phase
        grid_y, grid_x = np.mgrid[0:192, 0:192]
        dx = float(i) * 0.5 + 8.0 * np.sin((grid_y + i * 15.0) / 15.0) * np.cos((grid_x + i * 10.0) / 15.0)
        dy = float(i) * 0.5 + 8.0 * np.cos((grid_y + i * 10.0) / 15.0) * np.sin((grid_x + i * 15.0) / 15.0)
        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)
        morphed = cv2.remap(base_frame, map_x, map_y, cv2.INTER_LINEAR)
        out.write(morphed)
    out.release()

    try:
        res = classifier.predict_video(temp_warp_vid)
        assert res["prediction"] in ["AI_GENERATED", "FAKE_MANIPULATED", "SUSPICIOUS"]
        assert res["prediction"] != "REAL", "CRITICAL: Generative morphing video was classified as REAL!"
        assert res["temporal_signals"]["mean_warp_residual"] > 1.5
    finally:
        if os.path.exists(temp_warp_vid):
            os.remove(temp_warp_vid)

def test_authentic_camera_video_classification(classifier):
    """Verify genuine camera videos are correctly classified as REAL with low risk."""
    real_v = os.path.join(DEMO_MEDIA_DIR, "0094b77fc378929f95b0bdef1801c3175b854f7c2271b570981e06fada07d35b_real.avi")
    if not os.path.exists(real_v):
        if os.path.exists(CAMERA_VIDEO_1):
            real_v = CAMERA_VIDEO_1
        else:
            pytest.skip("Authentic video sample not found")

    res = classifier.predict_video(real_v)
    assert res["prediction"] == "REAL", f"Authentic video classified as {res['prediction']}"
    assert res["deepfake_risk"] == "LOW"
    assert res["probabilities"]["real"] >= 0.50

def test_deepfake_video_distinction(classifier):
    """Verify deepfake face-swap videos are classified as DEEPFAKE (distinguished from general AI)."""
    fake_v = os.path.join(DEMO_MEDIA_DIR, "0094b77fc378929f95b0bdef1801c3175b854f7c2271b570981e06fada07d35b_fake.avi")
    if not os.path.exists(fake_v):
        pytest.skip("DFDC fake video sample not found")

    res = classifier.predict_video(fake_v)
    assert res["prediction"] in ["DEEPFAKE", "AI_GENERATED", "FAKE_MANIPULATED"]
    assert res["prediction"] != "REAL", "CRITICAL: Deepfake video classified as REAL!"
    assert res["deepfake_risk"] in ["HIGH", "CRITICAL"]

def test_multi_category_video_benchmark_metrics(classifier):
    """
    Calculate accuracy, precision, recall, F1, confusion matrix,
    and false negative rate specifically for AI-generated and synthetic videos.
    """
    evaluation_set = []

    # 1. Authentic Camera Videos
    dfdc_real = os.path.join(DEMO_MEDIA_DIR, "0094b77fc378929f95b0bdef1801c3175b854f7c2271b570981e06fada07d35b_real.avi")
    if os.path.exists(dfdc_real):
        evaluation_set.append((dfdc_real, "REAL"))
    if os.path.exists(CAMERA_VIDEO_1):
        evaluation_set.append((CAMERA_VIDEO_1, "REAL"))

    # 2. AI-Generated Videos
    if os.path.exists(SYNTH_VIDEO_1):
        evaluation_set.append((SYNTH_VIDEO_1, "AI_GENERATED"))
    if os.path.exists(SYNTH_VIDEO_2):
        evaluation_set.append((SYNTH_VIDEO_2, "AI_GENERATED"))

    # 3. Deepfake Face-Swap Videos
    dfdc_fake = os.path.join(DEMO_MEDIA_DIR, "0094b77fc378929f95b0bdef1801c3175b854f7c2271b570981e06fada07d35b_fake.avi")
    if os.path.exists(dfdc_fake):
        evaluation_set.append((dfdc_fake, "DEEPFAKE"))

    assert len(evaluation_set) >= 3, "Insufficient evaluation samples found"

    y_true = []
    y_pred = []
    ai_false_negatives = 0
    total_ai_videos = 0

    for path, ground_truth in evaluation_set:
        res = classifier.predict_video(path)
        pred = res["prediction"]
        y_true.append(ground_truth)
        y_pred.append(pred)

        if ground_truth == "AI_GENERATED":
            total_ai_videos += 1
            if pred == "REAL":
                ai_false_negatives += 1

    # Binary metrics: Fake (including AI_GENERATED, DEEPFAKE, FAKE_MANIPULATED) vs Real
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != "REAL" and yp != "REAL")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "REAL" and yp != "REAL")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != "REAL" and yp == "REAL")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "REAL" and yp == "REAL")

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
    ai_fnr = (ai_false_negatives / total_ai_videos) if total_ai_videos > 0 else 0.0

    print(f"\n--- AI Video Detection Benchmark ---")
    print(f"Total Videos: {len(y_true)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AI Video False-Negative Rate: {ai_fnr * 100:.2f}%")
    print(f"Confusion Matrix (Fake/Real): TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    assert accuracy >= 0.80, f"Accuracy too low: {accuracy}"
    assert ai_fnr == 0.0, f"CRITICAL: AI video false negative rate must be 0% (got {ai_fnr})"
