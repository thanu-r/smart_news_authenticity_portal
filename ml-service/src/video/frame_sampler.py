import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

from .video_metadata_parser import inspect_video_container_metadata

class VideoFrameSampler:
    """
    Decodes video files across MP4, MOV, AVI, WEBM, and MKV containers.
    Extracts keyframes at a configurable rate, tracks facial continuity across frames,
    measures optical flow affine warp residuals (detecting generative morphing),
    and computes temporal sensor pattern noise (PRNU) correlation.
    """
    def __init__(self, target_fps: float = 2.0, max_frames: int = 30):
        self.target_fps = target_fps
        self.max_frames = max_frames

    def _detect_facial_region(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Fast, robust skin-cluster facial locator."""
        r = frame_bgr[:, :, 2]
        g = frame_bgr[:, :, 1]
        b = frame_bgr[:, :, 0]
        skin = (r > g) & (g > b) & ((r - g) > 10) & (r > 50) & (b < 185)
        if np.sum(skin) > 400:
            ys, xs = np.where(skin)
            min_x, max_x = int(np.min(xs)), int(np.max(xs))
            min_y, max_y = int(np.min(ys)), int(np.max(ys))
            w, h = max_x - min_x, max_y - min_y
            aspect = h / float(w + 1e-4)
            if 0.7 <= aspect <= 2.2 and w > 24 and h > 24:
                return (min_x, min_y, w, h)
        return None

    def _compute_motion_and_prnu(self, gray_curr: np.ndarray, gray_prev: np.ndarray) -> Tuple[float, float]:
        """
        Compute:
          1. Optical flow affine warp residual (high for non-rigid diffusion morphing).
          2. Temporal sensor pattern noise (PRNU) correlation (high for physical camera sensors).
        """
        # 1. Optical flow warp residual
        warp_residual = 0.0
        try:
            flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 2, 12, 2, 5, 1.2, 0)
            u = flow[:, :, 0]
            v = flow[:, :, 1]
            h, w = gray_curr.shape
            y, x = np.mgrid[:h, :w]
            A = np.column_stack([x.ravel(), y.ravel(), np.ones(h * w)])
            pu, _, _, _ = np.linalg.lstsq(A, u.ravel(), rcond=None)
            pv, _, _, _ = np.linalg.lstsq(A, v.ravel(), rcond=None)
            u_pred = A @ pu
            v_pred = A @ pv
            warp_residual = float(np.mean((u.ravel() - u_pred) ** 2 + (v.ravel() - v_pred) ** 2))
        except Exception:
            warp_residual = 0.0

        # 2. Temporal noise correlation (sensor PRNU fingerprint)
        noise_corr = 0.0
        try:
            blur1 = cv2.blur(gray_prev.astype(np.float32), (3, 3))
            blur2 = cv2.blur(gray_curr.astype(np.float32), (3, 3))
            n1 = gray_prev.astype(np.float32) - blur1
            n2 = gray_curr.astype(np.float32) - blur2
            n1_zero = n1 - np.mean(n1)
            n2_zero = n2 - np.mean(n2)
            denom = (np.std(n1_zero) * np.std(n2_zero)) + 1e-8
            noise_corr = float(np.mean(n1_zero * n2_zero) / denom)
        except Exception:
            noise_corr = 0.0

        return warp_residual, noise_corr

    def sample_frames(self, video_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Sample frames from a video file and compute motion-compensated temporal discontinuity metrics.
        Returns:
          frames: List of dicts containing frame_index, timestamp, PIL image, face_detected, face_boxes, temporal_diff, face_jitter
          video_meta: Dict containing duration_seconds, total_frames, fps, resolution, frames_sampled, mean_flicker, face_ratio_var,
                      mean_warp_residual, mean_temporal_prnu_corr, container_analysis
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Container metadata check
        container_info = inspect_video_container_metadata(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open or decode video file: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if orig_fps <= 0 or np.isnan(orig_fps):
            orig_fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        duration = total_frames / orig_fps if total_frames > 0 and orig_fps > 0 else 0.0

        frame_interval = max(1, int(orig_fps / self.target_fps))
        
        frames = []
        prev_small_gray = None
        consecutive_diffs = []
        consecutive_ratios = []
        warp_residuals = []
        noise_correlations = []
        current_frame_idx = 0

        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Fast downsampled analysis frame (256x256) for motion and noise metrics
            small_bgr = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

            # Consecutive frame motion analysis (independent of sampling skip)
            if prev_small_gray is not None:
                # 1. Fast C++ absdiff for flicker
                diff_consec = cv2.absdiff(small_gray, prev_small_gray)
                mean_d = float(np.mean(diff_consec))
                consecutive_diffs.append(mean_d)

                # 2. Skin / Face motion ratio
                r, g, b = small_bgr[:, :, 2], small_bgr[:, :, 1], small_bgr[:, :, 0]
                skin = (r > g) & (g > b) & ((r - g) > 10) & (r > 50) & (b < 185)
                if np.sum(skin) > 200:
                    f_d = float(np.mean(diff_consec[skin]))
                    b_d = float(np.mean(diff_consec[~skin]))
                    consecutive_ratios.append(f_d / (b_d + 1e-3))

                # 3. Optical flow warp residual & PRNU noise correlation (every 2-3 frames for speed)
                if current_frame_idx % max(1, frame_interval // 2) == 0:
                    w_res, n_corr = self._compute_motion_and_prnu(small_gray, prev_small_gray)
                    warp_residuals.append(w_res)
                    noise_correlations.append(n_corr)

            prev_small_gray = small_gray.copy()

            if current_frame_idx % frame_interval == 0:
                timestamp = round(current_frame_idx / orig_fps, 2)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                face_box = self._detect_facial_region(frame)
                has_face = face_box is not None

                interval_diffs = consecutive_diffs[-frame_interval:] if consecutive_diffs else []
                diff_score = float(np.max(interval_diffs)) / 255.0 if interval_diffs else 0.0
                face_jitter = 0.0
                if consecutive_ratios:
                    # Deviation from neutral 1.0 ratio
                    face_jitter = float(np.clip(abs(consecutive_ratios[-1] - 1.0), 0.0, 1.0))

                frames.append({
                    "frame_index": current_frame_idx,
                    "timestamp": timestamp,
                    "image": pil_img,
                    "face_detected": has_face,
                    "face_boxes": [{"x": face_box[0], "y": face_box[1], "w": face_box[2], "h": face_box[3]}] if face_box else [],
                    "temporal_diff": round(diff_score, 4),
                    "face_jitter": round(face_jitter, 4)
                })

            current_frame_idx += 1

        cap.release()

        ext = Path(video_path).suffix.lower()
        if duration == 0.0 and len(frames) > 0:
            duration = round(len(frames) / self.target_fps, 2)

        mean_flicker = float(np.mean(consecutive_diffs)) if consecutive_diffs else 0.0
        face_ratio_var = float(np.var(consecutive_ratios)) if len(consecutive_ratios) >= 3 else 0.0
        mean_warp = float(np.mean(warp_residuals)) if warp_residuals else 0.0
        mean_noise_corr = float(np.mean(noise_correlations)) if noise_correlations else 0.0

        video_meta = {
            "duration_seconds": round(duration, 2),
            "total_frames": total_frames if total_frames > 0 else current_frame_idx,
            "fps": round(orig_fps, 2),
            "resolution": f"{width}x{height}" if width and height else "adaptive",
            "frames_sampled": len(frames),
            "container_format": ext.replace(".", "").upper(),
            "mean_optical_flicker": round(mean_flicker, 2),
            "face_motion_ratio_variance": round(face_ratio_var, 6),
            "mean_warp_residual": round(mean_warp, 4),
            "mean_temporal_prnu_corr": round(mean_noise_corr, 4),
            "container_analysis": container_info
        }

        return frames, video_meta
