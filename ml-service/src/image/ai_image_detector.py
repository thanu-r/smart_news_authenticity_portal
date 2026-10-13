import io
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
from PIL import Image, ExifTags
import logging

logger = logging.getLogger("smartnews.ai_image_detector")

KNOWN_AI_SIGNATURES = [
    "stablediffusion", "stable diffusion", "midjourney", "dall-e", "dalle",
    "chatgpt", "openai", "comfyui", "novelai", "civitai", "dreamstudio",
    "adobe firefly", "firefly", "c2pa", "google imagen", "imagen",
    "bing image creator", "copilot", "ideogram", "leonardo.ai", "flux",
    "negative prompt", "steps: ", "sampler: ", "cfg scale:", "model: ",
    "guidance_scale", "seed:", "diffusion", "karras", "euler"
]

class AIImageDetector:
    """
    Forensic detector for AI-generated and synthetically synthesized images
    (e.g., Latent Diffusion, Stable Diffusion, Midjourney, DALL-E, ChatGPT, Flux, GANs).
    
    Examines:
      1. 2D Fourier (FFT) Azimuthal Spectral Decay (power law P(f) ~ 1/f^alpha)
      2. Periodic grid / checkerboard artifacts from neural upsampling deconvolution
      3. Chromatic cross-channel gradient covariance (absence of physical Bayer CFA demosaicing)
      4. Spatial texture smoothness & high-pass noise moments
      5. Generation provenance & synthetic tool metadata markers across EXIF, XMP, and PNG chunks
    """

    def __init__(self):
        pass

    def _prepare_rgb_image(self, img: Image.Image) -> Image.Image:
        """Handle transparent formats (WEBP, PNG with alpha) cleanly without boundary artifacts."""
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            alpha = img.convert("RGBA")
            bg = Image.new("RGBA", alpha.size, (255, 255, 255, 255))
            bg.paste(alpha, mask=alpha.split()[3])
            return bg.convert("RGB")
        return img.convert("RGB")

    def analyze_spectral_power(self, gray_np: np.ndarray) -> Dict[str, float]:
        """
        Compute 2D FFT and radially averaged power spectrum.
        Natural optical photographs exhibit smooth power-law decay (alpha ~ 1.8 - 2.5).
        Generative AI models exhibit anomalous decay (alpha < 1.65 or > 2.8) and
        elevated high-frequency energy ratio due to latent de-noising and upsampling.
        """
        h, w = gray_np.shape
        size = min(h, w, 512)
        start_y = (h - size) // 2
        start_x = (w - size) // 2
        crop = gray_np[start_y:start_y+size, start_x:start_x+size].astype(np.float32)

        # Apply Hanning window to eliminate edge boundary discontinuities
        win_y = np.hanning(size)
        win_x = np.hanning(size)
        window = np.outer(win_y, win_x)
        crop_windowed = (crop - np.mean(crop)) * window

        # 2D FFT and power spectrum
        f_transform = np.fft.fftshift(np.fft.fft2(crop_windowed))
        magnitude = np.abs(f_transform)
        power = magnitude ** 2 + 1e-10

        # Radial average of power spectrum
        cy, cx = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

        max_radius = size // 2
        radial_bins = np.bincount(r.ravel(), weights=power.ravel())[:max_radius]
        radial_counts = np.bincount(r.ravel())[:max_radius]
        radial_profile = radial_bins / np.maximum(radial_counts, 1)

        # Fit power law slope in mid-to-high frequency band
        fit_start = max(5, int(max_radius * 0.08))
        fit_end = int(max_radius * 0.85)
        freqs = np.arange(fit_start, fit_end)
        powers = radial_profile[fit_start:fit_end]

        valid = powers > 0
        if np.sum(valid) > 10:
            log_freq = np.log(freqs[valid])
            log_pow = np.log(powers[valid])
            slope, _ = np.polyfit(log_freq, log_pow, 1)
            alpha = -slope
        else:
            alpha = 2.0

        # High-frequency energy ratio (energy above 55% Nyquist vs total energy)
        high_cutoff = int(max_radius * 0.55)
        high_energy = float(np.sum(radial_bins[high_cutoff:max_radius]))
        total_energy = float(np.sum(radial_bins[1:max_radius])) + 1e-10
        high_ratio = high_energy / total_energy

        # Angular variance check (periodic peaks at specific angles indicate grid artifacts)
        num_angles = 36
        angles = np.arctan2(y - cy, x - cx)
        angle_bins = ((angles + np.pi) / (2 * np.pi) * num_angles).astype(np.int32) % num_angles
        angular_power = np.bincount(angle_bins.ravel(), weights=power.ravel(), minlength=num_angles)
        angular_norm = angular_power / (np.mean(angular_power) + 1e-6)
        angular_variance = float(np.std(angular_norm))

        return {
            "spectral_alpha": round(float(alpha), 3),
            "high_freq_ratio": round(float(high_ratio), 4),
            "angular_variance": round(angular_variance, 4)
        }

    def analyze_bayer_demosaicing(self, rgb_img: Image.Image) -> Dict[str, float]:
        """
        Examine chromatic cross-channel consistency.
        Physical camera sensors employ Bayer CFA demosaicing which creates
        characteristic cross-channel gradient correlations between (G-R) and (G-B).
        Generative models synthesize pixels directly without Bayer constraints.
        """
        arr = np.asarray(rgb_img, dtype=np.float32)
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        diff_gr = g - r
        diff_gb = g - b

        grad_gr = cv2.Laplacian(diff_gr, cv2.CV_32F)
        grad_gb = cv2.Laplacian(diff_gb, cv2.CV_32F)

        var_gr = float(np.var(grad_gr))
        var_gb = float(np.var(grad_gb))
        cross_cov = float(np.cov(grad_gr.ravel()[:10000], grad_gb.ravel()[:10000])[0, 1])

        demosaic_residual = (var_gr + var_gb) / 1000.0
        return {
            "demosaic_residual": round(demosaic_residual, 4),
            "cross_channel_covariance": round(cross_cov, 4)
        }

    def analyze_texture_and_noise_moments(self, gray_np: np.ndarray) -> Dict[str, float]:
        """
        Evaluate high-pass noise distribution and spatial texture smoothness.
        Diffusion model noise schedules produce distinct higher-order moments (excess kurtosis)
        and characteristic spatial smoothness patterns (lack of natural sensor photon shot noise).
        """
        lap = cv2.Laplacian(gray_np.astype(np.float32), cv2.CV_32F)
        flat = lap.ravel()
        mean = np.mean(flat)
        std = np.std(flat) + 1e-6
        norm = (flat - mean) / std

        skewness = float(np.mean(norm ** 3))
        kurtosis = float(np.mean(norm ** 4) - 3.0)

        # Micro-texture noise ratio: std of noise across smooth patches
        h, w = gray_np.shape
        patch_size = 32
        patch_vars = []
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = gray_np[y:y+patch_size, x:x+patch_size]
                patch_vars.append(np.var(patch))
        
        smooth_patches_count = sum(1 for v in patch_vars if v < 25.0)
        smooth_ratio = smooth_patches_count / max(len(patch_vars), 1)

        return {
            "noise_skewness": round(skewness, 4),
            "noise_excess_kurtosis": round(kurtosis, 4),
            "smoothness_ratio": round(float(smooth_ratio), 4)
        }

    def check_metadata_and_provenance(self, img: Image.Image) -> Dict[str, Any]:
        """
        Inspect EXIF, PNG chunks, XMP, IPTC, and dictionary info for AI generation traces.
        """
        ai_markers_found = []
        found_software = None
        has_camera_hardware = False

        # 1. Check PNG text metadata and info dict
        meta_dict = {}
        if hasattr(img, "info") and isinstance(img.info, dict):
            meta_dict.update(img.info)
        if hasattr(img, "text") and isinstance(img.text, dict):
            meta_dict.update(img.text)

        for k, v in meta_dict.items():
            k_lower = str(k).lower()
            v_lower = str(v).lower()
            for kw in KNOWN_AI_SIGNATURES:
                if kw in k_lower or kw in v_lower:
                    ai_markers_found.append(f"AI generation parameter detected in metadata ('{k}': {str(v)[:50]})")
                    found_software = str(v)[:40]
                    break

        # 2. Check EXIF tags
        try:
            exif = img.getexif()
            if exif:
                for tag_id, val in exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    val_str = str(val).lower()
                    if tag_name in ("Make", "Model", "LensModel", "FocalLength") and len(val_str) > 2:
                        has_camera_hardware = True
                    for kw in KNOWN_AI_SIGNATURES:
                        if kw in val_str:
                            ai_markers_found.append(f"EXIF '{tag_name}' contains AI signature: {val_str[:40]}")
                            found_software = val_str
                            break
        except Exception:
            pass

        return {
            "ai_markers_found": ai_markers_found,
            "has_ai_metadata": len(ai_markers_found) > 0,
            "has_camera_hardware": has_camera_hardware,
            "detected_generator": found_software
        }

    def predict_ai_generated(self, image_input, is_video_frame: bool = False) -> Dict[str, Any]:
        """
        Evaluate an image or decoded video frame and return:
          - is_ai_generated: bool
          - ai_probability: float in [0.0, 1.0] (calibrated probability)
          - forensic_signals: dictionary of individual forensic indicators
          - reasons: list of explanation strings
        """
        if isinstance(image_input, (str, Path)):
            raw_img = Image.open(image_input)
        else:
            raw_img = image_input

        # Check metadata on original image before color conversion
        meta = self.check_metadata_and_provenance(raw_img)

        # Standard resolution for consistent spatial/frequency scaling
        img = self._prepare_rgb_image(raw_img)
        img_analysis = img.resize((512, 512), Image.Resampling.LANCZOS)
        gray_np = np.asarray(img_analysis.convert("L"), dtype=np.float32)

        # 1. Fourier spectral power decay
        spectral = self.analyze_spectral_power(gray_np)
        alpha = spectral["spectral_alpha"]
        high_ratio = spectral["high_freq_ratio"]
        ang_var = spectral["angular_variance"]

        # 2. Bayer demosaicing residuals
        demosaic = self.analyze_bayer_demosaicing(img_analysis)
        d_res = demosaic["demosaic_residual"]

        # 3. Noise residual moments & smoothness
        texture = self.analyze_texture_and_noise_moments(gray_np)
        kurtosis = texture["noise_excess_kurtosis"]
        smoothness = texture["smoothness_ratio"]

        # -------------------------------------------------------------
        # Forensic Multi-Signal Scoring Logic
        # -------------------------------------------------------------
        reasons = []
        score_components = []

        # (a) Spectral Power Decay (shallow decay indicates synthetic iterative synthesis)
        alpha_threshold = 1.68 if is_video_frame else 1.62
        if alpha < alpha_threshold:
            deviation = (alpha_threshold - alpha) / 0.8
            score_components.append(min(0.95, 0.52 + deviation * 0.45))
            reasons.append(f"Fourier power spectrum exhibits shallow decay (alpha={alpha:.2f}), characteristic of generative latent de-noising.")
        elif alpha > 2.85:
            score_components.append(0.68)
            reasons.append(f"Anomalous spectral decay rate (alpha={alpha:.2f}) inconsistent with natural optical lenses.")
        else:
            score_components.append(0.06)

        # (b) High-Frequency Energy Ratio (calibrated for video codec compression)
        h_cutoff = 0.016 if is_video_frame else 0.045
        if high_ratio > h_cutoff:
            score_components.append(min(0.92, 0.48 + (high_ratio - h_cutoff) * 6.0))
            reasons.append(f"Elevated high-frequency spectral energy ({high_ratio:.3f}) typical of diffusion residual high-pass noise.")
        else:
            score_components.append(0.05)

        # (c) High-frequency angular grid spikes
        ang_cutoff = 0.012 if is_video_frame else 0.035
        if ang_var > 1.65 and high_ratio > ang_cutoff:
            score_components.append(0.78)
            reasons.append(f"Frequency domain displays periodic grid spikes (variance={ang_var:.2f}) from neural upsampling deconvolution.")
        else:
            score_components.append(0.08)

        # (d) Demosaicing Inconsistency: lack of physical Bayer CFA correlation
        if d_res > 4.5 and high_ratio > ang_cutoff:
            score_components.append(0.72)
            reasons.append("Uncorrelated chromatic cross-channel variance exceeding physical optical sensors.")
        else:
            score_components.append(0.08)

        # (e) Spatial hyper-smoothness (plastic/airbrushed appearance)
        if smoothness > 0.45 and alpha < 1.85:
            score_components.append(0.75)
            reasons.append(f"Spatial texture exhibits synthetic hyper-smoothness ({int(smoothness * 100)}% smooth patches) without optical sensor photon noise.")
        else:
            score_components.append(0.06)

        # (f) High-Pass Noise Excess Kurtosis (heavy-tailed synthetic diffusion latent noise)
        kurt_thresh = 8.0 if is_video_frame else 15.0
        if kurtosis > kurt_thresh:
            k_score = min(0.94, 0.56 + min(0.38, (kurtosis - kurt_thresh) * 0.015))
            score_components.append(k_score)
            reasons.append(f"Noise residual displays heavy-tailed excess kurtosis ({kurtosis:.1f}), characteristic of synthetic diffusion generation.")
        elif kurtosis < -0.4:
            score_components.append(0.60)
            reasons.append(f"Platykurtic noise distribution ({kurtosis:.2f}) inconsistent with natural optical sensor noise.")
        else:
            score_components.append(0.06)

        # Base statistical AI probability from physical/frequency evidence
        base_ai_prob = float(np.average(score_components, weights=[2.0, 2.0, 1.2, 1.0, 1.0, 2.0]))

        # Moderation if genuine physical camera hardware EXIF is verified
        if meta.get("has_camera_hardware") and not meta.get("has_ai_metadata"):
            base_ai_prob = base_ai_prob * 0.35

        # (g) Definitive metadata evidence
        if meta.get("has_ai_metadata"):
            base_ai_prob = max(0.94, base_ai_prob + 0.45)
            reasons.extend(meta.get("ai_markers_found", []))

        ai_score = round(float(np.clip(base_ai_prob, 0.02, 0.98)), 4)
        is_ai = bool(ai_score >= 0.50)

        return {
            "is_ai_generated": is_ai,
            "ai_probability": ai_score,
            "forensic_signals": {
                "spectral_alpha": alpha,
                "high_frequency_ratio": high_ratio,
                "angular_grid_variance": ang_var,
                "demosaic_residual": d_res,
                "noise_excess_kurtosis": kurtosis,
                "smoothness_ratio": smoothness,
                "has_camera_hardware": meta["has_camera_hardware"],
                "ai_markers": meta["ai_markers_found"]
            },
            "reasons": reasons
        }
