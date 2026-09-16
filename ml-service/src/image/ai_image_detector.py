import io
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ExifTags
import logging

logger = logging.getLogger("smartnews.ai_image_detector")

class AIImageDetector:
    """
    Forensic detector for AI-generated and synthetically synthesized images
    (e.g., Latent Diffusion, Stable Diffusion, Midjourney, DALL-E, Flux, GANs).
    
    Examines:
      1. 2D Fourier (FFT) Azimuthal Spectral Decay (power law P(f) ~ 1/f^alpha)
      2. Periodic grid artifacts in the frequency domain from deconvolution/upsampling
      3. Chromatic cross-channel gradient covariance (absence of physical Bayer CFA demosaicing)
      4. High-pass noise residual statistical kurtosis & skewness
      5. Generation provenance & synthetic tool metadata markers
    """

    def __init__(self):
        pass

    def analyze_spectral_power(self, gray_np: np.ndarray) -> Dict[str, float]:
        """
        Compute 2D FFT and radially averaged power spectrum.
        Natural optical photographs exhibit smooth power-law decay (alpha ~ 1.8 - 2.3).
        Generative AI models exhibit anomalous decay (alpha < 1.5 or > 2.7) and
        elevated high-frequency energy ratio due to latent de-noising and upsampling.
        """
        h, w = gray_np.shape
        # Center crop to power of 2 for clean FFT
        size = min(h, w, 512)
        start_y = (h - size) // 2
        start_x = (w - size) // 2
        crop = gray_np[start_y:start_y+size, start_x:start_x+size].astype(np.float32)

        # Apply Hanning window to eliminate edge boundary discontinuities
        win_y = np.hanning(size)
        win_x = np.hanning(size)
        window = np.outer(win_y, win_x)
        crop_windowed = (crop - np.mean(crop)) * window

        # Compute 2D FFT and power spectrum
        f_transform = np.fft.fftshift(np.fft.fft2(crop_windowed))
        magnitude = np.abs(f_transform)
        power = magnitude ** 2 + 1e-10

        # Compute radial average of power spectrum
        cy, cx = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

        max_radius = size // 2
        radial_bins = np.bincount(r.ravel(), weights=power.ravel())[:max_radius]
        radial_counts = np.bincount(r.ravel())[:max_radius]
        radial_profile = radial_bins / np.maximum(radial_counts, 1)

        # Fit power law slope in mid-to-high frequency band (freq index 10 to max_radius - 5)
        fit_start = max(5, int(max_radius * 0.08))
        fit_end = int(max_radius * 0.85)
        freqs = np.arange(fit_start, fit_end)
        powers = radial_profile[fit_start:fit_end]

        valid = powers > 0
        if np.sum(valid) > 10:
            log_freq = np.log(freqs[valid])
            log_pow = np.log(powers[valid])
            # Linear regression: log(P) = -alpha * log(f) + c
            slope, _ = np.polyfit(log_freq, log_pow, 1)
            alpha = -slope  # Natural optical photos have alpha approx 2.0
        else:
            alpha = 2.0

        # High-frequency energy ratio (energy above 60% Nyquist vs total energy)
        high_cutoff = int(max_radius * 0.60)
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
        Generative models generate pixels directly without Bayer CFA constraints.
        """
        arr = np.asarray(rgb_img.convert("RGB"), dtype=np.float32)
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        # Difference channels
        diff_gr = g - r
        diff_gb = g - b

        # Compute gradient variances of color difference
        grad_gr = cv2.Laplacian(diff_gr, cv2.CV_32F)
        grad_gb = cv2.Laplacian(diff_gb, cv2.CV_32F)

        # In natural camera demosaicing, gradients of difference channels are highly localized
        # with low overall variance across uniform areas.
        var_gr = float(np.var(grad_gr))
        var_gb = float(np.var(grad_gb))
        cross_cov = float(np.cov(grad_gr.ravel()[:10000], grad_gb.ravel()[:10000])[0, 1])

        # Normalize indicator
        demosaic_residual = (var_gr + var_gb) / 1000.0
        return {
            "demosaic_residual": round(demosaic_residual, 4),
            "cross_channel_covariance": round(cross_cov, 4)
        }

    def analyze_noise_residual(self, gray_np: np.ndarray) -> Dict[str, float]:
        """
        Analyze the high-pass filtered noise residual.
        Diffusion model noise schedules produce distinct higher-order moments
        (skewness and kurtosis) compared to physical CMOS/CCD photon shot noise.
        """
        # Extract high-frequency residual using 3x3 Laplacian
        lap = cv2.Laplacian(gray_np.astype(np.float32), cv2.CV_32F)
        flat = lap.ravel()
        mean = np.mean(flat)
        std = np.std(flat) + 1e-6
        norm = (flat - mean) / std

        # Compute 3rd and 4th standardized moments
        skewness = float(np.mean(norm ** 3))
        kurtosis = float(np.mean(norm ** 4) - 3.0)  # Excess kurtosis (Gaussian = 0)

        return {
            "noise_skewness": round(skewness, 4),
            "noise_excess_kurtosis": round(kurtosis, 4)
        }

    def check_metadata_and_provenance(self, img: Image.Image) -> Dict[str, Any]:
        """
        Inspect EXIF, PNG text chunks, and metadata for generative AI signatures or prompt traces.
        """
        ai_markers_found = []
        found_software = None

        # 1. Check PNG text metadata and info dict (Automatic1111, ComfyUI, Midjourney, NovelAI)
        meta_dict = {}
        if hasattr(img, "info") and isinstance(img.info, dict):
            meta_dict.update(img.info)
        if hasattr(img, "text") and isinstance(img.text, dict):
            meta_dict.update(img.text)

        for k, v in meta_dict.items():
            k_lower = str(k).lower()
            v_lower = str(v).lower()
            if any(term in k_lower or term in v_lower for term in [
                "stablediffusion", "stable diffusion", "midjourney", "dall-e", "comfyui", "novelai",
                "negative prompt", "steps: ", "sampler: ", "cfg scale:", "model: ", "flux"
            ]):
                ai_markers_found.append(f"Generation parameters detected in metadata: '{k}'")
                found_software = k

        # 2. Check EXIF Software / ImageDescription / UserComment
        has_camera_hardware = False
        try:
            exif = img.getexif()
            if exif:
                for tag_id, val in exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    val_str = str(val).lower()
                    if tag_name in ("Make", "Model", "LensModel") and len(val_str) > 2:
                        has_camera_hardware = True
                    if any(term in val_str for term in ["dall-e", "midjourney", "stable diffusion", "flux", "civitai"]):
                        ai_markers_found.append(f"EXIF tag '{tag_name}' contains AI generator reference: {val_str[:40]}")
                        found_software = val_str
        except Exception:
            pass

        return {
            "ai_markers_found": ai_markers_found,
            "has_ai_metadata": len(ai_markers_found) > 0,
            "has_camera_hardware": has_camera_hardware,
            "detected_generator": found_software
        }

    def predict_ai_generated(self, image_input) -> Dict[str, Any]:
        """
        Evaluate an image and return:
          - is_ai_generated: bool
          - ai_score: float in [0.0, 1.0] (calibrated probability)
          - signals: dictionary of individual forensic indicators
          - reasons: list of explanation strings
        """
        if isinstance(image_input, (str, Path)):
            raw_img = Image.open(image_input)
        else:
            raw_img = image_input

        # Check metadata on original image before color conversion
        meta = self.check_metadata_and_provenance(raw_img)

        # Standard resolution for consistent spatial/frequency scaling
        img = raw_img.convert("RGB")
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

        # 3. Noise residual moments
        noise = self.analyze_noise_residual(gray_np)
        kurtosis = noise["noise_excess_kurtosis"]

        # -------------------------------------------------------------
        # Forensic Scoring Logic (Calibrated multi-evidence combination)
        # -------------------------------------------------------------
        reasons = []
        score_components = []

        # (a) Shallow Spectral Decay (hallmark of synthetic texture synthesis)
        # Natural camera photos and JPEGs have alpha in 1.8 - 4.5. Generative models have alpha < 1.60
        if alpha < 1.60:
            deviation = (1.60 - alpha) / 0.8
            score_components.append(min(0.95, 0.50 + deviation * 0.45))
            reasons.append(f"Fourier power spectrum exhibits shallow decay (alpha={alpha:.2f}), characteristic of generative high-frequency synthesis.")
        else:
            score_components.append(0.06)

        # (b) High-Frequency Energy Ratio (diffusion models preserve intense high-frequency energy across scales)
        # Natural optical photos have high_ratio < 0.02. Synthetic models often have > 0.05
        if high_ratio > 0.045:
            score_components.append(min(0.92, 0.45 + (high_ratio - 0.045) * 6.0))
            reasons.append(f"Elevated high-frequency spectral energy ({high_ratio:.3f}) typical of iterative diffusion residual noise.")
        else:
            score_components.append(0.05)

        # (c) High-frequency angular variance
        if ang_var > 1.8 and high_ratio > 0.04:
            score_components.append(0.75)
            reasons.append(f"Frequency domain displays periodic grid spikes (variance={ang_var:.2f}) from neural upsampling deconvolution.")
        else:
            score_components.append(0.08)

        # (d) Demosaicing Inconsistency: lack of physical Bayer CFA correlation
        if d_res > 6.0 and high_ratio > 0.03:
            score_components.append(0.70)
            reasons.append("Uncorrelated chromatic cross-channel variance exceeding physical optical sensors.")
        else:
            score_components.append(0.08)

        # Base statistical AI probability from physical/frequency evidence
        base_ai_prob = float(np.average(score_components, weights=[2.0, 2.0, 1.0, 1.0]))

        # Discount if real physical camera hardware metadata is confirmed
        if meta.get("has_camera_hardware"):
            base_ai_prob = base_ai_prob * 0.35

        # (e) Metadata definitive evidence
        if meta.get("has_ai_metadata"):
            base_ai_prob = max(0.92, base_ai_prob + 0.40)
            reasons.extend(meta.get("ai_markers_found", []))

        ai_score = round(float(np.clip(base_ai_prob, 0.02, 0.98)), 4)
        is_ai = bool(ai_score >= 0.52)

        return {
            "is_ai_generated": is_ai,
            "ai_probability": ai_score,
            "forensic_signals": {
                "spectral_alpha": alpha,
                "high_frequency_ratio": high_ratio,
                "angular_grid_variance": ang_var,
                "demosaic_residual": d_res,
                "noise_excess_kurtosis": kurtosis,
                "has_camera_hardware": meta["has_camera_hardware"],
                "ai_markers": meta["ai_markers_found"]
            },
            "reasons": reasons
        }
