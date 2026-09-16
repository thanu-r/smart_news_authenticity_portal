import io
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance, ExifTags

def compute_ela(image: Image.Image, quality: int = 90, scale: float = 12.0) -> Tuple[np.ndarray, float]:
    """
    Perform Error Level Analysis (ELA).
    Re-saves the image at specified JPEG quality and computes the absolute pixel difference.
    Manipulated/spliced regions that were inserted or re-saved at different compression levels
    exhibit prominent error levels compared to unmodified regions.
    """
    # Convert to RGB
    img_rgb = image.convert("RGB")
    
    # Save to memory buffer with specific JPEG compression
    buffer = io.BytesIO()
    img_rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    # Compute difference
    diff = ImageChops.difference(img_rgb, recompressed)
    
    # Enhance difference for visual analysis
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0:
        max_diff = 1
    scale_factor = 255.0 / max_diff
    enhanced = ImageEnhance.Brightness(diff).enhance(scale_factor)

    # Compute numerical ELA score (mean intensity of residual error)
    diff_arr = np.asarray(diff, dtype=np.float32)
    ela_score = float(np.mean(diff_arr))

    return np.asarray(enhanced), ela_score

def compute_noise_inconsistency(image: Image.Image) -> float:
    """
    Estimate high-frequency noise variance across 8x8 patches.
    Spliced or composite images show inconsistent noise variance across different visual regions.
    """
    img_gray = np.array(image.convert("L"), dtype=np.float32)
    
    # High-pass filter using Laplacian
    laplacian = cv2.Laplacian(img_gray, cv2.CV_32F)
    
    # Divide into 16x16 grid patches and calculate standard deviation per patch
    h, w = laplacian.shape
    patch_size = 32
    stds = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = laplacian[y:y+patch_size, x:x+patch_size]
            stds.append(np.std(patch))

    if not stds:
        return 0.0

    # Coefficient of variation of noise across patches
    noise_variance = float(np.std(stds) / (np.mean(stds) + 1e-6))
    return round(min(1.0, noise_variance), 4)

def extract_metadata_forensics(image: Image.Image) -> Dict[str, Any]:
    """
    Extract EXIF tags and check for editing software traces (Photoshop, GIMP, etc.).
    """
    exif_data = {}
    software_flag = False
    software_name = ""
    has_camera_info = False

    try:
        raw_exif = image.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(value, (str, int, float)):
                    exif_data[tag_name] = value

            # Check Software tag
            software = str(exif_data.get("Software", "")).lower()
            if any(tool in software for tool in ["photoshop", "gimp", "canva", "lightroom", "paint.net"]):
                software_flag = True
                software_name = exif_data.get("Software", "")

            # Check Camera Make / Model
            if "Make" in exif_data or "Model" in exif_data:
                has_camera_info = True
    except Exception:
        pass

    return {
        "has_exif": len(exif_data) > 0,
        "editing_software_detected": software_flag,
        "software_name": software_name,
        "has_camera_metadata": has_camera_info,
        "exif_summary": {k: exif_data[k] for k in list(exif_data.keys())[:6]}
    }

def extract_image_features(image_path_or_obj) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract forensic representation vector and interpretable signals from an image.
    Features:
      - ELA mean, std, 95th percentile
      - Noise variance across patches
      - Color channel variance
      - Image sharpness
      - Metadata indicators
    """
    if isinstance(image_path_or_obj, (str, Path)):
        img = Image.open(image_path_or_obj).convert("RGB")
    else:
        img = image_path_or_obj.convert("RGB")

    # Resize to standard analysis size for consistent feature scale
    img_resized = img.resize((384, 384))
    
    # 1. ELA
    _, ela_score = compute_ela(img_resized)
    
    # 2. Noise inconsistency
    noise_inconsistency = compute_noise_inconsistency(img_resized)
    
    # 3. Color channel statistics
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    r_mean, g_mean, b_mean = np.mean(img_np[:, :, 0]), np.mean(img_np[:, :, 1]), np.mean(img_np[:, :, 2])
    r_std, g_std, b_std = np.std(img_np[:, :, 0]), np.std(img_np[:, :, 1]), np.std(img_np[:, :, 2])

    # 4. Laplacian sharpness
    gray_np = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray_np, cv2.CV_64F).var()) / 1000.0

    # 5. Metadata
    metadata_info = extract_metadata_forensics(img)
    software_val = 1.0 if metadata_info["editing_software_detected"] else 0.0

    # Compact 10-dimensional forensic descriptor
    feature_vector = np.array([
        ela_score / 20.0,
        noise_inconsistency,
        r_mean, g_mean, b_mean,
        r_std, g_std, b_std,
        min(sharpness, 10.0) / 10.0,
        software_val
    ], dtype=np.float32)

    signals = {
        "ela_error_score": round(ela_score, 3),
        "noise_inconsistency": noise_inconsistency,
        "sharpness_index": round(sharpness, 3),
        "metadata_forensics": metadata_info
    }

    return feature_vector, signals
