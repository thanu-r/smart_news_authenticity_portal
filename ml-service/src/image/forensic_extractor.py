import io
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance, ExifTags

# Load Haar cascade once for face region extraction
FACE_CASCADE = None
try:
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
except Exception:
    FACE_CASCADE = None

def compute_ela(image: Image.Image, quality: int = 90) -> Tuple[np.ndarray, float]:
    """
    Perform single-scale Error Level Analysis (ELA).
    Re-saves the image in memory at JPEG quality and returns enhanced visual map & mean score.
    """
    img_rgb = image.convert("RGB")
    buffer = io.BytesIO()
    img_rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    diff = ImageChops.difference(img_rgb, recompressed)
    diff_arr = np.asarray(diff, dtype=np.float32)
    ela_score = float(np.mean(diff_arr))

    # Brightness enhanced visual representation
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0:
        max_diff = 1
    scale_factor = 255.0 / max_diff
    enhanced = ImageEnhance.Brightness(diff).enhance(scale_factor)

    return np.asarray(enhanced), round(ela_score, 4)

def compute_multi_scale_ela(image: Image.Image) -> Dict[str, float]:
    """
    Multi-scale Patch Error Level Analysis (Q=90 and Q=75).
    Evaluates both global error and localized patch contrast.
    Authentic camera photos show spatially uniform ELA across blocks.
    Spliced or inserted regions exhibit prominent patch-level ELA contrast.
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    
    # Q=90 ELA
    buf90 = io.BytesIO()
    img_rgb.save(buf90, "JPEG", quality=90)
    buf90.seek(0)
    diff90 = np.asarray(ImageChops.difference(img_rgb, Image.open(buf90)), dtype=np.float32)
    diff90_gray = np.mean(diff90, axis=2)
    
    # Q=75 ELA
    buf75 = io.BytesIO()
    img_rgb.save(buf75, "JPEG", quality=75)
    buf75.seek(0)
    diff75 = np.asarray(ImageChops.difference(img_rgb, Image.open(buf75)), dtype=np.float32)
    diff75_gray = np.mean(diff75, axis=2)

    # Patch analysis (16x16 spatial grid)
    patch_size = 24
    patches_90 = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            p = diff90_gray[y:y+patch_size, x:x+patch_size]
            patches_90.append(np.mean(p))

    mean_90 = float(np.mean(diff90_gray))
    mean_75 = float(np.mean(diff75_gray))
    
    if patches_90:
        max_patch_90 = float(np.max(patches_90))
        std_patch_90 = float(np.std(patches_90))
        contrast_90 = std_patch_90 / (mean_90 + 1e-4)
    else:
        max_patch_90 = mean_90
        contrast_90 = 0.0

    return {
        "ela_mean_90": round(mean_90, 4),
        "ela_max_patch_90": round(max_patch_90, 4),
        "ela_contrast_90": round(contrast_90, 4),
        "ela_mean_75": round(mean_75, 4)
    }

def compute_noise_inconsistency(image: Image.Image) -> float:
    """
    Estimate high-frequency noise variance inconsistency across image blocks.
    Spliced or composite images show inconsistent noise variance across different visual regions.
    """
    img_gray = np.array(image.convert("L"), dtype=np.float32)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_32F)
    
    h, w = laplacian.shape
    patch_size = 32
    stds = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = laplacian[y:y+patch_size, x:x+patch_size]
            stds.append(np.std(patch))

    if not stds:
        return 0.0

    noise_variance = float(np.std(stds) / (np.mean(stds) + 1e-6))
    return round(min(1.0, noise_variance), 4)

def compute_noise_kurtosis(image: Image.Image) -> float:
    """
    Compute excess kurtosis of the high-pass filtered noise residual.
    Physical sensor photon shot noise follows Gaussian distribution (kurtosis approx 0).
    Nonlinear processing, deepfake blending, and generative synthesis alter kurtosis.
    """
    img_gray = np.array(image.convert("L"), dtype=np.float32)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_32F).ravel()
    mean = np.mean(laplacian)
    std = np.std(laplacian) + 1e-6
    norm = (laplacian - mean) / std
    kurt = float(np.mean(norm ** 4) - 3.0)
    return round(float(np.clip(kurt, -2.0, 15.0)), 4)

def extract_facial_manipulation_signals(image: Image.Image) -> Dict[str, float]:
    """
    Detect human faces and evaluate boundary blending seams and noise discrepancy.
    Face-swaps and deepfake facial insertions produce blending seam artifacts
    along the perimeter of the face and noise discrepancy between facial skin and background.
    """
    img_rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    face_detected = 0.0
    seam_discontinuity = 0.0
    noise_ratio = 1.0
    face_box = None

    # 1. Try Haar Cascade if available
    if FACE_CASCADE is not None:
        try:
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(36, 36))
            if len(faces) > 0:
                face_box = max(faces, key=lambda f: f[2] * f[3])
        except Exception:
            face_box = None

    # 2. Skin-cluster facial region locator (robust fallback)
    if face_box is None:
        r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        skin = (r > g) & (g > b) & ((r - g) > 10) & (r > 50) & (b < 185)
        if np.sum(skin) > 500:
            ys, xs = np.where(skin)
            min_x, max_x = int(np.min(xs)), int(np.max(xs))
            min_y, max_y = int(np.min(ys)), int(np.max(ys))
            w, h = max_x - min_x, max_y - min_y
            aspect = h / float(w + 1e-4)
            if 0.7 <= aspect <= 2.2 and w > 24 and h > 24:
                face_box = (min_x, min_y, w, h)

    if face_box is not None:
        x, y, w, h = face_box
        face_detected = 1.0
        
        # Face crop
        face_crop = gray[y:y+h, x:x+w]
        
        # Perimeter context ring
        pad = 8
        y1 = max(0, y - pad)
        y2 = min(gray.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(gray.shape[1], x + w + pad)
        context_crop = gray[y1:y2, x1:x2]
        
        if face_crop.size > 0 and context_crop.size > 0:
            face_lap = cv2.Laplacian(face_crop, cv2.CV_32F)
            ctx_lap = cv2.Laplacian(context_crop, cv2.CV_32F)
            
            face_noise = float(np.std(face_lap)) + 1e-4
            ctx_noise = float(np.std(ctx_lap)) + 1e-4
            noise_ratio = float(face_noise / ctx_noise)
            
            # Boundary seam gradient: Sobel magnitude along outer 4 pixels of face crop
            boundary_mask = np.zeros_like(face_crop, dtype=np.uint8)
            cv2.rectangle(boundary_mask, (0, 0), (w, h), 255, 4)
            sobel_x = cv2.Sobel(face_crop, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(face_crop, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            boundary_grad = float(np.mean(grad_mag[boundary_mask > 0])) if np.sum(boundary_mask > 0) > 0 else 0.0
            inner_grad = float(np.mean(grad_mag[boundary_mask == 0])) + 1e-4 if np.sum(boundary_mask == 0) > 0 else 1.0
            seam_discontinuity = float(boundary_grad / inner_grad)

    return {
        "face_detected": face_detected,
        "boundary_seam_discontinuity": round(min(seam_discontinuity, 5.0), 4),
        "face_noise_ratio": round(float(np.clip(noise_ratio, 0.1, 5.0)), 4)
    }

def extract_metadata_forensics(image: Image.Image) -> Dict[str, Any]:
    """
    Extract EXIF metadata and inspect for physical camera hardware tags vs. editing software traces.
    """
    exif_data = {}
    software_flag = False
    software_name = ""
    has_camera_hardware = False
    camera_make = ""
    camera_model = ""

    try:
        raw_exif = image.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(value, (str, int, float)):
                    exif_data[tag_name] = value

            software = str(exif_data.get("Software", "")).lower()
            if any(tool in software for tool in ["photoshop", "gimp", "canva", "lightroom", "paint.net", "snapseed", "picsart", "vsco"]):
                software_flag = True
                software_name = exif_data.get("Software", "")

            # Check Camera Make / Model / Lens / Exposure
            if "Make" in exif_data or "Model" in exif_data or "FocalLength" in exif_data or "ISOSpeedRatings" in exif_data:
                has_camera_hardware = True
                camera_make = str(exif_data.get("Make", ""))
                camera_model = str(exif_data.get("Model", ""))
    except Exception:
        pass

    return {
        "has_exif": len(exif_data) > 0,
        "editing_software_detected": software_flag,
        "software_name": software_name,
        "has_camera_hardware": has_camera_hardware,
        "camera_make": camera_make,
        "camera_model": camera_model,
        "exif_summary": {k: exif_data[k] for k in list(exif_data.keys())[:6]}
    }

def extract_image_features(image_path_or_obj, is_video_frame: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract a content-invariant, physically grounded 14-dimensional forensic descriptor.
    
    Features:
      0: ELA Mean Error (Q=90) / 20.0
      1: ELA Max Patch Error (Q=90) / 40.0
      2: ELA Patch Contrast Ratio / 3.0
      3: ELA Mean Error (Q=75) / 25.0
      4: Block Noise Variance Inconsistency (0.0 to 1.0)
      5: Noise Residual Excess Kurtosis / 5.0
      6: Laplacian Sharpness / 10.0
      7: Edge Gradient Variance / 500.0
      8: Face Detected Flag (0.0 or 1.0)
      9: Face Boundary Seam Discontinuity / 3.0
      10: Face-to-Context Noise Discrepancy / 3.0
      11: Editing Software Signature Present (0.0 or 1.0)
      12: Confirmed Camera Hardware EXIF (0.0 or 1.0)
      13: CFA High-Pass Residual Ratio (0.0 to 1.0)
    """
    if isinstance(image_path_or_obj, (str, Path)):
        img = Image.open(image_path_or_obj).convert("RGB")
    else:
        img = image_path_or_obj.convert("RGB")

    # Standard scale for consistent spatial analysis
    img_resized = img.resize((384, 384), Image.Resampling.LANCZOS)
    
    # 1. Multi-scale Patch ELA
    ela_dict = compute_multi_scale_ela(img_resized)
    
    # 2. Block Noise Inconsistency & Kurtosis
    noise_inc = compute_noise_inconsistency(img_resized)
    noise_kurt = compute_noise_kurtosis(img_resized)
    
    # 3. High-Pass Laplacian Sharpness & Edge Profiles
    gray_np = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray_np, cv2.CV_32F)
    sharpness = float(np.var(lap)) / 1000.0
    
    sobelx = cv2.Sobel(gray_np, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_np, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    edge_gradient_var = float(np.var(grad_mag))
    
    # 4. Facial Manipulation & Deepfake Boundary Seams
    face_signals = extract_facial_manipulation_signals(img_resized)
    
    # 5. Metadata & Provenance
    metadata_info = extract_metadata_forensics(img)
    software_val = 1.0 if metadata_info["editing_software_detected"] else 0.0
    camera_val = 1.0 if metadata_info["has_camera_hardware"] else 0.0
    
    # 6. CFA Chromatic Difference Gradient
    arr = np.asarray(img_resized, dtype=np.float32)
    diff_gr = arr[:, :, 1] - arr[:, :, 0]
    cfa_var = float(np.var(cv2.Laplacian(diff_gr, cv2.CV_32F))) / 500.0

    # 14-dimensional normalized forensic feature vector (content/color-invariant)
    feature_vector = np.array([
        min(1.0, ela_dict["ela_mean_90"] / 20.0),
        min(1.0, ela_dict["ela_max_patch_90"] / 40.0),
        min(1.0, ela_dict["ela_contrast_90"] / 3.0),
        min(1.0, ela_dict["ela_mean_75"] / 25.0),
        float(noise_inc),
        min(1.0, max(-0.5, noise_kurt / 5.0)),
        min(1.0, sharpness / 10.0),
        min(1.0, edge_gradient_var / 500.0),
        float(face_signals["face_detected"]),
        min(1.0, face_signals["boundary_seam_discontinuity"] / 3.0),
        min(1.0, abs(face_signals["face_noise_ratio"] - 1.0) / 2.0),
        float(software_val),
        float(camera_val),
        min(1.0, cfa_var)
    ], dtype=np.float32)

    signals = {
        "ela_error_score": ela_dict["ela_mean_90"],
        "ela_max_patch": ela_dict["ela_max_patch_90"],
        "ela_contrast": ela_dict["ela_contrast_90"],
        "noise_inconsistency": noise_inc,
        "noise_kurtosis": noise_kurt,
        "sharpness_index": round(sharpness, 3),
        "face_forensics": face_signals,
        "metadata_forensics": metadata_info
    }

    return feature_vector, signals
