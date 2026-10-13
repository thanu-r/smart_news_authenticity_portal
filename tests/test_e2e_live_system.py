import io
import cv2
import numpy as np
from PIL import Image
import requests
import json
import uuid

BASE_URL = "http://127.0.0.1:8000"

def run_e2e_tests():
    print("================================================================")
    print("STARTING END-TO-END SYSTEM INTEGRATION TESTS")
    print(f"Target Server: {BASE_URL}")
    print("================================================================")

    # 1. Health Check
    print("\n[1] Health Check:")
    res = requests.get(f"{BASE_URL}/api/health", timeout=5)
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    print(f" -> Status: {health_data['status']}, DB Connected: {health_data['database_connected']}")

    # 2. User Registration and Login Flow
    print("\n[2] User Authentication:")
    test_username = f"researcher_{uuid.uuid4().hex[:6]}"
    test_email = f"{test_username}@smartnews.org"
    test_password = "SecurePassword123!"

    reg_res = requests.post(
        f"{BASE_URL}/api/auth/register",
        json={"username": test_username, "email": test_email, "password": test_password, "full_name": "AI Verification Researcher"},
        timeout=5
    )
    assert reg_res.status_code == 201, f"Registration failed: {reg_res.text}"
    token = reg_res.json()["access_token"]
    print(f" -> User registered: {test_username} (Token received: {token[:16]}...)")

    login_res = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"email": test_email, "password": test_password},
        timeout=5
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    auth_headers = {"Authorization": f"Bearer {token}"}
    print(f" -> User login successful: role={login_res.json()['user']['role']}")

    # 3. Text Verification (Transformer Model)
    print("\n[3] Text Verification (BERT-Tiny Transformer):")
    real_text = (
        "Astronomers utilizing the James Webb Space Telescope have detected definitive spectral signatures "
        "of atmospheric water vapor on a rocky exoplanet orbiting within the habitable zone of a nearby red dwarf star. "
        "The observations were confirmed across three distinct transit spectroscopy passes, corroborating prior theoretical models "
        "developed by international astrophysics consortia."
    )
    text_res = requests.post(
        f"{BASE_URL}/api/verify/text",
        headers=auth_headers,
        json={
            "title": "NASA James Webb Telescope Discovers Atmospheric Water Vapor on Habitable-Zone Exoplanet",
            "text": real_text,
            "model_type": "transformer"
        },
        timeout=10
    )
    assert text_res.status_code == 200, f"Text verification failed: {text_res.text}"
    text_data = text_res.json()
    print(f" -> Prediction: {text_data['prediction']}, Calibrated Confidence: {text_data['calibrated_confidence']:.2%}")
    print(f" -> Probabilities: {text_data['probabilities']}")
    print(f" -> Explanation: {text_data['explanation']['summary'][:100]}...")

    # 4. Text Verification (Baseline Model)
    print("\n[4] Text Verification (TF-IDF + Logistic Regression Baseline):")
    fake_text = (
        "BREAKING EXCLUSIVE: Leaked whistleblower documents confirm an underground reptilian humanoid cloning facility "
        "operates directly beneath the Washington Monument, powered by secret crystalline zero-point energy grids hidden since 1962."
    )
    base_res = requests.post(
        f"{BASE_URL}/api/verify/text",
        headers=auth_headers,
        json={
            "title": "Secret Alien Bio-Lab Under Washington Monument",
            "text": fake_text,
            "model_type": "baseline"
        },
        timeout=10
    )
    assert base_res.status_code == 200, f"Baseline verification failed: {base_res.text}"
    base_data = base_res.json()
    print(f" -> Prediction: {base_data['prediction']}, Calibrated Confidence: {base_data['calibrated_confidence']:.2%}")
    print(f" -> Model used: {base_data.get('text_analysis', {}).get('model')}")

    # 5. Image Verification (Natural Image)
    print("\n[5] Image Verification (Natural Synthetic Optical Image):")
    # Generate natural optical pattern with smooth gradient
    img_arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            img_arr[y, x] = [int(x * 0.7), int(y * 0.7), int((x + y) * 0.35)]
    # Add subtle optical sensor noise
    noise = np.random.normal(0, 3, (256, 256, 3)).astype(np.int16)
    img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img_arr)
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="JPEG", quality=90)
    img_bytes.seek(0)

    img_res = requests.post(
        f"{BASE_URL}/api/verify/image",
        headers=auth_headers,
        files={"file": ("natural_photo.jpg", img_bytes.getvalue(), "image/jpeg")},
        data={"title": "Natural Test Image"}
    )
    assert img_res.status_code == 200, f"Image verification failed: {img_res.text}"
    img_data = img_res.json()
    print(f" -> Natural Image Prediction: {img_data['prediction']}, Confidence: {img_data['confidence']:.2%}")
    print(f" -> ELA Residual: {img_data.get('image_forensics', {}).get('forensic_signals', {}).get('ela_error_score')}")

    # 6. Image Verification (AI-Generated Image with Diffusion Markers)
    print("\n[6] Image Verification (AI-Generated Image with Metadata & Frequency Markers):")
    ai_img = Image.fromarray(img_arr)
    # Add AI generation parameters into PNG text info
    ai_png_bytes = io.BytesIO()
    from PIL.PngImagePlugin import PngInfo
    png_info = PngInfo()
    png_info.add_text("parameters", "photorealistic portrait, 8k, highly detailed\nSteps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Model: SDXL_v1.0")
    ai_img.save(ai_png_bytes, format="PNG", pnginfo=png_info)
    ai_png_bytes.seek(0)

    ai_res = requests.post(
        f"{BASE_URL}/api/verify/image",
        headers=auth_headers,
        files={"file": ("diffusion_render.png", ai_png_bytes.getvalue(), "image/png")},
        data={"title": "Synthetic Portrait Render"}
    )
    assert ai_res.status_code == 200, f"AI image verification failed: {ai_res.text}"
    ai_data = ai_res.json()
    print(f" -> AI Image Prediction: {ai_data['prediction']}, Confidence: {ai_data['confidence']:.2%}")
    assert ai_data['prediction'] != 'REAL', "AI Generated image was incorrectly classified as REAL!"
    print(f" -> Verified: AI image is NOT classified as REAL (Verdict: {ai_data['prediction']})")

    # 7. Video Verification (Keyframe & Temporal Flicker Analysis)
    print("\n[7] Video Verification (Keyframe Sampling & Temporal Consistency):")
    # Synthesize clean 2-second MP4 video in memory
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_vid_path = "tests/test_temp_clip.mp4"
    out = cv2.VideoWriter(temp_vid_path, fourcc, 10.0, (160, 160))
    for i in range(20):
        frame = np.full((160, 160, 3), (i * 10) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()

    with open(temp_vid_path, "rb") as vf:
        vid_res = requests.post(
            f"{BASE_URL}/api/verify/video",
            headers=auth_headers,
            files={"file": ("temporal_test.mp4", vf.read(), "video/mp4")},
            data={"title": "News Video Segment"}
        )
    assert vid_res.status_code == 200, f"Video verification failed: {vid_res.text}"
    vid_data = vid_res.json()
    print(f" -> Video Prediction: {vid_data['prediction']}, Confidence: {vid_data['confidence']:.2%}")
    print(f" -> Temporal Flicker Index: {vid_data.get('video_forensics', {}).get('temporal_flicker')}")
    print(f" -> Frames Analyzed: {vid_data.get('video_forensics', {}).get('frames_analyzed')}")

    # 8. URL Verification
    print("\n[8] URL Verification & SSRF Security Defense:")
    # Test SSRF block on internal loopback
    ssrf_res = requests.post(
        f"{BASE_URL}/api/verify/url",
        headers=auth_headers,
        json={"url": "http://127.0.0.1:8000/api/admin/metrics"}
    )
    assert ssrf_res.status_code == 400, f"SSRF attack was not blocked! Code: {ssrf_res.status_code}"
    print(f" -> SSRF Attack to internal network successfully blocked: HTTP 400 ('{ssrf_res.json()['detail']}')")

    # Test invalid scheme
    invalid_scheme_res = requests.post(
        f"{BASE_URL}/api/verify/url",
        headers=auth_headers,
        json={"url": "ftp://malicious-ftp-server.xyz/article"}
    )
    assert invalid_scheme_res.status_code == 400, "Invalid protocol was not rejected!"
    print(" -> Invalid protocol scheme rejected properly (HTTP 400)")

    # 9. Multimodal Fusion Verification
    print("\n[9] Multimodal Fusion Verification:")
    multi_res = requests.post(
        f"{BASE_URL}/api/verify/multimodal",
        headers=auth_headers,
        data={
            "title": "NASA James Webb Space Telescope Update",
            "text": real_text,
            "url": "https://www.reuters.com/science/space-discovery-webb"
        },
        files={"image": ("space_photo.jpg", img_bytes.getvalue(), "image/jpeg")}
    )
    assert multi_res.status_code == 200, f"Multimodal verification failed: {multi_res.text}"
    multi_data = multi_res.json()
    print(f" -> Multimodal Prediction: {multi_data['prediction']}, Calibrated Score: {multi_data['calibrated_confidence']:.2%}")
    print(f" -> Modalities Analyzed: {multi_data['modalities_analyzed']}")
    print(f" -> Cross-Modal Consistency: {multi_data.get('cross_modal_consistency', {}).get('headline_body_similarity')}")

    # 10. Verification History & User Feedback
    print("\n[10] Verification History & Human-in-the-Loop Feedback:")
    hist_res = requests.get(f"{BASE_URL}/api/history", headers=auth_headers)
    assert hist_res.status_code == 200, f"History fetch failed: {hist_res.text}"
    history_records = hist_res.json()["records"]
    assert len(history_records) > 0, "No verification records found in history!"
    target_record_id = history_records[0]["id"]
    print(f" -> Retrieved {len(history_records)} historical records from MongoDB.")
    print(f" -> Submitting feedback for record '{target_record_id}'...")

    fb_res = requests.post(
        f"{BASE_URL}/api/history/{target_record_id}/feedback",
        headers=auth_headers,
        json={"is_correct": True, "comments": "Accurate prediction verified by auditor."}
    )
    assert fb_res.status_code == 200, f"Feedback submission failed: {fb_res.text}"
    print(f" -> Feedback recorded successfully: {fb_res.json()['status']}")

    # 11. Live News Feeds & Search
    print("\n[11] Live News Feeds & Search:")
    news_res = requests.get(f"{BASE_URL}/api/news/latest?limit=5", headers=auth_headers)
    assert news_res.status_code == 200, f"News fetch failed: {news_res.text}"
    news_articles = news_res.json().get("articles", [])
    print(f" -> Ingested news feed count: {len(news_articles)} articles")

    search_res = requests.get(f"{BASE_URL}/api/news/search?q=Space", headers=auth_headers)
    assert search_res.status_code == 200, f"News search failed: {search_res.text}"
    print(f" -> News search returned HTTP 200")

    # 12. Model Metrics & Analytics
    print("\n[12] Model Metrics & Analytics:")
    metrics_res = requests.get(f"{BASE_URL}/api/models/metrics", headers=auth_headers)
    assert metrics_res.status_code == 200, f"Metrics fetch failed: {metrics_res.text}"
    metrics = metrics_res.json()
    print(f" -> Baseline Text Accuracy: {metrics['baseline_text']['test_accuracy']:.2%}")
    print(f" -> Transformer Suspicious Recall: {metrics['transformer_text']['suspicious_recall']:.2%}")
    print(f" -> Visual Forensics Accuracy: {metrics['image_forensics']['test_accuracy']:.2%}")
    print(f" -> Video Deepfake Accuracy: {metrics['video_deepfake']['test_accuracy']:.2%}")

    print("\n================================================================")
    print("ALL 12/12 END-TO-END SUBSYSTEM TESTS PASSED WITH ZERO ERRORS!")
    print("================================================================")

if __name__ == "__main__":
    run_e2e_tests()
