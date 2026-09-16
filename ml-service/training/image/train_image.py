import os
import sys
import gc
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.image.forensic_extractor import extract_image_features

CASIA_DIR = Path(r"C:\Users\priya\Downloads\CASIA2")
CASIA_CSV = Path(r"C:\Users\priya\Downloads\ml-service-efficientnet-upgrade\ml-service\data\research\casia_subset.csv")
CASIA_IMG_DIR = Path(r"C:\Users\priya\Downloads\ml-service-efficientnet-upgrade\ml-service\data\research\casia_subset")
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "image"
EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation"

def train_image_model(max_samples: int = 300):
    print("=" * 60)
    print("TRAINING IMAGE FORENSIC MANIPULATION CLASSIFIER (CASIA 2.0)")
    print("=" * 60)

    valid_records = []
    # 1. First priority: Check full CASIA2 dataset
    au_dir = CASIA_DIR / "Au"
    tp_dir = CASIA_DIR / "Tp"
    if au_dir.exists() and tp_dir.exists():
        print(f"Loading image samples directly from {CASIA_DIR}...")
        au_files = list(au_dir.glob("*.jpg")) + list(au_dir.glob("*.tif"))
        tp_files = list(tp_dir.glob("*.jpg")) + list(tp_dir.glob("*.tif"))
        print(f"Found {len(au_files)} authentic images and {len(tp_files)} tampered images.")
        
        # Sample balanced subset
        half = max_samples // 2 if max_samples else 150
        import random
        random.seed(42)
        sampled_au = random.sample(au_files, min(half, len(au_files)))
        sampled_tp = random.sample(tp_files, min(half, len(tp_files)))

        for f in sampled_au:
            valid_records.append({"path": str(f), "label": "real"})
        for f in sampled_tp:
            valid_records.append({"path": str(f), "label": "fake"})
    elif CASIA_CSV.exists() and CASIA_IMG_DIR.exists():
        print(f"Loading CASIA dataset metadata from {CASIA_CSV}...")
        df = pd.read_csv(CASIA_CSV).fillna("")
        for _, row in df.iterrows():
            raw_path = Path(row["path"])
            img_file = CASIA_IMG_DIR / raw_path.name
            if img_file.exists():
                valid_records.append({
                    "path": str(img_file),
                    "label": str(row["label"]).lower().strip()
                })
    else:
        print(f"Error: Neither {CASIA_DIR} nor {CASIA_IMG_DIR} found.")
        sys.exit(1)

    print(f"Total curated training candidate images: {len(valid_records)}")
    valid_df = pd.DataFrame(valid_records).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Subsample balanced fake and real records
    if max_samples and max_samples < len(valid_df):
        sub_dfs = []
        for lbl in ["fake", "real"]:
            sub = valid_df[valid_df["label"] == lbl]
            sub_dfs.append(sub.sample(min(len(sub), max_samples // 2), random_state=42))
        valid_df = pd.concat(sub_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Extracting forensic features from {len(valid_df)} images...")
    X_list = []
    y_list = []
    for idx, row in valid_df.iterrows():
        try:
            feats, _ = extract_image_features(row["path"])
            X_list.append(feats)
            y_list.append(row["label"])
        except Exception:
            continue
        if (idx + 1) % 40 == 0:
            print(f"  Processed {idx + 1}/{len(valid_df)} images...")
            gc.collect()

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Extracted feature matrix shape: {X.shape}, labels: {len(y)}")

    # Split into Train (70%), Val (15%), Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, stratify=y_train_val, random_state=42
    )

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Held-out Test: {len(X_test)}")

    # Train Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=60,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )
    rf_model.fit(X_train, y_train)

    val_preds = rf_model.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")

    # Evaluate on Unseen Held-out Test Set
    test_preds = rf_model.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_preds))
    report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, test_preds, labels=["fake", "real"]).tolist()

    print(f"\nUnseen Held-out Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Macro F1-score: {report['macro avg']['f1-score']:.4f}")
    for c in ["fake", "real"]:
        if c in report:
            print(f"  {c.upper():8s}: Precision={report[c]['precision']:.4f}, Recall={report[c]['recall']:.4f}, F1={report[c]['f1-score']:.4f}")

    # Save model and metrics
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "image_forensic_model.joblib"
    joblib.dump({
        "model": rf_model,
        "classes": list(rf_model.classes_),
        "feature_dim": X.shape[1],
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }, model_path)

    metrics_path = OUTPUT_DIR / "image_metrics.json"
    metrics_data = {
        "model_name": "RandomForest Forensics (ELA + Noise Inconsistency + Color + Metadata)",
        "dataset": "CASIA 2.0 Splicing Detection Benchmark",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "validation_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "confusion_matrix": cm,
        "classification_report": report
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nSaved image forensic model to: {model_path}")
    print(f"Saved image metrics to: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    train_image_model(max_samples=args.samples)
